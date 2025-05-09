"""
한국 주식 자동매매 - 웹소켓 모듈
실시간 시세 데이터 수신 처리
"""

import json
import threading
import time
import queue
import datetime
import logging
from typing import List, Dict, Any, Optional, Callable, Set, Union
from websocket import WebSocketApp

from korea_stock_auto.config import WS_URL, APP_KEY, APP_SECRET, is_prod_env
from korea_stock_auto.utils.utils import send_message, rate_limit_wait, retry_on_failure
from korea_stock_auto.api.auth import request_ws_connection_key

# 로깅 설정
logger = logging.getLogger(__name__)

# 실시간 TR 코드
REAL_TYPE = {
    "주식체결": "H1_",  # 주식 체결
    "호가발생": "H2_",  # 호가 발생
    "주식우선호가": "O23",  # 주식 우선호가 
    "주식당일거래원": "K1_",  # 주식 당일 거래원
    "ETF호가": "HB_",  # ETF 호가
    "ETF체결": "HA_",  # ETF 체결
    "ETF NAV": "I5",  # ETF NAV
    "지수": "BM_",  # 지수 (KOSPI, KOSDAQ)
}

class StockWebSocket:
    """실시간 주식 시세 웹소켓 클래스"""
    
    def __init__(self):
        """웹소켓 클래스 초기화"""
        self.ws_conn_key = None
        self.ws = None
        self.price_queue = queue.Queue()
        self.symbol_list = []
        self.target_buy_price = {}
        self.target_sell_price = {}
        self.holding_stock = {}
        self.last_price = {}  # 각 종목별 최근 수신 가격 저장
        self.connection_retry_count = 0
        self.max_connection_retries = 10
        self.reconnect_interval = 5  # 초기 재연결 간격 (초)
        self.max_reconnect_interval = 300  # 최대 재연결 간격 (5분)
        self.last_ping_time = 0
        self.ping_interval = 55  # 핑 전송 간격 (초)
        self.subscribed_symbols = set()  # 구독 중인 종목 목록
        self.ws_connected = False
        self.connection_lock = threading.Lock()
        self.heartbeat_thread = None
        self.running = False
        
    def start(self, symbol_list: Optional[List[str]] = None):
        """
        웹소켓 연결 시작
        
        Args:
            symbol_list (list): 실시간 조회할 종목 코드 리스트
        """
        if symbol_list:
            self.symbol_list = symbol_list
            
        self.running = True
        
        # 웹소켓 접속키 발급
        if not self.ws_conn_key:
            self.ws_conn_key = request_ws_connection_key()
            if not self.ws_conn_key:
                error_msg = "웹소켓 접속키 발급 실패, 웹소켓 연결을 시작할 수 없습니다."
                logger.error(error_msg)
                send_message(error_msg)
                self.running = False
                return
        
        # 웹소켓 연결 시작
        self._connect_websocket()
        
        # 하트비트 스레드 시작 (웹소켓 연결 유지)
        if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_thread, daemon=True)
            self.heartbeat_thread.start()
            logger.info("하트비트 스레드 시작됨")
        
        # 종목 구독
        if self.symbol_list and self.ws_connected:
            self.subscribe_symbols(self.symbol_list)
            
    def _connect_websocket(self):
        """웹소켓 연결 설정"""
        with self.connection_lock:
            if self.ws_connected:
                logger.debug("이미 웹소켓에 연결됨")
                return
            
            if not self.ws_conn_key:
                self.ws_conn_key = request_ws_connection_key()
                if not self.ws_conn_key:
                    error_msg = "웹소켓 접속키 발급 실패"
                    logger.error(error_msg)
                    send_message(error_msg)
                    return
            
            try:
                url = f"{WS_URL}/tryitout/H0STCNT0"
                
                # WebSocket 객체 생성 및 콜백 함수 등록
                self.ws = WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_ping=self._on_ping,
                    on_pong=self._on_pong
                )
                
                # WebSocket 연결 스레드 시작
                ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
                ws_thread.start()
                logger.info("웹소켓 연결 스레드 시작됨")
                
                # 연결 완료 대기
                wait_time = 0
                max_wait = 10  # 최대 10초 대기
                
                while not self.ws_connected and wait_time < max_wait:
                    time.sleep(0.5)
                    wait_time += 0.5
                
                if not self.ws_connected and wait_time >= max_wait:
                    error_msg = f"웹소켓 연결 시간 초과 (대기시간: {wait_time}초)"
                    logger.error(error_msg)
                    send_message(error_msg)
                    
            except Exception as e:
                error_msg = f"웹소켓 연결 중 예외 발생: {e}"
                logger.error(error_msg)
                send_message(error_msg)
                self.reconnect()
    
    def _on_open(self, ws):
        """
        웹소켓 연결 성공 콜백
        
        Args:
            ws: 웹소켓 객체
        """
        try:
            # 웹소켓 접속 요청 전문 생성
            tr_id = "H0STCNT0"
            header = {
                "approval_key": self.ws_conn_key,
                "custtype": "P",
                "tr_type": "1",
                "content-type": "utf-8"
            }
            
            body = {
                "input": {
                    "tr_id": tr_id,
                    "tr_key": " "
                }
            }
            
            # 웹소켓 접속 요청 전송
            ws_request = {
                "header": header,
                "body": body
            }
            
            ws.send(json.dumps(ws_request))
            logger.info("웹소켓 접속 요청 전송 완료")
            
            self.ws_connected = True
            self.connection_retry_count = 0
            self.last_ping_time = time.time()
            
            send_message("웹소켓 연결 성공")
            logger.info("웹소켓 연결 성공")
            
            # 기존에 구독하던 종목이 있으면 재구독
            if self.subscribed_symbols:
                symbols_to_resubscribe = list(self.subscribed_symbols)
                logger.info(f"{len(symbols_to_resubscribe)}개 종목 재구독 시도")
                time.sleep(1)  # 잠시 대기 후 구독 시도
                self.subscribe_symbols(symbols_to_resubscribe)
        
        except Exception as e:
            error_msg = f"웹소켓 연결 설정 중 예외 발생: {e}"
            logger.error(error_msg)
            send_message(error_msg)
    
    def _on_message(self, ws, message):
        """
        웹소켓 메시지 수신 콜백
        
        Args:
            ws: 웹소켓 객체
            message: 수신한 메시지
        """
        try:
            data = json.loads(message)
            
            # 로그인 응답 처리
            if "header" in data and "body" in data:
                header = data["header"]
                body = data["body"]
                
                # 로그인 응답
                if header.get("tr_id") == "H0STCNT0":
                    rt_cd = header.get("rt_cd")
                    if rt_cd == "0":
                        logger.info("웹소켓 로그인 성공")
                    else:
                        error_msg = f"웹소켓 로그인 실패: {header.get('msg1', '알 수 없는 오류')}"
                        logger.error(error_msg)
                        send_message(error_msg)
                        
                        # 재연결 시도
                        self.ws_connected = False
                        self.reconnect()
                        return
                        
                # 실시간 시세 응답
                elif "real_time" in header:
                    tr_type = body.get("header", {}).get("tr_id", "")
                    
                    # 실시간 체결 데이터
                    if tr_type.startswith(REAL_TYPE["주식체결"]):
                        stock_code = body.get("body", {}).get("mksc_shrn_iscd", "")
                        price = body.get("body", {}).get("stck_prpr", "")
                        
                        if stock_code and price:
                            try:
                                price_int = int(price)
                                self.last_price[stock_code] = price_int
                                
                                # 큐에 데이터 추가
                                price_data = {
                                    "code": stock_code,
                                    "price": price_int,
                                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                                }
                                self.price_queue.put(price_data)
                                
                                # 매수/매도 조건 확인 로직은 별도 스레드에서 처리
                            except ValueError:
                                logger.warning(f"가격 변환 실패: {price} (종목코드: {stock_code})")
                
                # 핑퐁 처리
                elif header.get("tr_id") == "PINGPONG":
                    # 클라이언트가 보낸 핑에 대한 서버의 퐁 응답
                    self.last_ping_time = time.time()
                    logger.debug("핑퐁 응답 받음")
        
        except json.JSONDecodeError:
            logger.warning("잘못된 JSON 형식의 메시지 수신")
        except Exception as e:
            error_msg = f"메시지 처리 중 예외 발생: {e}"
            logger.error(error_msg)
    
    def _on_error(self, ws, error):
        """
        웹소켓 오류 콜백
        
        Args:
            ws: 웹소켓 객체
            error: 발생한 오류
        """
        error_msg = f"웹소켓 오류: {error}"
        logger.error(error_msg)
        send_message(error_msg)
        
        # 웹소켓 연결 실패로 설정
        self.ws_connected = False
        
        # 재연결 시도
        self.reconnect()
    
    def _on_close(self, ws, close_status_code, close_reason):
        """
        웹소켓 연결 종료 콜백
        
        Args:
            ws: 웹소켓 객체
            close_status_code: 종료 상태 코드
            close_reason: 종료 이유
        """
        close_msg = f"웹소켓 연결 종료 (코드: {close_status_code}, 이유: {close_reason})"
        logger.info(close_msg)
        send_message(close_msg)
        
        # 웹소켓 연결 종료로 설정
        self.ws_connected = False
        
        # 실행 중이면 재연결 시도
        if self.running:
            self.reconnect()
    
    def _on_ping(self, ws, message):
        """
        핑 수신 콜백 (서버가 핑을 보낼 경우)
        
        Args:
            ws: 웹소켓 객체
            message: 핑 메시지
        """
        logger.debug("핑 수신")
        # 서버로부터 핑을 받으면 퐁 응답
        self._send_pong(ws)
    
    def _on_pong(self, ws, message):
        """
        퐁 수신 콜백 (서버가 퐁을 보낼 경우)
        
        Args:
            ws: 웹소켓 객체
            message: 퐁 메시지
        """
        logger.debug("퐁 수신")
        self.last_ping_time = time.time()
    
    def _send_ping(self, ws):
        """
        핑 전송 함수
        
        Args:
            ws: 웹소켓 객체
        """
        if ws and ws.sock and ws.sock.connected:
            try:
                ping_data = {
                    "header": {
                        "tr_id": "PINGPONG"
                    },
                    "body": {
                        "msg": "PING"
                    }
                }
                ws.send(json.dumps(ping_data))
                logger.debug("핑 전송함")
            except Exception as e:
                logger.error(f"핑 전송 중 예외 발생: {e}")
    
    def _send_pong(self, ws):
        """
        퐁 전송 함수
        
        Args:
            ws: 웹소켓 객체
        """
        if ws and ws.sock and ws.sock.connected:
            try:
                pong_data = {
                    "header": {
                        "tr_id": "PINGPONG"
                    },
                    "body": {
                        "msg": "PONG"
                    }
                }
                ws.send(json.dumps(pong_data))
                logger.debug("퐁 전송함")
            except Exception as e:
                logger.error(f"퐁 전송 중 예외 발생: {e}")
    
    def _heartbeat_thread(self):
        """하트비트(Ping) 스레드 함수"""
        while self.running:
            try:
                current_time = time.time()
                
                # 마지막 핑 전송 후 일정 시간 경과했으면 핑 전송
                if self.ws_connected and (current_time - self.last_ping_time) > self.ping_interval:
                    self._send_ping(self.ws)
                    self.last_ping_time = current_time
                
                # 연결 상태 확인
                if self.ws and not self.ws_connected:
                    logger.warning("하트비트 스레드: 웹소켓 연결 끊김 감지")
                    self.reconnect()
                
                # 1초마다 확인
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"하트비트 스레드에서 예외 발생: {e}")
                time.sleep(5)  # 오류 발생 시 잠시 대기
    
    def reconnect(self):
        """웹소켓 재연결 시도"""
        with self.connection_lock:
            if not self.running:
                logger.info("웹소켓이 중지 상태여서 재연결을 시도하지 않습니다.")
                return
                
            # 기존 연결 종료
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass
                
            self.ws_connected = False
            
            # 재연결 시도 횟수 증가
            self.connection_retry_count += 1
            
            # 지수 백오프로 대기 시간 계산
            wait_time = min(
                self.reconnect_interval * (2 ** (self.connection_retry_count - 1)),
                self.max_reconnect_interval
            )
            
            reconnect_msg = f"웹소켓 재연결 시도 {self.connection_retry_count}/{self.max_connection_retries} ({wait_time}초 후)"
            logger.info(reconnect_msg)
            send_message(reconnect_msg)
            
            # 최대 재시도 횟수를 초과하면 중지
            if self.connection_retry_count > self.max_connection_retries:
                error_msg = f"웹소켓 재연결 최대 시도 횟수 초과 ({self.max_connection_retries}회)"
                logger.error(error_msg)
                send_message(error_msg)
                self.running = False
                return
            
            # 대기 후 재연결
            time.sleep(wait_time)
            
            # 접속키 재발급
            self.ws_conn_key = request_ws_connection_key()
            if not self.ws_conn_key:
                error_msg = "웹소켓 접속키 재발급 실패"
                logger.error(error_msg)
                send_message(error_msg)
                # 다시 재연결 시도
                self.reconnect()
                return
            
            try:
                # 웹소켓 연결 시도
                self._connect_websocket()
            except Exception as e:
                error_msg = f"웹소켓 재연결 중 예외 발생: {e}"
                logger.error(error_msg)
                send_message(error_msg)
                # 다시 재연결 시도
                self.reconnect()
    
    def subscribe_symbols(self, symbols: List[str]):
        """
        종목 구독 요청
        
        Args:
            symbols (list): 구독할 종목 코드 목록
        """
        if not self.ws_connected:
            logger.warning("웹소켓 연결이 되어있지 않아 종목 구독을 할 수 없습니다.")
            send_message("웹소켓 연결이 되어있지 않아 종목 구독을 할 수 없습니다.")
            return False
        
        if not symbols:
            logger.warning("구독할 종목 목록이 비어있습니다.")
            return False
        
        try:
            # 실시간 시세 요청
            tr_type = REAL_TYPE["주식체결"]
            
            for symbol in symbols:
                # 이미 구독 중인 종목은 건너뛰기
                if symbol in self.subscribed_symbols:
                    logger.debug(f"이미 구독 중인 종목: {symbol}")
                    continue
                
                # 실시간 요청 전문 생성
                header = {
                    "approval_key": self.ws_conn_key,
                    "custtype": "P",
                    "tr_type": "1",
                    "content-type": "utf-8"
                }
                
                body = {
                    "input": {
                        "tr_id": tr_type,
                        "tr_key": symbol
                    }
                }
                
                # 웹소켓 요청 전송
                ws_request = {
                    "header": header,
                    "body": body
                }
                
                self.ws.send(json.dumps(ws_request))
                logger.info(f"종목 구독 요청 전송: {symbol}")
                
                # 구독 목록에 추가
                self.subscribed_symbols.add(symbol)
                
                # API 호출 제한 방지
                rate_limit_wait(0.3)
            
            subscribe_msg = f"{len(symbols)}개 종목 구독 요청 완료"
            logger.info(subscribe_msg)
            send_message(subscribe_msg)
            return True
            
        except Exception as e:
            error_msg = f"종목 구독 요청 중 예외 발생: {e}"
            logger.error(error_msg)
            send_message(error_msg)
            return False
    
    def unsubscribe_symbols(self, symbols: List[str]):
        """
        종목 구독 해지 요청
        
        Args:
            symbols (list): 구독 해지할 종목 코드 목록
        """
        if not self.ws_connected:
            logger.warning("웹소켓 연결이 되어있지 않아 종목 구독 해지를 할 수 없습니다.")
            return False
        
        if not symbols:
            logger.warning("구독 해지할 종목 목록이 비어있습니다.")
            return False
        
        try:
            # 실시간 시세 요청
            tr_type = REAL_TYPE["주식체결"]
            
            for symbol in symbols:
                # 구독 중이지 않은 종목은 건너뛰기
                if symbol not in self.subscribed_symbols:
                    logger.debug(f"구독 중이지 않은 종목: {symbol}")
                    continue
                
                # 실시간 요청 전문 생성
                header = {
                    "approval_key": self.ws_conn_key,
                    "custtype": "P",
                    "tr_type": "2",  # 구독 해지는 tr_type=2
                    "content-type": "utf-8"
                }
                
                body = {
                    "input": {
                        "tr_id": tr_type,
                        "tr_key": symbol
                    }
                }
                
                # 웹소켓 요청 전송
                ws_request = {
                    "header": header,
                    "body": body
                }
                
                self.ws.send(json.dumps(ws_request))
                logger.info(f"종목 구독 해지 요청 전송: {symbol}")
                
                # 구독 목록에서 제거
                self.subscribed_symbols.remove(symbol)
                
                # API 호출 제한 방지
                rate_limit_wait(0.3)
            
            unsubscribe_msg = f"{len(symbols)}개 종목 구독 해지 요청 완료"
            logger.info(unsubscribe_msg)
            send_message(unsubscribe_msg)
            return True
            
        except Exception as e:
            error_msg = f"종목 구독 해지 요청 중 예외 발생: {e}"
            logger.error(error_msg)
            send_message(error_msg)
            return False
    
    def is_connected(self) -> bool:
        """
        웹소켓 연결 상태 확인
        
        Returns:
            bool: 연결 상태
        """
        return self.ws_connected
    
    def get_subscribed_symbols(self) -> List[str]:
        """
        현재 구독 중인 종목 목록 반환
        
        Returns:
            list: 구독 중인 종목 코드 목록
        """
        return list(self.subscribed_symbols)
    
    def stop(self):
        """웹소켓 연결 종료"""
        self.running = False
        
        # 구독 중인 종목 해지
        if self.subscribed_symbols:
            self.unsubscribe_symbols(list(self.subscribed_symbols))
        
        # 웹소켓 종료
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        
        # 웹소켓 연결 상태 초기화
        self.ws_connected = False
        
        logger.info("웹소켓 연결 종료 완료")
        send_message("웹소켓 연결 종료 완료") 