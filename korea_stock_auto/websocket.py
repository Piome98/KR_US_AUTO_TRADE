"""
한국 주식 자동매매 - 웹소켓 모듈
실시간 시세 데이터 수신 처리
"""

import json
import threading
import time
import queue
from websocket import WebSocketApp

from korea_stock_auto.config import WS_URL
from korea_stock_auto.utils import send_message
from korea_stock_auto.auth import request_ws_connection_key

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
        
    def set_symbol_list(self, symbols):
        """
        관심 종목 설정
        
        Args:
            symbols (list): 관심 종목 코드 리스트
        """
        self.symbol_list = symbols
        self.target_buy_price = {sym: None for sym in symbols}
        self.target_sell_price = {sym: None for sym in symbols}
        self.holding_stock = {sym: 0 for sym in symbols}
        send_message(f"관심 종목 설정 완료: {symbols}")
    
    def _on_message(self, ws, message):
        """
        웹소켓 메시지 수신 처리 함수
        
        Args:
            ws: 웹소켓 객체
            message (str): 수신된 메시지
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            send_message("JSON 파싱 오류 발생")
            return
        
        if "body" in data and "output" in data["body"]:
            output = data["body"]["output"]
            
            for item in output:
                code = item.get("MKSC_SHRN_ISCD")
                try:
                    price = int(item.get("STCK_PRPR", 0))
                except Exception:
                    price = None
                
                # 유효한 가격 데이터인 경우만 처리
                if code and price is not None:
                    send_message(f"[{code}] 현재가: {price}원")
                    self.price_queue.put((code, price))
    
    def _on_error(self, ws, error):
        """
        웹소켓 오류 처리 함수
        
        Args:
            ws: 웹소켓 객체
            error: 발생한 오류
        """
        send_message(f"웹소켓 오류: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """
        웹소켓 연결 종료 처리 함수
        
        Args:
            ws: 웹소켓 객체
            close_status_code: 종료 상태 코드
            close_msg: 종료 메시지
        """
        send_message("웹소켓 연결 종료")
    
    def _on_open(self, ws):
        """
        웹소켓 연결 시 구독 요청 함수
        
        Args:
            ws: 웹소켓 객체
        """
        if not self.ws_conn_key:
            send_message("WebSocket 접속키 없음. 웹소켓 구독 요청 중단.")
            return
        
        if not self.symbol_list:
            send_message("관심 종목 없음. 웹소켓 구독 요청 중단.")
            return
        
        subscribe_data = {
            "header": {
                "approval_key": self.ws_conn_key,
                "custtype": "P",
                "tr_type": "1",
                "content-type": "utf-8"
            },
            "body": {
                "input": [{"tr_id": "H0STCNT0", "tr_key": sym} for sym in self.symbol_list]
            }
        }
        ws.send(json.dumps(subscribe_data))
        send_message("웹소켓 구독 요청 완료")
    
    def start_websocket(self):
        """웹소켓 연결 및 재연결 관리 함수"""
        while True:
            if not self.symbol_list:
                send_message("관심 종목 없음. 웹소켓 구독 요청 진행 중단.")
                return
            
            if not self.ws_conn_key:
                send_message("WebSocket 접속키 없음. 웹소켓 실행 중단.")
                return
            
            send_message(f"웹소켓 구독 요청 중: {self.symbol_list}")
            
            try:
                self.ws = WebSocketApp(
                    WS_URL,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.ws.on_open = self._on_open
                # 연결 유지: ping_interval 및 ping_timeout 추가
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                send_message(f"WebSocket 실행 오류: {e}")
            
            send_message("웹소켓 연결이 종료되었습니다. 10초 후 재연결 시도합니다.")
            time.sleep(10)
    
    def start(self):
        """
        웹소켓 스레드 시작 함수
        
        Returns:
            bool: 시작 성공 여부
        """
        if not self.symbol_list:
            send_message("관심 종목이 설정되지 않았습니다. 웹소켓을 시작할 수 없습니다.")
            return False
        
        # 웹소켓 접속키 요청
        self.ws_conn_key = request_ws_connection_key()
        if not self.ws_conn_key:
            send_message("웹소켓 접속키 발급에 실패했습니다.")
            return False
        
        # 웹소켓 스레드 시작
        thread = threading.Thread(target=self.start_websocket)
        thread.daemon = True
        thread.start()
        send_message("실시간 웹소켓 스레드 시작")
        return True
    
    def get_price(self, code, timeout=10):
        """
        특정 종목의 실시간 가격 조회
        
        Args:
            code (str): 종목 코드
            timeout (int): 타임아웃 시간(초)
            
        Returns:
            int or None: 현재가 또는 조회 실패시 None
        """
        try:
            # 큐 내의 모든 항목을 확인
            items = []
            while not self.price_queue.empty():
                items.append(self.price_queue.get(timeout=timeout))
            
            # 가장 최근 값을 찾기 위해 역순으로 검색
            for received_code, price in reversed(items):
                if received_code == code:
                    # 처리한 항목 외에는 다시 큐에 넣기
                    for item in items:
                        if item[0] != code:
                            self.price_queue.put(item)
                    return price
            
            # 찾지 못한 항목은 다시 큐에 넣기
            for item in items:
                self.price_queue.put(item)
            
            return None
            
        except queue.Empty:
            send_message(f"{code}의 실시간 시세 데이터를 수신하지 못함.")
            return None 