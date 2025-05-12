"""
한국 주식 자동매매 - 웹소켓 데이터 처리 모듈
실시간 시세 데이터 처리 기능 제공
"""

import json
import logging
import datetime
import queue
from typing import Dict, Any, List, Set, Optional, Callable, TYPE_CHECKING

from korea_stock_auto.utils.utils import send_message

# 로깅 설정
logger = logging.getLogger("stock_auto")

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

class DataProcessor:
    """웹소켓 데이터 처리 클래스"""
    
    def __init__(self):
        """데이터 처리기 초기화"""
        self.price_queue = queue.Queue()  # 가격 데이터 큐
        self.last_price = {}  # 각 종목별 최근 수신 가격 저장
        self.target_buy_price = {}  # 매수 목표가
        self.target_sell_price = {}  # 매도 목표가
        self.holding_stock = {}  # 보유 주식 정보
        self.data_handlers: List[Callable[[Dict[str, Any]], None]] = []  # 데이터 처리 핸들러 목록
    
    def add_data_handler(self, handler):
        """
        데이터 처리 핸들러 추가
        
        Args:
            handler: 데이터 처리 핸들러 함수
        """
        if handler not in self.data_handlers:
            self.data_handlers.append(handler)
    
    def remove_data_handler(self, handler):
        """
        데이터 처리 핸들러 제거
        
        Args:
            handler: 제거할 데이터 처리 핸들러 함수
        """
        if handler in self.data_handlers:
            self.data_handlers.remove(handler)
    
    def process_message(self, data: Dict[str, Any]):
        """
        웹소켓 메시지 처리
        
        Args:
            data: 수신한 메시지 데이터
        """
        try:
            # 실시간 시세 응답 처리
            if "header" in data and "body" in data:
                header = data["header"]
                body = data["body"]
                
                # 실시간 시세 응답인 경우
                if "real_time" in header:
                    tr_type = body.get("header", {}).get("tr_id", "")
                    
                    # 실시간 체결 데이터
                    if tr_type.startswith(REAL_TYPE["주식체결"]):
                        self._process_stock_price(body)
                    
                    # 실시간 호가 데이터
                    elif tr_type.startswith(REAL_TYPE["호가발생"]):
                        self._process_stock_quote(body)
                    
                    # 기타 실시간 데이터
                    else:
                        self._process_other_data(tr_type, body)
                        
        except Exception as e:
            logger.error(f"데이터 처리 중 예외 발생: {e}")
    
    def _process_stock_price(self, body: Dict[str, Any]):
        """
        주식 체결 데이터 처리
        
        Args:
            body: 체결 데이터
        """
        try:
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
                    
                    # 등록된 핸들러에 데이터 전달
                    for handler in self.data_handlers:
                        try:
                            handler(price_data)
                        except Exception as e:
                            logger.error(f"핸들러 실행 중 오류 발생: {e}")
                    
                except ValueError:
                    logger.warning(f"가격 변환 실패: {price} (종목코드: {stock_code})")
        
        except Exception as e:
            logger.error(f"주식 체결 데이터 처리 중 오류 발생: {e}")
    
    def _process_stock_quote(self, body: Dict[str, Any]):
        """
        주식 호가 데이터 처리
        
        Args:
            body: 호가 데이터
        """
        try:
            stock_code = body.get("body", {}).get("mksc_shrn_iscd", "")
            if not stock_code:
                return
                
            # 호가 데이터 추출
            ask_price1 = body.get("body", {}).get("askp1", "")
            bid_price1 = body.get("body", {}).get("bidp1", "")
            
            if ask_price1 and bid_price1:
                try:
                    ask_price_int = int(ask_price1)
                    bid_price_int = int(bid_price1)
                    
                    # 호가 데이터 저장 및 처리
                    quote_data = {
                        "code": stock_code,
                        "ask_price": ask_price_int,
                        "bid_price": bid_price_int,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    }
                    
                    # 등록된 핸들러에 데이터 전달
                    for handler in self.data_handlers:
                        try:
                            handler(quote_data)
                        except Exception as e:
                            logger.error(f"핸들러 실행 중 오류 발생: {e}")
                    
                except ValueError:
                    logger.warning(f"호가 변환 실패 (종목코드: {stock_code})")
        
        except Exception as e:
            logger.error(f"주식 호가 데이터 처리 중 오류 발생: {e}")
    
    def _process_other_data(self, tr_type: str, body: Dict[str, Any]):
        """
        기타 실시간 데이터 처리
        
        Args:
            tr_type: 실시간 데이터 유형
            body: 데이터 본문
        """
        # 지수 데이터 처리
        if tr_type.startswith(REAL_TYPE["지수"]):
            self._process_index_data(body)
        
        # ETF NAV 데이터 처리
        elif tr_type == REAL_TYPE["ETF NAV"]:
            self._process_etf_nav_data(body)
    
    def _process_index_data(self, body: Dict[str, Any]):
        """
        지수 데이터 처리
        
        Args:
            body: 지수 데이터
        """
        try:
            index_code = body.get("body", {}).get("bstp_nmix_idcd", "")
            index_name = body.get("body", {}).get("bstp_nmix_nm", "")
            index_value = body.get("body", {}).get("bstp_nmix_prpr", "")
            
            if index_code and index_value:
                try:
                    index_value_float = float(index_value)
                    
                    # 지수 데이터 저장 및 처리
                    index_data = {
                        "code": index_code,
                        "name": index_name,
                        "value": index_value_float,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    }
                    
                    # 등록된 핸들러에 데이터 전달
                    for handler in self.data_handlers:
                        try:
                            handler(index_data)
                        except Exception as e:
                            logger.error(f"핸들러 실행 중 오류 발생: {e}")
                    
                except ValueError:
                    logger.warning(f"지수 값 변환 실패: {index_value} (지수코드: {index_code})")
        
        except Exception as e:
            logger.error(f"지수 데이터 처리 중 오류 발생: {e}")
    
    def _process_etf_nav_data(self, body: Dict[str, Any]):
        """
        ETF NAV 데이터 처리
        
        Args:
            body: ETF NAV 데이터
        """
        try:
            etf_code = body.get("body", {}).get("etf_cd", "")
            nav_value = body.get("body", {}).get("nav", "")
            
            if etf_code and nav_value:
                try:
                    nav_value_float = float(nav_value)
                    
                    # ETF NAV 데이터 저장 및 처리
                    nav_data = {
                        "code": etf_code,
                        "nav": nav_value_float,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    }
                    
                    # 등록된 핸들러에 데이터 전달
                    for handler in self.data_handlers:
                        try:
                            handler(nav_data)
                        except Exception as e:
                            logger.error(f"핸들러 실행 중 오류 발생: {e}")
                    
                except ValueError:
                    logger.warning(f"NAV 값 변환 실패: {nav_value} (ETF코드: {etf_code})")
        
        except Exception as e:
            logger.error(f"ETF NAV 데이터 처리 중 오류 발생: {e}")
    
    def get_last_price(self, stock_code: str) -> Optional[int]:
        """
        특정 종목의 최근 가격 조회
        
        Args:
            stock_code: 종목 코드
            
        Returns:
            int or None: 최근 가격 또는 데이터가 없는 경우 None
        """
        return self.last_price.get(stock_code)
    
    def set_target_price(self, stock_code: str, buy_price: Optional[int] = None, sell_price: Optional[int] = None):
        """
        종목별 목표가 설정
        
        Args:
            stock_code: 종목 코드
            buy_price: 매수 목표가
            sell_price: 매도 목표가
        """
        if buy_price is not None:
            self.target_buy_price[stock_code] = buy_price
            
        if sell_price is not None:
            self.target_sell_price[stock_code] = sell_price
    
    def get_price_queue(self) -> queue.Queue:
        """
        가격 데이터 큐 반환
        
        Returns:
            Queue: 가격 데이터 큐
        """
        return self.price_queue 