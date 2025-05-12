"""
한국 주식 자동매매 - 웹소켓 통합 모듈
실시간 시세 데이터 수신 및 처리 기능 통합
"""

import logging
import time
import queue
from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING

from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api.websocket.connection_manager import ConnectionManager
from korea_stock_auto.api.websocket.subscription_manager import SubscriptionManager
from korea_stock_auto.api.websocket.data_processor import DataProcessor

# 타입 힌트를 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.websocket.connection_manager import ConnectionManager
    from korea_stock_auto.api.websocket.subscription_manager import SubscriptionManager
    from korea_stock_auto.api.websocket.data_processor import DataProcessor

# 로깅 설정
logger = logging.getLogger("stock_auto")

class StockWebSocket:
    """실시간 주식 시세 웹소켓 클래스"""
    
    def __init__(self):
        """웹소켓 클래스 초기화"""
        # 컴포넌트 초기화
        self.connection_manager = ConnectionManager()
        self.data_processor = DataProcessor()
        self.subscription_manager = SubscriptionManager(self.connection_manager)
        
        # 상태 변수
        self.symbol_list = []
        
        # 메시지 처리 콜백 설정
        self.connection_manager.set_message_callback(self.data_processor.process_message)
    
    def start(self, symbol_list: Optional[List[str]] = None):
        """
        웹소켓 연결 시작
        
        Args:
            symbol_list (list): 실시간 조회할 종목 코드 리스트
        """
        if symbol_list:
            self.symbol_list = symbol_list
            
        # 웹소켓 연결 시작
        self.connection_manager.start()
        
        # 종목 구독
        if self.symbol_list and self.connection_manager.is_connected():
            self.subscription_manager.subscribe_symbols(self.symbol_list)
    
    def subscribe_symbols(self, symbols: List[str]) -> bool:
        """
        종목 구독 요청
        
        Args:
            symbols (list): 구독할 종목 코드 목록
            
        Returns:
            bool: 구독 요청 성공 여부
        """
        # 종목 리스트 업데이트
        for symbol in symbols:
            if symbol not in self.symbol_list:
                self.symbol_list.append(symbol)
                
        # 구독 요청
        return self.subscription_manager.subscribe_symbols(symbols)
    
    def unsubscribe_symbols(self, symbols: List[str]) -> bool:
        """
        종목 구독 해지 요청
        
        Args:
            symbols (list): 구독 해지할 종목 코드 목록
            
        Returns:
            bool: 구독 해지 요청 성공 여부
        """
        # 종목 리스트 업데이트
        for symbol in symbols:
            if symbol in self.symbol_list:
                self.symbol_list.remove(symbol)
                
        # 구독 해지 요청
        return self.subscription_manager.unsubscribe_symbols(symbols)
    
    def add_data_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        데이터 처리 핸들러 추가
        
        Args:
            handler: 데이터 처리 핸들러 함수
        """
        self.data_processor.add_data_handler(handler)
    
    def remove_data_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        데이터 처리 핸들러 제거
        
        Args:
            handler: 제거할 데이터 처리 핸들러 함수
        """
        self.data_processor.remove_data_handler(handler)
    
    def get_price_queue(self) -> queue.Queue:
        """
        가격 데이터 큐 반환
        
        Returns:
            Queue: 가격 데이터 큐
        """
        return self.data_processor.get_price_queue()
    
    def get_last_price(self, stock_code: str) -> Optional[int]:
        """
        특정 종목의 최근 가격 조회
        
        Args:
            stock_code: 종목 코드
            
        Returns:
            int or None: 최근 가격 또는 데이터가 없는 경우 None
        """
        return self.data_processor.get_last_price(stock_code)
    
    def set_target_price(self, stock_code: str, buy_price: Optional[int] = None, sell_price: Optional[int] = None):
        """
        종목별 목표가 설정
        
        Args:
            stock_code: 종목 코드
            buy_price: 매수 목표가
            sell_price: 매도 목표가
        """
        self.data_processor.set_target_price(stock_code, buy_price, sell_price)
    
    def is_connected(self) -> bool:
        """
        웹소켓 연결 상태 확인
        
        Returns:
            bool: 연결 상태
        """
        return self.connection_manager.is_connected()
    
    def get_subscribed_symbols(self) -> List[str]:
        """
        현재 구독 중인 종목 목록 반환
        
        Returns:
            list: 구독 중인 종목 코드 목록
        """
        return self.subscription_manager.get_subscribed_symbols()
    
    def reconnect(self):
        """웹소켓 재연결 시도"""
        self.connection_manager.reconnect()
    
    def stop(self):
        """웹소켓 연결 종료"""
        # 구독 중인 종목 해지
        self.subscription_manager.unsubscribe_all()
        
        # 웹소켓 연결 종료
        self.connection_manager.stop()
        
        logger.info("웹소켓 종료 완료")
        send_message("웹소켓 종료 완료") 