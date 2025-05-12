"""
한국 주식 자동매매 - 웹소켓 구독 관리 모듈
종목 구독 및 해지 관리 기능 제공
"""

import json
import logging
import time
from typing import List, Set, Dict, Any, Optional, TYPE_CHECKING

from korea_stock_auto.utils.utils import send_message, rate_limit_wait

# 타입 힌트를 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.websocket.connection_manager import ConnectionManager

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

class SubscriptionManager:
    """웹소켓 구독 관리 클래스"""
    
    def __init__(self, connection_manager: 'ConnectionManager'):
        """
        구독 관리자 초기화
        
        Args:
            connection_manager: 웹소켓 연결 관리자 인스턴스
        """
        self.conn_manager = connection_manager
        self.subscribed_symbols = set()  # 구독 중인 종목 목록
    
    def subscribe_symbols(self, symbols: List[str]) -> bool:
        """
        종목 구독 요청
        
        Args:
            symbols: 구독할 종목 코드 목록
            
        Returns:
            bool: 구독 요청 성공 여부
        """
        if not self.conn_manager.is_connected():
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
                    "approval_key": self.conn_manager.ws_conn_key,
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
                
                success = self.conn_manager.send_message(ws_request)
                if success:
                    logger.info(f"종목 구독 요청 전송: {symbol}")
                    # 구독 목록에 추가
                    self.subscribed_symbols.add(symbol)
                else:
                    logger.warning(f"종목 구독 요청 전송 실패: {symbol}")
                
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
    
    def unsubscribe_symbols(self, symbols: List[str]) -> bool:
        """
        종목 구독 해지 요청
        
        Args:
            symbols: 구독 해지할 종목 코드 목록
            
        Returns:
            bool: 구독 해지 요청 성공 여부
        """
        if not self.conn_manager.is_connected():
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
                    "approval_key": self.conn_manager.ws_conn_key,
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
                
                success = self.conn_manager.send_message(ws_request)
                if success:
                    logger.info(f"종목 구독 해지 요청 전송: {symbol}")
                    # 구독 목록에서 제거
                    self.subscribed_symbols.remove(symbol)
                else:
                    logger.warning(f"종목 구독 해지 요청 전송 실패: {symbol}")
                
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
    
    def unsubscribe_all(self) -> bool:
        """
        모든 종목 구독 해지
        
        Returns:
            bool: 구독 해지 요청 성공 여부
        """
        if not self.subscribed_symbols:
            logger.info("구독 중인 종목이 없습니다.")
            return True
            
        return self.unsubscribe_symbols(list(self.subscribed_symbols))
    
    def get_subscribed_symbols(self) -> List[str]:
        """
        현재 구독 중인 종목 목록 반환
        
        Returns:
            list: 구독 중인 종목 코드 목록
        """
        return list(self.subscribed_symbols)
    
    def is_subscribed(self, symbol: str) -> bool:
        """
        특정 종목 구독 여부 확인
        
        Args:
            symbol: 종목 코드
            
        Returns:
            bool: 구독 중인지 여부
        """
        return symbol in self.subscribed_symbols 