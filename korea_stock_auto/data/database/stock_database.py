"""
한국 주식 자동매매 - 통합 데이터베이스 모듈

기존 StockDatabase 클래스의 기능을 유지하면서 모듈화된 매니저들을 내부적으로 사용합니다.
하위 호환성을 위한 클래스입니다.
"""

import os
import json
import datetime
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple

from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.data.database.price_data import PriceDataManager
from korea_stock_auto.data.database.transaction_data import TransactionDataManager
from korea_stock_auto.data.database.technical_data import TechnicalDataManager
from korea_stock_auto.data.database.market_data import MarketDataManager
from korea_stock_auto.data.database.news_data import NewsDataManager
from korea_stock_auto.data.database.trading_stats import TradingStatsManager

# 로깅 설정
logger = logging.getLogger(__name__)

class StockDatabase:
    """
    주식 데이터 및 거래 내역 데이터베이스
    
    주가 데이터, 거래 내역, 기술적 지표, 시장 지수 등의 데이터를 저장하고 조회하는 기능을 제공합니다.
    """
    
    def __init__(self, db_path: str = "stock_data.db"):
        """
        데이터베이스 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        
        # 각 데이터 매니저 초기화
        self.price_manager = PriceDataManager(db_path)
        self.transaction_manager = TransactionDataManager(db_path)
        self.technical_manager = TechnicalDataManager(db_path)
        self.market_manager = MarketDataManager(db_path)
        self.news_manager = NewsDataManager(db_path)
        self.stats_manager = TradingStatsManager(db_path)
        
        logger.info("데이터베이스 초기화 완료")
    
    # 가격 데이터 관련 메서드
    def save_price_data(self, code: str, date: str, price_data: Dict[str, Any]) -> bool:
        """
        종목 가격 데이터 저장
        
        Args:
            code: 종목 코드
            date: 날짜 (YYYY-MM-DD)
            price_data: 가격 데이터 딕셔너리
            
        Returns:
            bool: 저장 성공 여부
        """
        return self.price_manager.save_price_data(code, date, price_data)
    
    def get_price_history(self, code: str, days: int = 30) -> pd.DataFrame:
        """
        종목의 가격 이력 조회
        
        Args:
            code: 종목 코드
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 가격 이력 데이터프레임
        """
        return self.price_manager.get_price_history(code, days)
    
    # 거래 내역 관련 메서드
    def log_transaction(self, code: str, action: str, price: float, quantity: int, 
                       state: Optional[str] = None, entry_price: Optional[float] = None) -> bool:
        """
        거래 내역 저장
        
        Args:
            code: 종목 코드
            action: 매매 유형 (BUY/SELL)
            price: 거래 가격
            quantity: 거래 수량
            state: 거래 상태
            entry_price: 매수 시 진입 가격 (매도시에만 사용)
            
        Returns:
            bool: 저장 성공 여부
        """
        return self.transaction_manager.log_transaction(code, action, price, quantity, state, entry_price)
    
    def log_risk_event(self, code: str, event_type: str, price: float, description: str = "") -> bool:
        """
        위험 관리 이벤트 기록
        
        Args:
            code: 종목 코드
            event_type: 이벤트 유형 (STOP_LOSS/TAKE_PROFIT/TRAILING_STOP)
            price: 현재가
            description: 추가 설명
            
        Returns:
            bool: 저장 성공 여부
        """
        return self.transaction_manager.log_risk_event(code, event_type, price, description)
    
    # 기술적 지표 관련 메서드
    def save_technical_indicators(self, code: str, date: str, indicators: Dict[str, Any]) -> bool:
        """
        기술적 지표 저장
        
        Args:
            code: 종목 코드
            date: 날짜 (YYYY-MM-DD)
            indicators: 기술적 지표 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        return self.technical_manager.save_technical_indicators(code, date, indicators)
    
    def get_technical_indicators(self, code: str, days: int = 30) -> pd.DataFrame:
        """
        종목의 기술적 지표 이력 조회
        
        Args:
            code: 종목 코드
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 기술적 지표 데이터프레임
        """
        return self.technical_manager.get_technical_indicators(code, days)
    
    # 시장 지수 관련 메서드
    def save_market_index(self, date: str, index_data: Dict[str, Any]) -> bool:
        """
        시장 지수 데이터 저장
        
        Args:
            date: 날짜 (YYYY-MM-DD)
            index_data: 지수 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        return self.market_manager.save_market_index(date, index_data)
    
    def get_market_index_history(self, index_code: str, days: int = 30) -> pd.DataFrame:
        """
        시장 지수 이력 조회
        
        Args:
            index_code: 지수 코드 (KOSPI/KOSDAQ)
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 지수 이력 데이터프레임
        """
        return self.market_manager.get_market_index_history(index_code, days)
    
    # 뉴스 이벤트 관련 메서드
    def save_news_event(self, code: str, news_data: Dict[str, Any]) -> bool:
        """
        종목 관련 뉴스 이벤트 저장
        
        Args:
            code: 종목 코드
            news_data: 뉴스 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        return self.news_manager.save_news_event(code, news_data)
    
    def get_news_events(self, code: str, days: int = 7) -> pd.DataFrame:
        """
        종목 관련 뉴스 이벤트 조회
        
        Args:
            code: 종목 코드
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 뉴스 이벤트 데이터프레임
        """
        return self.news_manager.get_news_events(code, days)
    
    # 거래 통계 관련 메서드
    def save_trading_stats(self, code: str, date: str, stats_data: Dict[str, Any]) -> bool:
        """
        거래 통계 데이터 저장
        
        Args:
            code: 종목 코드
            date: 날짜 (YYYY-MM-DD)
            stats_data: 통계 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        return self.stats_manager.save_trading_stats(code, date, stats_data)
    
    def get_trading_stats(self, code: str, days: int = 30) -> pd.DataFrame:
        """
        거래 통계 조회
        
        Args:
            code: 종목 코드
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 거래 통계 데이터프레임
        """
        return self.stats_manager.get_trading_stats(code, days)
    
    # 통합 데이터 조회
    def get_combined_data(self, code: str, days: int = 30) -> pd.DataFrame:
        """
        가격, 기술적 지표, 거래 통계를 모두 결합한 데이터 조회
        
        Args:
            code: 종목 코드
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 결합된 데이터프레임
        """
        try:
            price_df = self.get_price_history(code, days)
            tech_df = self.get_technical_indicators(code, days)
            stats_df = self.get_trading_stats(code, days)
            
            # 날짜 기준으로 데이터프레임 결합
            if not price_df.empty:
                result = price_df.set_index('date')
                
                if not tech_df.empty:
                    tech_df = tech_df.set_index('date')
                    result = result.join(tech_df, how='left')
                
                if not stats_df.empty:
                    stats_df = stats_df.set_index('date')
                    result = result.join(stats_df, how='left')
                
                return result.reset_index()
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"결합 데이터 조회 실패: {e}")
            send_message(f"결합 데이터 조회 실패: {e}")
            return pd.DataFrame() 