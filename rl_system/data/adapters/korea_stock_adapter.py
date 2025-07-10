"""
한국 주식 데이터 어댑터
"""

import sys
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging

# 상위 디렉토리를 패스에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from rl_system.core.interfaces.data_provider import DataProvider

# 한국 주식 시스템 임포트
try:
    from korea_stock_auto.config import AppConfig
    from korea_stock_auto.data.database import DatabaseManager
    from korea_stock_auto.api.client import KoreaInvestmentApiClient
    from korea_stock_auto.data.hybrid_data_collector import HybridDataCollector
except ImportError as e:
    logging.warning(f"Korea stock modules not available: {e}")
    AppConfig = None
    DatabaseManager = None
    KoreaInvestmentApiClient = None
    HybridDataCollector = None

logger = logging.getLogger(__name__)


class KoreaStockAdapter(DataProvider):
    """한국 주식 데이터 어댑터"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        초기화
        
        Args:
            config: 한국 주식 설정
        """
        self.config = config or AppConfig() if AppConfig else None
        self.db_manager = None
        self.api_client = None
        self.data_collector = None
        
        if self.config:
            try:
                self.db_manager = DatabaseManager()
                self.api_client = KoreaInvestmentApiClient(self.config)
                self.data_collector = HybridDataCollector(self.config)
            except Exception as e:
                logger.error(f"Failed to initialize Korea stock components: {e}")
    
    def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        시장 데이터 조회
        
        Args:
            symbol: 종목 코드
            start_date: 시작 날짜
            end_date: 종료 날짜
            interval: 데이터 간격
            
        Returns:
            pd.DataFrame: 시장 데이터
        """
        if not self.db_manager:
            logger.error("Database manager not initialized")
            return pd.DataFrame()
        
        try:
            # 데이터베이스에서 데이터 조회
            data = self.db_manager.get_price_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty and self.data_collector:
                # 데이터가 없으면 수집 시도
                logger.info(f"No data found for {symbol}, attempting to collect...")
                self.data_collector.collect_daily_data(symbol)
                
                # 다시 조회
                data = self.db_manager.get_price_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_technical_indicators(
        self,
        symbol: str,
        indicators: List[str],
        period: int = 252
    ) -> pd.DataFrame:
        """
        기술적 지표 조회
        
        Args:
            symbol: 종목 코드
            indicators: 지표 목록
            period: 조회 기간
            
        Returns:
            pd.DataFrame: 기술적 지표 데이터
        """
        if not self.db_manager:
            logger.error("Database manager not initialized")
            return pd.DataFrame()
        
        try:
            # 기술적 지표 데이터 조회
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period)
            
            data = self.db_manager.get_technical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # 요청된 지표만 필터링
            if not data.empty and indicators:
                available_indicators = [col for col in indicators if col in data.columns]
                if available_indicators:
                    data = data[['date'] + available_indicators]
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get technical indicators for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_fundamental_data(
        self,
        symbol: str,
        data_types: List[str]
    ) -> Dict[str, Any]:
        """
        기본적 분석 데이터 조회
        
        Args:
            symbol: 종목 코드
            data_types: 데이터 타입 목록
            
        Returns:
            Dict[str, Any]: 기본적 분석 데이터
        """
        if not self.api_client:
            logger.error("API client not initialized")
            return {}
        
        try:
            fundamental_data = {}
            
            # 기본 정보 조회
            if 'basic_info' in data_types:
                stock_info = self.api_client.get_stock_info(symbol)
                if stock_info:
                    fundamental_data['basic_info'] = stock_info
            
            # 재무 정보 조회
            if 'financial_info' in data_types:
                # 재무 정보 조회 구현 필요
                pass
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Failed to get fundamental data for {symbol}: {e}")
            return {}
    
    def get_realtime_data(
        self,
        symbol: str,
        data_types: List[str]
    ) -> Dict[str, Any]:
        """
        실시간 데이터 조회
        
        Args:
            symbol: 종목 코드
            data_types: 데이터 타입 목록
            
        Returns:
            Dict[str, Any]: 실시간 데이터
        """
        if not self.api_client:
            logger.error("API client not initialized")
            return {}
        
        try:
            realtime_data = {}
            
            # 현재가 조회
            if 'current_price' in data_types:
                current_price = self.api_client.get_current_price(symbol)
                if current_price:
                    realtime_data['current_price'] = current_price
            
            # 호가 정보 조회
            if 'orderbook' in data_types:
                orderbook = self.api_client.get_orderbook(symbol)
                if orderbook:
                    realtime_data['orderbook'] = orderbook
            
            return realtime_data
            
        except Exception as e:
            logger.error(f"Failed to get realtime data for {symbol}: {e}")
            return {}
    
    def get_symbol_list(
        self,
        market: str = "all",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        종목 목록 조회
        
        Args:
            market: 시장 구분 (kospi, kosdaq, all)
            filters: 필터 조건
            
        Returns:
            List[str]: 종목 코드 목록
        """
        if not self.api_client:
            logger.error("API client not initialized")
            return []
        
        try:
            # 시장별 종목 목록 조회
            symbols = []
            
            if market in ['kospi', 'all']:
                kospi_symbols = self.api_client.get_kospi_symbols()
                if kospi_symbols:
                    symbols.extend(kospi_symbols)
            
            if market in ['kosdaq', 'all']:
                kosdaq_symbols = self.api_client.get_kosdaq_symbols()
                if kosdaq_symbols:
                    symbols.extend(kosdaq_symbols)
            
            # 필터 적용
            if filters:
                symbols = self._apply_filters(symbols, filters)
            
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get symbol list: {e}")
            return []
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        종목 코드 유효성 검증
        
        Args:
            symbol: 종목 코드
            
        Returns:
            bool: 유효성 여부
        """
        if not symbol or len(symbol) != 6:
            return False
        
        try:
            # 종목 정보 조회로 유효성 확인
            if self.api_client:
                stock_info = self.api_client.get_stock_info(symbol)
                return stock_info is not None
            return False
            
        except Exception:
            return False
    
    def get_data_info(self, symbol: str) -> Dict[str, Any]:
        """
        데이터 정보 조회
        
        Args:
            symbol: 종목 코드
            
        Returns:
            Dict[str, Any]: 데이터 정보
        """
        if not self.db_manager:
            return {}
        
        try:
            # 데이터베이스에서 데이터 정보 조회
            info = self.db_manager.get_data_info(symbol)
            return info or {}
            
        except Exception as e:
            logger.error(f"Failed to get data info for {symbol}: {e}")
            return {}
    
    def get_trading_calendar(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[datetime]:
        """
        거래 일정 조회
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            List[datetime]: 거래일 목록
        """
        try:
            # 한국 주식 거래일 생성 (주말 제외)
            trading_days = []
            current_date = start_date
            
            while current_date <= end_date:
                # 주말 제외 (0: 월요일, 6: 일요일)
                if current_date.weekday() < 5:
                    trading_days.append(current_date)
                current_date += timedelta(days=1)
            
            return trading_days
            
        except Exception as e:
            logger.error(f"Failed to get trading calendar: {e}")
            return []
    
    def is_market_open(self, timestamp: datetime) -> bool:
        """
        시장 개장 여부 확인
        
        Args:
            timestamp: 확인할 시간
            
        Returns:
            bool: 개장 여부
        """
        try:
            # 한국 주식 시장 시간 확인 (9:00 ~ 15:30)
            if timestamp.weekday() >= 5:  # 주말
                return False
            
            market_open = timestamp.replace(hour=9, minute=0, second=0, microsecond=0)
            market_close = timestamp.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_open <= timestamp <= market_close
            
        except Exception:
            return False
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        시장 상태 조회
        
        Returns:
            Dict[str, Any]: 시장 상태 정보
        """
        try:
            now = datetime.now()
            is_open = self.is_market_open(now)
            
            return {
                'is_open': is_open,
                'market_time': now,
                'market_open': now.replace(hour=9, minute=0, second=0, microsecond=0),
                'market_close': now.replace(hour=15, minute=30, second=0, microsecond=0),
                'timezone': 'Asia/Seoul'
            }
            
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return {}
    
    def _apply_filters(self, symbols: List[str], filters: Dict[str, Any]) -> List[str]:
        """
        필터 적용
        
        Args:
            symbols: 종목 코드 목록
            filters: 필터 조건
            
        Returns:
            List[str]: 필터링된 종목 코드 목록
        """
        try:
            filtered_symbols = symbols
            
            # 시가총액 필터
            if 'min_market_cap' in filters:
                # 시가총액 필터 구현
                pass
            
            # 거래량 필터
            if 'min_volume' in filters:
                # 거래량 필터 구현
                pass
            
            return filtered_symbols
            
        except Exception as e:
            logger.error(f"Failed to apply filters: {e}")
            return symbols