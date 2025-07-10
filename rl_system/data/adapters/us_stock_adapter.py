"""
미국 주식 데이터 어댑터
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

# 미국 주식 시스템 임포트
try:
    from us_stock_auto.api_client import APIClient
    from us_stock_auto.config import APP_KEY, APP_SECRET, URL_BASE
    from us_stock_auto.utils import is_market_open
except ImportError as e:
    logging.warning(f"US stock modules not available: {e}")
    APIClient = None
    APP_KEY = None
    APP_SECRET = None
    URL_BASE = None
    is_market_open = None

logger = logging.getLogger(__name__)


class USStockAdapter(DataProvider):
    """미국 주식 데이터 어댑터"""
    
    def __init__(self):
        """초기화"""
        self.api_client = None
        
        if APIClient:
            try:
                self.api_client = APIClient()
            except Exception as e:
                logger.error(f"Failed to initialize US stock API client: {e}")
    
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
        if not self.api_client:
            logger.error("API client not initialized")
            return pd.DataFrame()
        
        try:
            # 미국 주식 시장 데이터 조회 (현재 API는 실시간 데이터만 제공)
            # 히스토리컬 데이터는 별도 구현 필요
            data = []
            
            # 현재가 조회
            current_price = self.api_client.get_current_price(
                market=self._get_market_code(symbol),
                code=symbol
            )
            
            if current_price:
                data.append({
                    'date': datetime.now(),
                    'symbol': symbol,
                    'close': current_price,
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'volume': 0
                })
            
            return pd.DataFrame(data)
            
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
        # 미국 주식 API에서 기술적 지표 직접 제공하지 않음
        # 시장 데이터를 기반으로 계산해야 함
        try:
            market_data = self.get_market_data(
                symbol=symbol,
                start_date=datetime.now() - timedelta(days=period),
                end_date=datetime.now()
            )
            
            if market_data.empty:
                return pd.DataFrame()
            
            # 기술적 지표 계산 (기본적인 것들만)
            technical_data = market_data.copy()
            
            # 간단한 이동평균 계산
            if 'sma_20' in indicators:
                if len(market_data) >= 20:
                    technical_data['sma_20'] = market_data['close'].rolling(window=20).mean()
            
            if 'sma_50' in indicators:
                if len(market_data) >= 50:
                    technical_data['sma_50'] = market_data['close'].rolling(window=50).mean()
            
            return technical_data
            
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
        # 미국 주식 API에서 기본적 분석 데이터 직접 제공하지 않음
        # 별도 데이터 소스 연동 필요
        try:
            fundamental_data = {}
            
            # 기본 정보
            if 'basic_info' in data_types:
                fundamental_data['basic_info'] = {
                    'symbol': symbol,
                    'market': self._get_market_name(symbol),
                    'currency': 'USD'
                }
            
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
                current_price = self.api_client.get_current_price(
                    market=self._get_market_code(symbol),
                    code=symbol
                )
                if current_price:
                    realtime_data['current_price'] = current_price
            
            # 목표가 조회
            if 'target_price' in data_types:
                target_price = self.api_client.get_target_price(
                    market=self._get_market_code(symbol),
                    code=symbol
                )
                if target_price:
                    realtime_data['target_price'] = target_price
            
            # 환율 조회
            if 'exchange_rate' in data_types:
                exchange_rate = self.api_client.get_exchange_rate()
                if exchange_rate:
                    realtime_data['exchange_rate'] = exchange_rate
            
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
            market: 시장 구분 (nasdaq, nyse, all)
            filters: 필터 조건
            
        Returns:
            List[str]: 종목 코드 목록
        """
        # 미국 주식 API에서 종목 목록 직접 제공하지 않음
        # 일반적인 주요 종목들 반환
        try:
            symbols = []
            
            if market in ['nasdaq', 'all']:
                nasdaq_symbols = [
                    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
                    'META', 'NVDA', 'NFLX', 'ADBE', 'INTC'
                ]
                symbols.extend(nasdaq_symbols)
            
            if market in ['nyse', 'all']:
                nyse_symbols = [
                    'JPM', 'JNJ', 'V', 'PG', 'MA',
                    'UNH', 'HD', 'BAC', 'DIS', 'ASML'
                ]
                symbols.extend(nyse_symbols)
            
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
        if not symbol or not isinstance(symbol, str):
            return False
        
        # 미국 주식 심볼 형식 검증 (대문자 1~5자리)
        if not symbol.isupper() or len(symbol) < 1 or len(symbol) > 5:
            return False
        
        try:
            # 현재가 조회로 유효성 확인
            if self.api_client:
                current_price = self.api_client.get_current_price(
                    market=self._get_market_code(symbol),
                    code=symbol
                )
                return current_price is not None
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
        try:
            return {
                'symbol': symbol,
                'market': self._get_market_name(symbol),
                'currency': 'USD',
                'timezone': 'America/New_York',
                'data_provider': 'Korea Investment API'
            }
            
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
            # 미국 주식 거래일 생성 (주말 제외)
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
            if is_market_open:
                return is_market_open()
            
            # 기본 구현: 미국 주식 시장 시간 확인 (9:30 ~ 16:00 EST)
            if timestamp.weekday() >= 5:  # 주말
                return False
            
            # EST 기준 시간 (임시 구현)
            market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
            
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
                'market_open': now.replace(hour=9, minute=30, second=0, microsecond=0),
                'market_close': now.replace(hour=16, minute=0, second=0, microsecond=0),
                'timezone': 'America/New_York'
            }
            
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return {}
    
    def _get_market_code(self, symbol: str) -> str:
        """
        종목 코드로 시장 코드 추정
        
        Args:
            symbol: 종목 코드
            
        Returns:
            str: 시장 코드
        """
        # 일반적인 NASDAQ 종목들
        nasdaq_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        
        if symbol in nasdaq_symbols:
            return "NAS"
        else:
            return "NYS"  # 기본값은 NYSE
    
    def _get_market_name(self, symbol: str) -> str:
        """
        종목 코드로 시장 이름 추정
        
        Args:
            symbol: 종목 코드
            
        Returns:
            str: 시장 이름
        """
        market_code = self._get_market_code(symbol)
        return "NASDAQ" if market_code == "NAS" else "NYSE"
    
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
                # 시가총액 필터 구현 필요
                pass
            
            # 거래량 필터
            if 'min_volume' in filters:
                # 거래량 필터 구현 필요
                pass
            
            # 섹터 필터
            if 'sector' in filters:
                # 섹터 필터 구현 필요
                pass
            
            return filtered_symbols
            
        except Exception as e:
            logger.error(f"Failed to apply filters: {e}")
            return symbols