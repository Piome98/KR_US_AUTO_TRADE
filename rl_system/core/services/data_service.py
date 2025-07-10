"""
데이터 서비스 - 통합 데이터 관리
"""

import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from rl_system.core.interfaces.data_provider import DataProvider
from rl_system.config.rl_config import RLConfig

logger = logging.getLogger(__name__)


class DataService:
    """통합 데이터 관리 서비스"""
    
    def __init__(self, config: RLConfig):
        """
        초기화
        
        Args:
            config: RL 시스템 설정
        """
        self.config = config
        self.data_providers: Dict[str, DataProvider] = {}
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_timeout = 300  # 5분 캐시 유지
        self.last_cache_update: Dict[str, datetime] = {}
    
    def register_data_provider(self, name: str, provider: DataProvider) -> None:
        """
        데이터 제공자 등록
        
        Args:
            name: 제공자 이름 (예: 'korea_stock', 'us_stock')
            provider: 데이터 제공자 인스턴스
        """
        self.data_providers[name] = provider
        logger.info(f"Data provider registered: {name}")
    
    def get_data_provider(self, name: str) -> Optional[DataProvider]:
        """
        데이터 제공자 조회
        
        Args:
            name: 제공자 이름
            
        Returns:
            DataProvider: 데이터 제공자 또는 None
        """
        return self.data_providers.get(name)
    
    def get_market_data(
        self,
        provider_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        시장 데이터 조회
        
        Args:
            provider_name: 데이터 제공자 이름
            symbol: 종목 코드
            start_date: 시작 날짜
            end_date: 종료 날짜
            interval: 데이터 간격
            use_cache: 캐시 사용 여부
            
        Returns:
            pd.DataFrame: 시장 데이터
        """
        cache_key = f"{provider_name}_{symbol}_{start_date}_{end_date}_{interval}"
        
        # 캐시 확인
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key].copy()
        
        # 데이터 제공자 확인
        provider = self.data_providers.get(provider_name)
        if not provider:
            logger.error(f"Data provider not found: {provider_name}")
            return pd.DataFrame()
        
        try:
            # 데이터 조회
            data = provider.get_market_data(symbol, start_date, end_date, interval)
            
            # 데이터 검증
            if self._validate_market_data(data):
                # 캐시 저장
                if use_cache:
                    self.cache[cache_key] = data.copy()
                    self.last_cache_update[cache_key] = datetime.now()
                
                logger.info(f"Market data retrieved: {provider_name}/{symbol}, {len(data)} records")
                return data
            else:
                logger.warning(f"Invalid market data: {provider_name}/{symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to get market data: {provider_name}/{symbol}, {e}")
            return pd.DataFrame()
    
    def get_multiple_market_data(
        self,
        requests: List[Dict[str, Any]],
        max_workers: int = 4
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 종목의 시장 데이터 동시 조회
        
        Args:
            requests: 데이터 요청 목록
                [{'provider': 'korea_stock', 'symbol': '005930', 'start_date': ..., 'end_date': ...}]
            max_workers: 최대 워커 수
            
        Returns:
            Dict[str, pd.DataFrame]: 종목별 데이터
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            future_to_request = {}
            for request in requests:
                future = executor.submit(
                    self.get_market_data,
                    request['provider'],
                    request['symbol'],
                    request['start_date'],
                    request['end_date'],
                    request.get('interval', '1d'),
                    request.get('use_cache', True)
                )
                future_to_request[future] = request
            
            # 결과 수집
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    data = future.result()
                    key = f"{request['provider']}_{request['symbol']}"
                    results[key] = data
                except Exception as e:
                    logger.error(f"Failed to get data for {request}: {e}")
                    results[f"{request['provider']}_{request['symbol']}"] = pd.DataFrame()
        
        return results
    
    def get_technical_indicators(
        self,
        provider_name: str,
        symbol: str,
        indicators: List[str],
        period: int = 252
    ) -> pd.DataFrame:
        """
        기술적 지표 조회
        
        Args:
            provider_name: 데이터 제공자 이름
            symbol: 종목 코드
            indicators: 지표 목록
            period: 조회 기간
            
        Returns:
            pd.DataFrame: 기술적 지표 데이터
        """
        provider = self.data_providers.get(provider_name)
        if not provider:
            logger.error(f"Data provider not found: {provider_name}")
            return pd.DataFrame()
        
        try:
            data = provider.get_technical_indicators(symbol, indicators, period)
            logger.info(f"Technical indicators retrieved: {provider_name}/{symbol}, {len(indicators)} indicators")
            return data
        except Exception as e:
            logger.error(f"Failed to get technical indicators: {provider_name}/{symbol}, {e}")
            return pd.DataFrame()
    
    def get_realtime_data(
        self,
        provider_name: str,
        symbol: str,
        data_types: List[str]
    ) -> Dict[str, Any]:
        """
        실시간 데이터 조회
        
        Args:
            provider_name: 데이터 제공자 이름
            symbol: 종목 코드
            data_types: 데이터 타입 목록
            
        Returns:
            Dict[str, Any]: 실시간 데이터
        """
        provider = self.data_providers.get(provider_name)
        if not provider:
            logger.error(f"Data provider not found: {provider_name}")
            return {}
        
        try:
            data = provider.get_realtime_data(symbol, data_types)
            logger.debug(f"Realtime data retrieved: {provider_name}/{symbol}")
            return data
        except Exception as e:
            logger.error(f"Failed to get realtime data: {provider_name}/{symbol}, {e}")
            return {}
    
    def get_symbol_list(
        self,
        provider_name: str,
        market: str = "all",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        종목 목록 조회
        
        Args:
            provider_name: 데이터 제공자 이름
            market: 시장 구분
            filters: 필터 조건
            
        Returns:
            List[str]: 종목 코드 목록
        """
        provider = self.data_providers.get(provider_name)
        if not provider:
            logger.error(f"Data provider not found: {provider_name}")
            return []
        
        try:
            symbols = provider.get_symbol_list(market, filters)
            logger.info(f"Symbol list retrieved: {provider_name}/{market}, {len(symbols)} symbols")
            return symbols
        except Exception as e:
            logger.error(f"Failed to get symbol list: {provider_name}/{market}, {e}")
            return []
    
    def validate_symbol(self, provider_name: str, symbol: str) -> bool:
        """
        종목 코드 유효성 검증
        
        Args:
            provider_name: 데이터 제공자 이름
            symbol: 종목 코드
            
        Returns:
            bool: 유효성 여부
        """
        provider = self.data_providers.get(provider_name)
        if not provider:
            return False
        
        try:
            return provider.validate_symbol(symbol)
        except Exception:
            return False
    
    def is_market_open(self, provider_name: str, timestamp: Optional[datetime] = None) -> bool:
        """
        시장 개장 여부 확인
        
        Args:
            provider_name: 데이터 제공자 이름
            timestamp: 확인할 시간 (None이면 현재 시간)
            
        Returns:
            bool: 개장 여부
        """
        provider = self.data_providers.get(provider_name)
        if not provider:
            return False
        
        try:
            timestamp = timestamp or datetime.now()
            return provider.is_market_open(timestamp)
        except Exception:
            return False
    
    def get_unified_data(
        self,
        symbols_by_provider: Dict[str, List[str]],
        start_date: datetime,
        end_date: datetime,
        include_technical: bool = True,
        technical_indicators: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        통합 데이터 조회 (여러 제공자에서 데이터 수집)
        
        Args:
            symbols_by_provider: 제공자별 종목 목록
                {'korea_stock': ['005930', '000660'], 'us_stock': ['AAPL', 'GOOGL']}
            start_date: 시작 날짜
            end_date: 종료 날짜
            include_technical: 기술적 지표 포함 여부
            technical_indicators: 기술적 지표 목록
            
        Returns:
            Dict[str, pd.DataFrame]: 종목별 통합 데이터
        """
        results = {}
        
        # 기본 시장 데이터 조회
        requests = []
        for provider_name, symbols in symbols_by_provider.items():
            for symbol in symbols:
                requests.append({
                    'provider': provider_name,
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                })
        
        market_data = self.get_multiple_market_data(requests)
        
        # 기술적 지표 추가
        if include_technical:
            indicators = technical_indicators or self.config.get_data_config()['technical_indicators']
            
            for provider_name, symbols in symbols_by_provider.items():
                for symbol in symbols:
                    key = f"{provider_name}_{symbol}"
                    if key in market_data and not market_data[key].empty:
                        try:
                            technical_data = self.get_technical_indicators(
                                provider_name, symbol, indicators
                            )
                            if not technical_data.empty:
                                # 데이터 병합
                                merged_data = self._merge_market_and_technical_data(
                                    market_data[key], technical_data
                                )
                                results[key] = merged_data
                            else:
                                results[key] = market_data[key]
                        except Exception as e:
                            logger.error(f"Failed to merge technical data for {key}: {e}")
                            results[key] = market_data[key]
                    else:
                        results[key] = market_data[key]
        else:
            results = market_data
        
        return results
    
    def clear_cache(self, pattern: Optional[str] = None) -> None:
        """
        캐시 클리어
        
        Args:
            pattern: 클리어할 캐시 패턴 (None이면 모든 캐시)
        """
        if pattern:
            keys_to_remove = [key for key in self.cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.last_cache_update:
                    del self.last_cache_update[key]
            logger.info(f"Cache cleared: {len(keys_to_remove)} entries matching '{pattern}'")
        else:
            self.cache.clear()
            self.last_cache_update.clear()
            logger.info("All cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 조회
        
        Returns:
            Dict[str, Any]: 캐시 통계
        """
        return {
            'cache_size': len(self.cache),
            'cache_keys': list(self.cache.keys()),
            'memory_usage_mb': sum(df.memory_usage(deep=True).sum() for df in self.cache.values()) / 1024 / 1024,
            'last_update': self.last_cache_update
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        캐시 유효성 확인
        
        Args:
            cache_key: 캐시 키
            
        Returns:
            bool: 유효성 여부
        """
        if cache_key not in self.cache:
            return False
        
        if cache_key not in self.last_cache_update:
            return False
        
        last_update = self.last_cache_update[cache_key]
        return (datetime.now() - last_update).total_seconds() < self.cache_timeout
    
    def _validate_market_data(self, data: pd.DataFrame) -> bool:
        """
        시장 데이터 유효성 검증
        
        Args:
            data: 시장 데이터
            
        Returns:
            bool: 유효성 여부
        """
        if data.empty:
            return False
        
        # 필수 컬럼 확인
        required_columns = ['close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 데이터 타입 확인
        try:
            pd.to_numeric(data['close'], errors='raise')
        except (ValueError, TypeError):
            return False
        
        return True
    
    def _merge_market_and_technical_data(
        self,
        market_data: pd.DataFrame,
        technical_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        시장 데이터와 기술적 지표 데이터 병합
        
        Args:
            market_data: 시장 데이터
            technical_data: 기술적 지표 데이터
            
        Returns:
            pd.DataFrame: 병합된 데이터
        """
        try:
            # 날짜 컬럼 기준으로 병합
            if 'date' in market_data.columns and 'date' in technical_data.columns:
                merged = pd.merge(
                    market_data, 
                    technical_data, 
                    on='date', 
                    how='left'
                )
            elif market_data.index.name == 'date' and technical_data.index.name == 'date':
                merged = pd.merge(
                    market_data, 
                    technical_data, 
                    left_index=True, 
                    right_index=True, 
                    how='left'
                )
            else:
                # 인덱스가 날짜인 경우
                merged = pd.merge(
                    market_data, 
                    technical_data, 
                    left_index=True, 
                    right_index=True, 
                    how='left'
                )
            
            return merged
            
        except Exception as e:
            logger.error(f"Failed to merge data: {e}")
            return market_data