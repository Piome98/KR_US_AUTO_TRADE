"""
데이터 제공자 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime


class DataProvider(ABC):
    """데이터 제공자 추상 인터페이스"""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_symbol_list(
        self,
        market: str = "all",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        종목 목록 조회
        
        Args:
            market: 시장 구분
            filters: 필터 조건
            
        Returns:
            List[str]: 종목 코드 목록
        """
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """
        종목 코드 유효성 검증
        
        Args:
            symbol: 종목 코드
            
        Returns:
            bool: 유효성 여부
        """
        pass
    
    @abstractmethod
    def get_data_info(self, symbol: str) -> Dict[str, Any]:
        """
        데이터 정보 조회
        
        Args:
            symbol: 종목 코드
            
        Returns:
            Dict[str, Any]: 데이터 정보
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def is_market_open(self, timestamp: datetime) -> bool:
        """
        시장 개장 여부 확인
        
        Args:
            timestamp: 확인할 시간
            
        Returns:
            bool: 개장 여부
        """
        pass
    
    @abstractmethod
    def get_market_status(self) -> Dict[str, Any]:
        """
        시장 상태 조회
        
        Returns:
            Dict[str, Any]: 시장 상태 정보
        """
        pass