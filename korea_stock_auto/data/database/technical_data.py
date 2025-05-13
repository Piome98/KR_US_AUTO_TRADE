"""
한국 주식 자동매매 - 기술적 지표 데이터 모듈

기술적 지표 데이터의 저장 및 조회 기능을 제공합니다.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.data.database.base import DatabaseManager

# 로깅 설정
logger = logging.getLogger(__name__)

class TechnicalDataManager(DatabaseManager):
    """
    기술적 지표 데이터 관리 클래스
    
    기술적 지표 데이터의 저장 및 조회 기능을 제공합니다.
    """
    
    def __init__(self, db_path: str = "stock_data.db"):
        """
        기술적 지표 데이터 매니저 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        super().__init__(db_path)
        self._initialize_table()
    
    def _initialize_table(self) -> None:
        """
        기술적 지표 테이블 초기화
        """
        table_schema = '''
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            code TEXT,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            sma_5 REAL,
            sma_20 REAL,
            sma_60 REAL,
            sma_120 REAL,
            ema_5 REAL,
            ema_20 REAL,
            ema_60 REAL,
            bollinger_upper REAL,
            bollinger_middle REAL,
            bollinger_lower REAL,
            atr REAL,
            adx REAL,
            obv REAL,
            UNIQUE(date, code)
        )
        '''
        
        if self.create_table(table_schema):
            logger.info("기술적 지표 테이블 초기화 완료")
        else:
            logger.error("기술적 지표 테이블 초기화 실패")
    
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
        try:
            data = (
                date,
                code,
                indicators.get('rsi', None),
                indicators.get('macd', None),
                indicators.get('macd_signal', None),
                indicators.get('macd_hist', None),
                indicators.get('sma_5', None),
                indicators.get('sma_20', None),
                indicators.get('sma_60', None),
                indicators.get('sma_120', None),
                indicators.get('ema_5', None),
                indicators.get('ema_20', None),
                indicators.get('ema_60', None),
                indicators.get('bollinger_upper', None),
                indicators.get('bollinger_middle', None),
                indicators.get('bollinger_lower', None),
                indicators.get('atr', None),
                indicators.get('adx', None),
                indicators.get('obv', None)
            )
            
            query = '''
            INSERT OR REPLACE INTO technical_indicators 
            (date, code, rsi, macd, macd_signal, macd_hist, 
            sma_5, sma_20, sma_60, sma_120, 
            ema_5, ema_20, ema_60, 
            bollinger_upper, bollinger_middle, bollinger_lower, 
            atr, adx, obv)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            result = self.execute_query(query, data)
            if result:
                logger.debug(f"기술적 지표 저장 완료: {code} {date}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"기술적 지표 저장 실패: {e}")
            send_message(f"기술적 지표 저장 실패: {e}")
            return False
    
    def get_technical_indicators(self, code: str, days: int = 30) -> pd.DataFrame:
        """
        종목의 기술적 지표 이력 조회
        
        Args:
            code: 종목 코드
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 기술적 지표 데이터프레임
        """
        try:
            query = '''
            SELECT date, rsi, macd, macd_signal, macd_hist, 
                   sma_5, sma_20, sma_60, sma_120, 
                   ema_5, ema_20, ema_60, 
                   bollinger_upper, bollinger_middle, bollinger_lower, 
                   atr, adx, obv
            FROM technical_indicators
            WHERE code = ?
            ORDER BY date DESC
            LIMIT ?
            '''
            
            result = self.execute_query(query, (code, days), fetch=True, as_df=True)
            if result is not None and not result.empty:
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"기술적 지표 조회 실패: {e}")
            send_message(f"기술적 지표 조회 실패: {e}")
            return pd.DataFrame()
    
    def get_latest_indicators(self, code: str) -> Optional[Dict[str, Any]]:
        """
        종목의 최신 기술적 지표 조회
        
        Args:
            code: 종목 코드
            
        Returns:
            Optional[Dict[str, Any]]: 최신 기술적 지표 정보 또는 None
        """
        try:
            query = '''
            SELECT date, rsi, macd, macd_signal, macd_hist, 
                   sma_5, sma_20, sma_60, sma_120, 
                   ema_5, ema_20, ema_60, 
                   bollinger_upper, bollinger_middle, bollinger_lower, 
                   atr, adx, obv
            FROM technical_indicators
            WHERE code = ?
            ORDER BY date DESC
            LIMIT 1
            '''
            
            result = self.execute_query(query, (code,), fetch=True, as_df=True)
            if result is not None and not result.empty:
                # 첫 번째 행을 딕셔너리로 변환
                row = result.iloc[0]
                return row.to_dict()
            else:
                return None
            
        except Exception as e:
            logger.error(f"최신 기술적 지표 조회 실패: {e}")
            send_message(f"최신 기술적 지표 조회 실패: {e}")
            return None 