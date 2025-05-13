"""
한국 주식 자동매매 - 가격 데이터 모듈

주식 가격 데이터의 저장 및 조회 기능을 제공합니다.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.data.database.base import DatabaseManager

# 로깅 설정
logger = logging.getLogger(__name__)

class PriceDataManager(DatabaseManager):
    """
    주식 가격 데이터 관리 클래스
    
    주식 가격 데이터의 저장 및 조회 기능을 제공합니다.
    """
    
    def __init__(self, db_path: str = "stock_data.db"):
        """
        가격 데이터 매니저 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        super().__init__(db_path)
        self._initialize_table()
    
    def _initialize_table(self) -> None:
        """
        가격 데이터 테이블 초기화
        """
        table_schema = '''
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            code TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            UNIQUE(date, code)
        )
        '''
        
        if self.create_table(table_schema):
            logger.info("가격 데이터 테이블 초기화 완료")
        else:
            logger.error("가격 데이터 테이블 초기화 실패")
    
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
        try:
            data = (
                date,
                code,
                price_data.get('open', 0),
                price_data.get('high', 0),
                price_data.get('low', 0),
                price_data.get('close', 0),
                price_data.get('volume', 0)
            )
            
            query = '''
            INSERT OR REPLACE INTO price_data 
            (date, code, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            
            result = self.execute_query(query, data)
            if result:
                logger.debug(f"가격 데이터 저장 완료: {code} {date}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"가격 데이터 저장 실패: {e}")
            send_message(f"가격 데이터 저장 실패: {e}")
            return False
    
    def get_price_history(self, code: str, days: int = 30) -> pd.DataFrame:
        """
        종목의 가격 이력 조회
        
        Args:
            code: 종목 코드
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 가격 이력 데이터프레임
        """
        try:
            query = '''
            SELECT date, open, high, low, close, volume
            FROM price_data
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
            logger.error(f"가격 이력 조회 실패: {e}")
            send_message(f"가격 이력 조회 실패: {e}")
            return pd.DataFrame()
    
    def get_latest_price(self, code: str) -> Optional[Dict[str, Any]]:
        """
        종목의 최신 가격 조회
        
        Args:
            code: 종목 코드
            
        Returns:
            Optional[Dict[str, Any]]: 최신 가격 정보 또는 None
        """
        try:
            query = '''
            SELECT date, open, high, low, close, volume
            FROM price_data
            WHERE code = ?
            ORDER BY date DESC
            LIMIT 1
            '''
            
            result = self.execute_query(query, (code,), fetch=True, as_df=True)
            if result is not None and not result.empty:
                row = result.iloc[0]
                return {
                    'date': row['date'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
            else:
                return None
            
        except Exception as e:
            logger.error(f"최신 가격 조회 실패: {e}")
            send_message(f"최신 가격 조회 실패: {e}")
            return None 