"""
한국 주식 자동매매 - 시장 지수 데이터 모듈

시장 지수 데이터의 저장 및 조회 기능을 제공합니다.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List

from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.data.database.base import DatabaseManager

# 로깅 설정
logger = logging.getLogger(__name__)

class MarketDataManager(DatabaseManager):
    """
    시장 지수 데이터 관리 클래스
    
    시장 지수 데이터의 저장 및 조회 기능을 제공합니다.
    """
    
    def __init__(self, db_path: str = "stock_data.db"):
        """
        시장 지수 데이터 매니저 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        super().__init__(db_path)
        self._initialize_table()
    
    def _initialize_table(self) -> None:
        """
        시장 지수 테이블 초기화
        """
        table_schema = '''
        CREATE TABLE IF NOT EXISTS market_indices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            index_code TEXT,
            index_name TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            value REAL,
            change_value REAL,
            change_percent REAL,
            UNIQUE(date, index_code)
        )
        '''
        
        if self.create_table(table_schema):
            logger.info("시장 지수 테이블 초기화 완료")
        else:
            logger.error("시장 지수 테이블 초기화 실패")
    
    def save_market_index(self, date: str, index_data: Dict[str, Any]) -> bool:
        """
        시장 지수 데이터 저장
        
        Args:
            date: 날짜 (YYYY-MM-DD)
            index_data: 지수 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            data = (
                date,
                index_data.get('index_code', 'KOSPI'),
                index_data.get('index_name', '코스피'),
                index_data.get('open', 0),
                index_data.get('high', 0),
                index_data.get('low', 0),
                index_data.get('close', 0),
                index_data.get('volume', 0),
                index_data.get('value', 0),
                index_data.get('change_value', 0),
                index_data.get('change_percent', 0)
            )
            
            query = '''
            INSERT OR REPLACE INTO market_indices 
            (date, index_code, index_name, open, high, low, close, 
            volume, value, change_value, change_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            result = self.execute_query(query, data)
            if result:
                logger.debug(f"시장 지수 데이터 저장 완료: {index_data.get('index_code', 'KOSPI')} {date}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"시장 지수 데이터 저장 실패: {e}")
            send_message(f"시장 지수 데이터 저장 실패: {e}")
            return False
    
    def get_market_index_history(self, index_code: str, days: int = 30) -> pd.DataFrame:
        """
        시장 지수 이력 조회
        
        Args:
            index_code: 지수 코드 (KOSPI/KOSDAQ)
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 지수 이력 데이터프레임
        """
        try:
            query = '''
            SELECT date, open, high, low, close, volume, value, change_value, change_percent
            FROM market_indices
            WHERE index_code = ?
            ORDER BY date DESC
            LIMIT ?
            '''
            
            result = self.execute_query(query, (index_code, days), fetch=True, as_df=True)
            if result is not None and not result.empty:
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"시장 지수 이력 조회 실패: {e}")
            send_message(f"시장 지수 이력 조회 실패: {e}")
            return pd.DataFrame()
    
    def get_latest_market_indices(self) -> Dict[str, Dict[str, Any]]:
        """
        최신 시장 지수 정보 조회
        
        Returns:
            Dict[str, Dict[str, Any]]: 시장 지수별 최신 정보
        """
        try:
            query = '''
            SELECT m1.*
            FROM market_indices m1
            JOIN (
                SELECT index_code, MAX(date) as max_date
                FROM market_indices
                GROUP BY index_code
            ) m2
            ON m1.index_code = m2.index_code AND m1.date = m2.max_date
            '''
            
            result = self.execute_query(query, fetch=True, as_df=True)
            if result is not None and not result.empty:
                # 인덱스 코드별 데이터 정리
                indices_data = {}
                for _, row in result.iterrows():
                    index_code = row['index_code']
                    indices_data[index_code] = {
                        'date': row['date'],
                        'index_name': row['index_name'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume'],
                        'value': row['value'],
                        'change_value': row['change_value'],
                        'change_percent': row['change_percent']
                    }
                return indices_data
            else:
                return {}
            
        except Exception as e:
            logger.error(f"최신 시장 지수 조회 실패: {e}")
            send_message(f"최신 시장 지수 조회 실패: {e}")
            return {} 