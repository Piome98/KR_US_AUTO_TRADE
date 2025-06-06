"""
한국 주식 자동매매 - 거래 통계 데이터 모듈

거래 통계 데이터의 저장 및 조회 기능을 제공합니다.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.data.database.base import DatabaseManager

# 로깅 설정
logger = logging.getLogger(__name__)

class TradingStatsManager(DatabaseManager):
    """
    거래 통계 데이터 관리 클래스
    
    거래 통계 데이터의 저장 및 조회 기능을 제공합니다.
    """
    
    def __init__(self, db_path: str = "stock_data.db"):
        """
        거래 통계 데이터 매니저 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        super().__init__(db_path)
        self._initialize_table()
    
    def _initialize_table(self) -> None:
        """
        거래 통계 테이블 초기화
        """
        table_schema = '''
        CREATE TABLE IF NOT EXISTS trading_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            code TEXT,
            foreign_buy_volume INTEGER,
            foreign_sell_volume INTEGER,
            institutional_buy_volume INTEGER,
            institutional_sell_volume INTEGER,
            retail_buy_volume INTEGER,
            retail_sell_volume INTEGER,
            trade_value REAL,
            market_cap REAL,
            per REAL,
            pbr REAL,
            eps REAL,
            bps REAL,
            UNIQUE(date, code)
        )
        '''
        
        if self.create_table(table_schema):
            logger.info("거래 통계 테이블 초기화 완료")
        else:
            logger.error("거래 통계 테이블 초기화 실패")
    
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
        try:
            data = (
                date,
                code,
                stats_data.get('foreign_buy_volume', 0),
                stats_data.get('foreign_sell_volume', 0),
                stats_data.get('institutional_buy_volume', 0),
                stats_data.get('institutional_sell_volume', 0),
                stats_data.get('retail_buy_volume', 0),
                stats_data.get('retail_sell_volume', 0),
                stats_data.get('trade_value', 0),
                stats_data.get('market_cap', 0),
                stats_data.get('per', 0),
                stats_data.get('pbr', 0),
                stats_data.get('eps', 0),
                stats_data.get('bps', 0)
            )
            
            query = '''
            INSERT OR REPLACE INTO trading_stats 
            (date, code, foreign_buy_volume, foreign_sell_volume, 
            institutional_buy_volume, institutional_sell_volume, 
            retail_buy_volume, retail_sell_volume, trade_value, 
            market_cap, per, pbr, eps, bps)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            result = self.execute_query(query, data)
            if result:
                logger.debug(f"거래 통계 저장 완료: {code} {date}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"거래 통계 저장 실패: {e}")
            send_message(f"거래 통계 저장 실패: {e}", config.notification.discord_webhook_url)
            return False
    
    def get_trading_stats(self, code: str, days: int = 30) -> pd.DataFrame:
        """
        거래 통계 조회
        
        Args:
            code: 종목 코드
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 거래 통계 데이터프레임
        """
        try:
            query = '''
            SELECT date, foreign_buy_volume, foreign_sell_volume, 
                   institutional_buy_volume, institutional_sell_volume, 
                   retail_buy_volume, retail_sell_volume, trade_value, 
                   market_cap, per, pbr, eps, bps
            FROM trading_stats
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
            logger.error(f"거래 통계 조회 실패: {e}")
            send_message(f"거래 통계 조회 실패: {e}", config.notification.discord_webhook_url)
            return pd.DataFrame()
    
    def get_latest_trading_stats(self, code: str) -> Optional[Dict[str, Any]]:
        """
        종목의 최신 거래 통계 조회
        
        Args:
            code: 종목 코드
            
        Returns:
            Optional[Dict[str, Any]]: 최신 거래 통계 정보 또는 None
        """
        try:
            query = '''
            SELECT date, foreign_buy_volume, foreign_sell_volume, 
                   institutional_buy_volume, institutional_sell_volume, 
                   retail_buy_volume, retail_sell_volume, trade_value, 
                   market_cap, per, pbr, eps, bps
            FROM trading_stats
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
            logger.error(f"최신 거래 통계 조회 실패: {e}")
            send_message(f"최신 거래 통계 조회 실패: {e}", config.notification.discord_webhook_url)
            return None
    
    def calculate_investor_flow(self, code: str, days: int = 5) -> Dict[str, Any]:
        """
        투자자별 자금 흐름 계산
        
        Args:
            code: 종목 코드
            days: 조회할 일수
            
        Returns:
            Dict[str, Any]: 투자자별 자금 흐름 정보
        """
        try:
            stats_df = self.get_trading_stats(code, days)
            if stats_df.empty:
                return {
                    'foreign_net': 0,
                    'institutional_net': 0,
                    'retail_net': 0
                }
            
            # 투자자별 순매수 계산
            stats_df['foreign_net'] = stats_df['foreign_buy_volume'] - stats_df['foreign_sell_volume']
            stats_df['institutional_net'] = stats_df['institutional_buy_volume'] - stats_df['institutional_sell_volume']
            stats_df['retail_net'] = stats_df['retail_buy_volume'] - stats_df['retail_sell_volume']
            
            # 합계 계산
            foreign_net = stats_df['foreign_net'].sum()
            institutional_net = stats_df['institutional_net'].sum()
            retail_net = stats_df['retail_net'].sum()
            
            return {
                'foreign_net': foreign_net,
                'institutional_net': institutional_net,
                'retail_net': retail_net,
                'days': days
            }
            
        except Exception as e:
            logger.error(f"투자자별 자금 흐름 계산 실패: {e}")
            send_message(f"투자자별 자금 흐름 계산 실패: {e}", config.notification.discord_webhook_url)
            return {} 