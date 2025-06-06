"""
한국 주식 자동매매 - 거래 데이터 모듈

거래 내역 및 위험 관리 이벤트 데이터의 저장 및 조회 기능을 제공합니다.
"""

import datetime
import logging
import pandas as pd
from typing import Dict, Any, Optional, List

from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.data.database.base import DatabaseManager

# 로깅 설정
logger = logging.getLogger(__name__)

class TransactionDataManager(DatabaseManager):
    """
    거래 데이터 관리 클래스
    
    거래 내역 및 위험 관리 이벤트 데이터의 저장 및 조회 기능을 제공합니다.
    """
    
    def __init__(self, db_path: str = "stock_data.db"):
        """
        거래 데이터 매니저 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        super().__init__(db_path)
        self._initialize_tables()
    
    def _initialize_tables(self) -> None:
        """
        거래 관련 테이블 초기화
        """
        # 거래 내역 테이블
        transactions_schema = '''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            code TEXT,
            action TEXT,
            price REAL,
            quantity INTEGER,
            amount REAL,
            profit_loss REAL,
            state TEXT
        )
        '''
        
        # 위험 관리 이벤트 테이블
        risk_events_schema = '''
        CREATE TABLE IF NOT EXISTS risk_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            code TEXT,
            event_type TEXT,
            price REAL,
            description TEXT
        )
        '''
        
        if self.create_table(transactions_schema) and self.create_table(risk_events_schema):
            logger.info("거래 데이터 테이블 초기화 완료")
        else:
            logger.error("거래 데이터 테이블 초기화 실패")
    
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
        try:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            amount = price * quantity
            
            # 손익 계산 (매도시에만)
            profit_loss = 0
            if action.upper() == 'SELL' and entry_price:
                profit_loss = (price - entry_price) * quantity
            
            data = (
                timestamp,
                code,
                action.upper(),
                price,
                quantity,
                amount,
                profit_loss,
                state or "COMPLETE"
            )
            
            query = '''
            INSERT INTO transactions 
            (timestamp, code, action, price, quantity, amount, profit_loss, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            result = self.execute_query(query, data)
            if result:
                log_msg = f"거래 내역 저장: {action} {code} {quantity}주 @ {price}원"
                logger.info(log_msg)
                send_message(log_msg, config.notification.discord_webhook_url)
                return True
            return False
            
        except Exception as e:
            logger.error(f"거래 내역 저장 실패: {e}")
            send_message(f"거래 내역 저장 실패: {e}", config.notification.discord_webhook_url)
            return False
    
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
        try:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            data = (
                timestamp,
                code,
                event_type.upper(),
                price,
                description
            )
            
            query = '''
            INSERT INTO risk_events 
            (timestamp, code, event_type, price, description)
            VALUES (?, ?, ?, ?, ?)
            '''
            
            result = self.execute_query(query, data)
            if result:
                log_msg = f"위험 관리 이벤트 기록: {event_type} {code} @ {price}원"
                logger.info(log_msg)
                send_message(log_msg, config.notification.discord_webhook_url)
                return True
            return False
            
        except Exception as e:
            logger.error(f"위험 관리 이벤트 기록 실패: {e}")
            send_message(f"위험 관리 이벤트 기록 실패: {e}", config.notification.discord_webhook_url)
            return False
    
    def get_transactions(self, code: Optional[str] = None, days: int = 30) -> pd.DataFrame:
        """
        거래 내역 조회
        
        Args:
            code: 종목 코드 (None인 경우 전체 조회)
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 거래 내역 데이터프레임
        """
        try:
            # 현재 날짜로부터 days일 이전 날짜 계산
            from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
            
            if code:
                query = '''
                SELECT * FROM transactions
                WHERE code = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                '''
                params = (code, from_date)
            else:
                query = '''
                SELECT * FROM transactions
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                '''
                params = (from_date,)
            
            result = self.execute_query(query, params, fetch=True, as_df=True)
            if result is not None and not result.empty:
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"거래 내역 조회 실패: {e}")
            send_message(f"거래 내역 조회 실패: {e}", config.notification.discord_webhook_url)
            return pd.DataFrame()
    
    def get_risk_events(self, code: Optional[str] = None, days: int = 30) -> pd.DataFrame:
        """
        위험 관리 이벤트 조회
        
        Args:
            code: 종목 코드 (None인 경우 전체 조회)
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 위험 관리 이벤트 데이터프레임
        """
        try:
            # 현재 날짜로부터 days일 이전 날짜 계산
            from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
            
            if code:
                query = '''
                SELECT * FROM risk_events
                WHERE code = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                '''
                params = (code, from_date)
            else:
                query = '''
                SELECT * FROM risk_events
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                '''
                params = (from_date,)
            
            result = self.execute_query(query, params, fetch=True, as_df=True)
            if result is not None and not result.empty:
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"위험 관리 이벤트 조회 실패: {e}")
            send_message(f"위험 관리 이벤트 조회 실패: {e}", config.notification.discord_webhook_url)
            return pd.DataFrame()
    
    def get_trading_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        거래 요약 정보 조회
        
        Args:
            days: 조회할 일수
            
        Returns:
            Dict[str, Any]: 거래 요약 정보
        """
        try:
            # 현재 날짜로부터 days일 이전 날짜 계산
            from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
            
            # 전체 거래 내역 조회
            transactions = self.get_transactions(days=days)
            
            if transactions.empty:
                return {
                    'total_trades': 0,
                    'total_profit_loss': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'avg_loss': 0
                }
            
            # 매수/매도 건수
            buy_count = len(transactions[transactions['action'] == 'BUY'])
            sell_count = len(transactions[transactions['action'] == 'SELL'])
            
            # 손익 계산
            profit_trades = transactions[transactions['profit_loss'] > 0]
            loss_trades = transactions[transactions['profit_loss'] < 0]
            
            total_profit_loss = transactions['profit_loss'].sum()
            win_count = len(profit_trades)
            loss_count = len(loss_trades)
            
            # 승률 계산 (매도 거래 중)
            win_rate = win_count / sell_count if sell_count > 0 else 0
            
            # 평균 수익/손실
            avg_profit = profit_trades['profit_loss'].mean() if not profit_trades.empty else 0
            avg_loss = loss_trades['profit_loss'].mean() if not loss_trades.empty else 0
            
            return {
                'total_trades': len(transactions),
                'buy_count': buy_count,
                'sell_count': sell_count,
                'total_profit_loss': total_profit_loss,
                'win_count': win_count,
                'loss_count': loss_count,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss
            }
            
        except Exception as e:
            logger.error(f"거래 요약 정보 조회 실패: {e}")
            send_message(f"거래 요약 정보 조회 실패: {e}", config.notification.discord_webhook_url)
            return {} 