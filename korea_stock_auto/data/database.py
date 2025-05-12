"""
한국 주식 자동매매 - 데이터베이스 모듈

주가 데이터 및 거래 내역 저장을 담당하는 모듈입니다.
주식 가격, 거래 내역, 기술적 지표, 시장 지수 등의 데이터를 관리합니다.
"""

import os
import json
import datetime
import logging
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, cast

from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.utils.db_helper import connect_db, execute_query

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
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """
        데이터베이스 초기화 및 테이블 생성
        
        필요한 모든 테이블이 없는 경우 생성합니다.
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                logger.error("데이터베이스 연결 실패")
                send_message("데이터베이스 연결 실패")
                return
            
            # 가격 데이터 테이블
            execute_query(conn, '''
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
            ''')
            
            # 거래 내역 테이블
            execute_query(conn, '''
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
            ''')
            
            # 위험 관리 이벤트 테이블 (손절, 익절 등)
            execute_query(conn, '''
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                code TEXT,
                event_type TEXT,
                price REAL,
                description TEXT
            )
            ''')
            
            # 기술적 지표 테이블 추가
            execute_query(conn, '''
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
            ''')
            
            # 시장 지수 데이터 테이블 추가
            execute_query(conn, '''
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
            ''')
            
            # 뉴스 이벤트 테이블 추가
            execute_query(conn, '''
            CREATE TABLE IF NOT EXISTS news_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                code TEXT,
                title TEXT,
                content TEXT,
                source TEXT,
                url TEXT,
                sentiment REAL,
                keywords TEXT
            )
            ''')
            
            # 거래 통계 테이블 추가
            execute_query(conn, '''
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
            ''')
            
            conn.close()
            logger.info("데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            send_message(f"데이터베이스 초기화 실패: {e}")
    
    def save_price_data(self, code: str, date: str, price_data: Dict[str, Any]) -> bool:
        """
        종목 가격 데이터 저장
        
        Args:
            code: 종목 코드
            date: 날짜 (YYYY-MM-DD)
            price_data: 가격 데이터 딕셔너리
            
        Returns:
            저장 성공 여부
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                logger.error("데이터베이스 연결 실패")
                return False
                
            data = (
                date,
                code,
                price_data.get('open', 0),
                price_data.get('high', 0),
                price_data.get('low', 0),
                price_data.get('close', 0),
                price_data.get('volume', 0)
            )
            
            execute_query(conn, '''
            INSERT OR REPLACE INTO price_data 
            (date, code, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', data)
            
            conn.close()
            logger.debug(f"가격 데이터 저장 완료: {code} {date}")
            return True
            
        except Exception as e:
            logger.error(f"가격 데이터 저장 실패: {e}")
            send_message(f"가격 데이터 저장 실패: {e}")
            return False
    
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
            저장 성공 여부
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                logger.error("데이터베이스 연결 실패")
                return False
                
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
            
            execute_query(conn, '''
            INSERT INTO transactions 
            (timestamp, code, action, price, quantity, amount, profit_loss, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            
            conn.close()
            
            log_msg = f"거래 내역 저장: {action} {code} {quantity}주 @ {price}원"
            logger.info(log_msg)
            send_message(log_msg)
            return True
            
        except Exception as e:
            logger.error(f"거래 내역 저장 실패: {e}")
            send_message(f"거래 내역 저장 실패: {e}")
            return False
    
    def log_risk_event(self, code, event_type, price, description=""):
        """
        위험 관리 이벤트 기록
        
        Args:
            code (str): 종목 코드
            event_type (str): 이벤트 유형 (STOP_LOSS/TAKE_PROFIT/TRAILING_STOP)
            price (float): 현재가
            description (str, optional): 추가 설명
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                return
                
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            data = (
                timestamp,
                code,
                event_type.upper(),
                price,
                description
            )
            
            execute_query(conn, '''
            INSERT INTO risk_events 
            (timestamp, code, event_type, price, description)
            VALUES (?, ?, ?, ?, ?)
            ''', data)
            
            conn.close()
            
            send_message(f"위험 관리 이벤트 기록: {event_type} {code} @ {price}원")
            
        except Exception as e:
            send_message(f"위험 관리 이벤트 기록 실패: {e}")
    
    def get_price_history(self, code, days=30):
        """
        종목의 가격 이력 조회
        
        Args:
            code (str): 종목 코드
            days (int): 조회할 일수
            
        Returns:
            pandas.DataFrame: 가격 이력 데이터프레임
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                return pd.DataFrame()
                
            query = '''
            SELECT date, open, high, low, close, volume
            FROM price_data
            WHERE code = ?
            ORDER BY date DESC
            LIMIT ?
            '''
            
            result = execute_query(conn, query, (code, days), fetch=True, as_df=True)
            conn.close()
            
            if result is not None and not result.empty:
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            send_message(f"가격 이력 조회 실패: {e}")
            return pd.DataFrame()
    
    def save_technical_indicators(self, code, date, indicators):
        """
        기술적 지표 저장
        
        Args:
            code (str): 종목 코드
            date (str): 날짜 (YYYY-MM-DD)
            indicators (dict): 기술적 지표 데이터
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                return
                
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
            
            execute_query(conn, '''
            INSERT OR REPLACE INTO technical_indicators 
            (date, code, rsi, macd, macd_signal, macd_hist, 
            sma_5, sma_20, sma_60, sma_120, 
            ema_5, ema_20, ema_60, 
            bollinger_upper, bollinger_middle, bollinger_lower, 
            atr, adx, obv)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            
            conn.close()
            
        except Exception as e:
            send_message(f"기술적 지표 저장 실패: {e}")
    
    def get_technical_indicators(self, code, days=30):
        """
        종목의 기술적 지표 이력 조회
        
        Args:
            code (str): 종목 코드
            days (int): 조회할 일수
            
        Returns:
            pandas.DataFrame: 기술적 지표 데이터프레임
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                return pd.DataFrame()
                
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
            
            result = execute_query(conn, query, (code, days), fetch=True, as_df=True)
            conn.close()
            
            if result is not None and not result.empty:
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            send_message(f"기술적 지표 조회 실패: {e}")
            return pd.DataFrame()
    
    def save_market_index(self, date, index_data):
        """
        시장 지수 데이터 저장
        
        Args:
            date (str): 날짜 (YYYY-MM-DD)
            index_data (dict): 지수 데이터
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                return
                
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
            
            execute_query(conn, '''
            INSERT OR REPLACE INTO market_indices 
            (date, index_code, index_name, open, high, low, close, 
            volume, value, change_value, change_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            
            conn.close()
            
        except Exception as e:
            send_message(f"시장 지수 데이터 저장 실패: {e}")
    
    def get_market_index_history(self, index_code, days=30):
        """
        시장 지수 이력 조회
        
        Args:
            index_code (str): 지수 코드 (KOSPI/KOSDAQ)
            days (int): 조회할 일수
            
        Returns:
            pandas.DataFrame: 지수 이력 데이터프레임
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                return pd.DataFrame()
                
            query = '''
            SELECT date, open, high, low, close, volume, value, change_value, change_percent
            FROM market_indices
            WHERE index_code = ?
            ORDER BY date DESC
            LIMIT ?
            '''
            
            result = execute_query(conn, query, (index_code, days), fetch=True, as_df=True)
            conn.close()
            
            if result is not None and not result.empty:
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            send_message(f"시장 지수 이력 조회 실패: {e}")
            return pd.DataFrame()
    
    def save_news_event(self, code, news_data):
        """
        종목 관련 뉴스 이벤트 저장
        
        Args:
            code (str): 종목 코드
            news_data (dict): 뉴스 데이터
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                return
                
            timestamp = news_data.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # JSON으로 변환할 키워드 리스트
            keywords = news_data.get('keywords', [])
            if keywords and isinstance(keywords, list):
                keywords_json = json.dumps(keywords, ensure_ascii=False)
            else:
                keywords_json = json.dumps([], ensure_ascii=False)
            
            data = (
                timestamp,
                code,
                news_data.get('title', ''),
                news_data.get('content', ''),
                news_data.get('source', ''),
                news_data.get('url', ''),
                news_data.get('sentiment', 0),
                keywords_json
            )
            
            execute_query(conn, '''
            INSERT INTO news_events 
            (timestamp, code, title, content, source, url, sentiment, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            
            conn.close()
            
        except Exception as e:
            send_message(f"뉴스 이벤트 저장 실패: {e}")
    
    def get_news_events(self, code, days=7):
        """
        종목 관련 뉴스 이벤트 조회
        
        Args:
            code (str): 종목 코드
            days (int): 조회할 일수
            
        Returns:
            pandas.DataFrame: 뉴스 이벤트 데이터프레임
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                return pd.DataFrame()
                
            # 현재 날짜로부터 days일 이전 날짜 계산
            from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
            
            query = '''
            SELECT timestamp, code, title, content, source, url, sentiment, keywords
            FROM news_events
            WHERE code = ? AND timestamp >= ?
            ORDER BY timestamp DESC
            '''
            
            result = execute_query(conn, query, (code, from_date), fetch=True, as_df=True)
            conn.close()
            
            if result is not None and not result.empty:
                # JSON 문자열을 파이썬 객체로 변환
                result['keywords'] = result['keywords'].apply(lambda x: json.loads(x) if x else [])
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            send_message(f"뉴스 이벤트 조회 실패: {e}")
            return pd.DataFrame()
    
    def save_trading_stats(self, code, date, stats_data):
        """
        거래 통계 데이터 저장
        
        Args:
            code (str): 종목 코드
            date (str): 날짜 (YYYY-MM-DD)
            stats_data (dict): 통계 데이터
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                return
                
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
            
            execute_query(conn, '''
            INSERT OR REPLACE INTO trading_stats 
            (date, code, foreign_buy_volume, foreign_sell_volume, 
            institutional_buy_volume, institutional_sell_volume, 
            retail_buy_volume, retail_sell_volume, trade_value, 
            market_cap, per, pbr, eps, bps)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            
            conn.close()
            
        except Exception as e:
            send_message(f"거래 통계 저장 실패: {e}")
    
    def get_trading_stats(self, code, days=30):
        """
        거래 통계 조회
        
        Args:
            code (str): 종목 코드
            days (int): 조회할 일수
            
        Returns:
            pandas.DataFrame: 거래 통계 데이터프레임
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                return pd.DataFrame()
                
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
            
            result = execute_query(conn, query, (code, days), fetch=True, as_df=True)
            conn.close()
            
            if result is not None and not result.empty:
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            send_message(f"거래 통계 조회 실패: {e}")
            return pd.DataFrame()
    
    def get_combined_data(self, code, days=30):
        """
        가격, 기술적 지표, 거래 통계를 모두 결합한 데이터 조회
        
        Args:
            code (str): 종목 코드
            days (int): 조회할 일수
            
        Returns:
            pandas.DataFrame: 결합된 데이터프레임
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
            send_message(f"결합 데이터 조회 실패: {e}")
            return pd.DataFrame() 