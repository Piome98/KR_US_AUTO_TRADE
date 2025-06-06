"""
한국 주식 자동매매 - 뉴스 데이터 모듈

뉴스 이벤트 데이터의 저장 및 조회 기능을 제공합니다.
"""

import json
import datetime
import logging
import pandas as pd
from typing import Dict, List, Any, Optional

from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.data.database.base import DatabaseManager

# 로깅 설정
logger = logging.getLogger(__name__)

class NewsDataManager(DatabaseManager):
    """
    뉴스 데이터 관리 클래스
    
    뉴스 이벤트 데이터의 저장 및 조회 기능을 제공합니다.
    """
    
    def __init__(self, db_path: str = "stock_data.db"):
        """
        뉴스 데이터 매니저 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        super().__init__(db_path)
        self._initialize_table()
    
    def _initialize_table(self) -> None:
        """
        뉴스 이벤트 테이블 초기화
        """
        table_schema = '''
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
        '''
        
        if self.create_table(table_schema):
            logger.info("뉴스 이벤트 테이블 초기화 완료")
        else:
            logger.error("뉴스 이벤트 테이블 초기화 실패")
    
    def save_news_event(self, code: str, news_data: Dict[str, Any]) -> bool:
        """
        종목 관련 뉴스 이벤트 저장
        
        Args:
            code: 종목 코드
            news_data: 뉴스 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
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
            
            query = '''
            INSERT INTO news_events 
            (timestamp, code, title, content, source, url, sentiment, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            result = self.execute_query(query, data)
            if result:
                logger.debug(f"뉴스 이벤트 저장 완료: {code} - {news_data.get('title', '')}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"뉴스 이벤트 저장 실패: {e}")
            send_message(f"뉴스 이벤트 저장 실패: {e}", config.notification.discord_webhook_url)
            return False
    
    def get_news_events(self, code: Optional[str] = None, days: int = 7) -> pd.DataFrame:
        """
        종목 관련 뉴스 이벤트 조회
        
        Args:
            code: 종목 코드 (None인 경우 전체 조회)
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 뉴스 이벤트 데이터프레임
        """
        try:
            # 현재 날짜로부터 days일 이전 날짜 계산
            from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
            
            if code:
                query = '''
                SELECT timestamp, code, title, content, source, url, sentiment, keywords
                FROM news_events
                WHERE code = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                '''
                params = (code, from_date)
            else:
                query = '''
                SELECT timestamp, code, title, content, source, url, sentiment, keywords
                FROM news_events
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                '''
                params = (from_date,)
            
            result = self.execute_query(query, params, fetch=True, as_df=True)
            if result is not None and not result.empty:
                # JSON 문자열을 파이썬 객체로 변환
                result['keywords'] = result['keywords'].apply(lambda x: json.loads(x) if x else [])
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"뉴스 이벤트 조회 실패: {e}")
            send_message(f"뉴스 이벤트 조회 실패: {e}", config.notification.discord_webhook_url)
            return pd.DataFrame()
    
    def get_news_by_sentiment(self, min_sentiment: float = 0.5, days: int = 7) -> pd.DataFrame:
        """
        감성 분석 점수로 뉴스 이벤트 필터링
        
        Args:
            min_sentiment: 최소 감성 점수 (0.0 ~ 1.0)
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 뉴스 이벤트 데이터프레임
        """
        try:
            # 현재 날짜로부터 days일 이전 날짜 계산
            from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
            
            query = '''
            SELECT timestamp, code, title, content, source, url, sentiment, keywords
            FROM news_events
            WHERE sentiment >= ? AND timestamp >= ?
            ORDER BY sentiment DESC, timestamp DESC
            '''
            
            result = self.execute_query(query, (min_sentiment, from_date), fetch=True, as_df=True)
            if result is not None and not result.empty:
                # JSON 문자열을 파이썬 객체로 변환
                result['keywords'] = result['keywords'].apply(lambda x: json.loads(x) if x else [])
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"감성 분석 기반 뉴스 조회 실패: {e}")
            send_message(f"감성 분석 기반 뉴스 조회 실패: {e}", config.notification.discord_webhook_url)
            return pd.DataFrame()
    
    def search_news_by_keyword(self, keyword: str, days: int = 30) -> pd.DataFrame:
        """
        키워드로 뉴스 검색
        
        Args:
            keyword: 검색할 키워드
            days: 조회할 일수
            
        Returns:
            pd.DataFrame: 뉴스 이벤트 데이터프레임
        """
        try:
            # 현재 날짜로부터 days일 이전 날짜 계산
            from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
            
            # 제목과 내용에서 키워드 검색
            query = '''
            SELECT timestamp, code, title, content, source, url, sentiment, keywords
            FROM news_events
            WHERE (title LIKE ? OR content LIKE ?) AND timestamp >= ?
            ORDER BY timestamp DESC
            '''
            
            search_pattern = f'%{keyword}%'
            params = (search_pattern, search_pattern, from_date)
            
            result = self.execute_query(query, params, fetch=True, as_df=True)
            if result is not None and not result.empty:
                # JSON 문자열을 파이썬 객체로 변환
                result['keywords'] = result['keywords'].apply(lambda x: json.loads(x) if x else [])
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"키워드 기반 뉴스 검색 실패: {e}")
            send_message(f"키워드 기반 뉴스 검색 실패: {e}", config.notification.discord_webhook_url)
            return pd.DataFrame() 