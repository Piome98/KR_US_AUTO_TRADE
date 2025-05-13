"""
한국 주식 자동매매 - 데이터베이스 기본 모듈

데이터베이스 연결 및 기본 작업을 위한 클래스를 제공합니다.
"""

import os
import logging
import sqlite3
from typing import Optional, Any, List, Dict, Union, Tuple

from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.utils.db_helper import connect_db, execute_query

# 로깅 설정
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    데이터베이스 관리 기본 클래스
    
    데이터베이스 연결 및 공통 작업을 처리하는 기본 클래스입니다.
    """
    
    def __init__(self, db_path: str = "stock_data.db"):
        """
        데이터베이스 매니저 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        
    def create_table(self, table_schema: str) -> bool:
        """
        테이블 생성
        
        Args:
            table_schema: 테이블 생성 SQL 문
            
        Returns:
            bool: 테이블 생성 성공 여부
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                logger.error("데이터베이스 연결 실패")
                send_message("데이터베이스 연결 실패")
                return False
            
            execute_query(conn, table_schema)
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"테이블 생성 실패: {e}")
            send_message(f"테이블 생성 실패: {e}")
            return False
    
    def execute_query(self, query: str, params: Tuple = (), fetch: bool = False, 
                     as_df: bool = False) -> Any:
        """
        SQL 쿼리 실행
        
        Args:
            query: SQL 쿼리문
            params: 쿼리 파라미터
            fetch: 결과 반환 여부
            as_df: 데이터프레임으로 변환 여부
            
        Returns:
            Any: 쿼리 실행 결과 또는 성공 여부
        """
        try:
            conn = connect_db(self.db_path)
            if not conn:
                logger.error("데이터베이스 연결 실패")
                send_message("데이터베이스 연결 실패")
                return None if fetch else False
            
            result = execute_query(conn, query, params, fetch=fetch, as_df=as_df)
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {e}")
            send_message(f"쿼리 실행 실패: {e}")
            return None if fetch else False 