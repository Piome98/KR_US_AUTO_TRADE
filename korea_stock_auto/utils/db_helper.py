"""
한국 주식 자동매매 - 데이터베이스 도우미 모듈

데이터베이스 연결 및 쿼리 실행 공통 기능을 제공합니다.
"""

import sqlite3
import pandas as pd
from typing import Optional, Union, List, Dict, Any, Tuple, cast
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.config import get_config

def connect_db(db_path: str) -> Optional[sqlite3.Connection]:
    """
    데이터베이스 연결 생성
    
    Args:
        db_path: 데이터베이스 파일 경로
        
    Returns:
        연결 객체 또는 연결 실패 시 None
    """
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except Exception as e:
        send_message(f"데이터베이스 연결 실패: {e}", get_config().notification.discord_webhook_url)
        return None

def execute_query(
    conn: sqlite3.Connection, 
    query: str, 
    params: Optional[Union[Tuple, Dict[str, Any]]] = None, 
    fetch: bool = False, 
    fetchall: bool = True, 
    as_df: bool = False
) -> Optional[Union[List[Tuple], Tuple, pd.DataFrame]]:
    """
    SQL 쿼리 실행
    
    Args:
        conn: 데이터베이스 연결 객체
        query: 실행할 SQL 쿼리
        params: 쿼리 파라미터 (튜플 또는 딕셔너리)
        fetch: 결과를 반환할지 여부
        fetchall: 모든 결과를 반환할지 여부 (False면 첫 번째 결과만)
        as_df: 결과를 DataFrame으로 반환할지 여부
        
    Returns:
        쿼리 결과 (리스트, 튜플, DataFrame) 또는 실패 시 None
    """
    try:
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        result = None
        
        if fetch:
            if fetchall:
                if as_df:
                    # DataFrame으로 결과 반환
                    cols = [col[0] for col in cursor.description]
                    result = pd.DataFrame(cursor.fetchall(), columns=cols)
                else:
                    result = cursor.fetchall()
            else:
                result = cursor.fetchone()
        
        conn.commit()
        return result
    
    except Exception as e:
        error_msg = f"쿼리 실행 실패: {e}"
        send_message(error_msg, get_config().notification.discord_webhook_url)
        conn.rollback()
        return None 