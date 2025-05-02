"""
한국 주식 자동매매 - 데이터베이스 도우미 모듈
데이터베이스 연결 및 쿼리 실행 공통 기능
"""

import sqlite3
import pandas as pd
from korea_stock_auto.utils.utils import send_message

def connect_db(db_path):
    """
    데이터베이스 연결 생성
    
    Args:
        db_path (str): 데이터베이스 파일 경로
        
    Returns:
        sqlite3.Connection: 연결 객체
    """
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except Exception as e:
        send_message(f"데이터베이스 연결 실패: {e}")
        return None

def execute_query(conn, query, params=None, fetch=False, fetchall=True, as_df=False):
    """
    SQL 쿼리 실행
    
    Args:
        conn (sqlite3.Connection): 데이터베이스 연결 객체
        query (str): 실행할 SQL 쿼리
        params (tuple or dict, optional): 쿼리 파라미터
        fetch (bool): 결과를 반환할지 여부
        fetchall (bool): 모든 결과를 반환할지 여부 (False면 첫 번째 결과만)
        as_df (bool): 결과를 DataFrame으로 반환할지 여부
        
    Returns:
        list or pandas.DataFrame or None: 쿼리 결과
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
        send_message(f"쿼리 실행 실패: {e}")
        conn.rollback()
        return None 