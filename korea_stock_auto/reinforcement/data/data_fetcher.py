"""
강화학습을 위한 데이터 수집 모듈
"""

import os
import json
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from korea_stock_auto.utils import send_message
from korea_stock_auto.database import DatabaseManager


class DataFetcher:
    """강화학습 데이터 수집기"""
    
    def __init__(self, db_manager=None, data_dir="data"):
        """
        데이터 수집기 초기화
        
        Args:
            db_manager (DatabaseManager): 데이터베이스 관리자
            data_dir (str): 데이터 저장 디렉토리
        """
        self.db_manager = db_manager or DatabaseManager()
        self.data_dir = data_dir
        
        # 데이터 디렉토리 생성
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_from_database(self, code, start_date=None, end_date=None, save=False):
        """
        데이터베이스에서 데이터 가져오기
        
        Args:
            code (str): 종목 코드
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            save (bool): 파일로 저장 여부
            
        Returns:
            pd.DataFrame: 가져온 데이터
        """
        try:
            # 기본 날짜 설정
            if end_date is None:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                # 기본적으로 2년치 데이터 사용
                start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
            
            # 데이터베이스에서 데이터 가져오기
            query = """
            SELECT date, open, high, low, close, volume
            FROM daily_price
            WHERE code = ? AND date >= ? AND date <= ?
            ORDER BY date
            """
            
            data = self.db_manager.execute_query(query, (code, start_date, end_date))
            
            if data is None or len(data) == 0:
                send_message(f"데이터 없음: {code}, 기간: {start_date} ~ {end_date}")
                return None
            
            # 데이터프레임 변환
            df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 숫자 컬럼으로 변환
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # 파일로 저장
            if save:
                file_path = os.path.join(self.data_dir, f"{code}_{start_date}_{end_date}.csv")
                df.to_csv(file_path)
                send_message(f"데이터 저장 완료: {file_path}")
            
            send_message(f"{code} 데이터 가져오기 완료: {len(df)}개 행")
            return df
            
        except Exception as e:
            send_message(f"데이터베이스에서 데이터 가져오기 실패: {e}")
            return None
    
    def fetch_from_yfinance(self, symbol, start_date=None, end_date=None, save=False):
        """
        Yahoo Finance에서 데이터 가져오기
        
        Args:
            symbol (str): 종목 심볼 (예: '005930.KS')
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            save (bool): 파일로 저장 여부
            
        Returns:
            pd.DataFrame: 가져온 데이터
        """
        try:
            # 기본 날짜 설정
            if end_date is None:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                # 기본적으로 2년치 데이터 사용
                start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
            
            # Yahoo Finance에서 데이터 가져오기
            df = yf.download(symbol, start=start_date, end=end_date)
            
            if df is None or len(df) == 0:
                send_message(f"데이터 없음: {symbol}, 기간: {start_date} ~ {end_date}")
                return None
            
            # 컬럼명 소문자로 변경
            df.columns = [col.lower() for col in df.columns]
            
            # 'adj close' 컬럼 제거
            if 'adj close' in df.columns:
                df = df.drop('adj close', axis=1)
            
            # 파일로 저장
            if save:
                file_path = os.path.join(self.data_dir, f"{symbol.replace('.', '_')}_{start_date}_{end_date}.csv")
                df.to_csv(file_path)
                send_message(f"데이터 저장 완료: {file_path}")
            
            send_message(f"{symbol} 데이터 가져오기 완료: {len(df)}개 행")
            return df
            
        except Exception as e:
            send_message(f"Yahoo Finance에서 데이터 가져오기 실패: {e}")
            return None
    
    def load_from_file(self, file_path):
        """
        파일에서 데이터 로드
        
        Args:
            file_path (str): 데이터 파일 경로
            
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        try:
            # 파일 경로가 상대 경로인 경우 절대 경로로 변환
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.data_dir, file_path)
            
            # 파일 확장자 확인
            _, ext = os.path.splitext(file_path)
            
            # 파일 형식에 따라 로드
            if ext.lower() == '.csv':
                df = pd.read_csv(file_path)
                
                # 'date' 컬럼이 있는 경우 인덱스로 설정
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    
            elif ext.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            elif ext.lower() == '.pickle' or ext.lower() == '.pkl':
                df = pd.read_pickle(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {ext}")
            
            send_message(f"파일 로드 완료: {file_path}, {len(df)}개 행")
            return df
            
        except Exception as e:
            send_message(f"파일 로드 실패: {file_path}, {e}")
            return None
    
    def save_to_file(self, df, filename, format='csv'):
        """
        데이터를 파일로 저장
        
        Args:
            df (pd.DataFrame): 저장할 데이터프레임
            filename (str): 파일명
            format (str): 파일 형식 ('csv', 'parquet', 'pickle')
            
        Returns:
            str: 저장된 파일 경로 또는 None
        """
        try:
            # 파일 경로 생성
            file_path = os.path.join(self.data_dir, filename)
            
            # 확장자 확인 및 추가
            _, ext = os.path.splitext(file_path)
            if not ext:
                if format.lower() == 'csv':
                    file_path += '.csv'
                elif format.lower() == 'parquet':
                    file_path += '.parquet'
                elif format.lower() in ['pickle', 'pkl']:
                    file_path += '.pkl'
            
            # 파일 형식에 따라 저장
            if file_path.endswith('.csv'):
                df.to_csv(file_path)
            elif file_path.endswith('.parquet'):
                df.to_parquet(file_path)
            elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
                df.to_pickle(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {file_path}")
            
            send_message(f"데이터 저장 완료: {file_path}")
            return file_path
            
        except Exception as e:
            send_message(f"데이터 저장 실패: {e}")
            return None
    
    def update_stock_data(self, code, days=30, save=True):
        """
        최신 주가 데이터 업데이트
        
        Args:
            code (str): 종목 코드
            days (int): 가져올 일수
            save (bool): 파일로 저장 여부
            
        Returns:
            pd.DataFrame: 업데이트된 데이터
        """
        try:
            # 현재 날짜 및 시작 날짜 계산
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            # 날짜 형식 변환
            end_date_str = end_date.strftime('%Y-%m-%d')
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            # 데이터베이스에서 기존 데이터 확인
            existing_data = self.fetch_from_database(code, start_date=None, end_date=start_date_str, save=False)
            
            # 새 데이터 가져오기
            new_data = self.fetch_from_database(code, start_date=start_date_str, end_date=end_date_str, save=False)
            
            if existing_data is not None and new_data is not None:
                # 기존 데이터와 새 데이터 병합
                updated_data = pd.concat([existing_data, new_data])
                updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
                
                if save:
                    # 파일명 생성
                    filename = f"{code}_updated_{end_date_str}.csv"
                    self.save_to_file(updated_data, filename)
                
                send_message(f"{code} 데이터 업데이트 완료: {len(updated_data)}개 행")
                return updated_data
                
            elif new_data is not None:
                if save:
                    filename = f"{code}_{start_date_str}_{end_date_str}.csv"
                    self.save_to_file(new_data, filename)
                
                send_message(f"{code} 새 데이터 가져오기 완료: {len(new_data)}개 행")
                return new_data
                
            else:
                send_message(f"{code} 업데이트할 데이터 없음")
                return existing_data
                
        except Exception as e:
            send_message(f"데이터 업데이트 실패: {e}")
            return None 