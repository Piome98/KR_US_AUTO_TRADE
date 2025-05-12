"""
한국 주식 자동매매 - 데이터 전처리 모듈
강화학습 모델을 위한 기본 데이터 정제 및 전처리 기능
"""

import os
import numpy as np
import pandas as pd
import datetime
import logging
from typing import Optional, Dict, List, Any, Union, Tuple

logger = logging.getLogger("stock_auto")

class DataPreprocessor:
    """강화학습을 위한 기본 데이터 전처리 클래스"""
    
    def __init__(self, cache_dir=None):
        """
        데이터 프로세서 초기화
        
        Args:
            cache_dir (str): 캐시 디렉토리 경로
        """
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '../../../data/cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_stock_data(self, code: str, days: int = 30) -> pd.DataFrame:
        """
        주식 데이터 로드
        
        Args:
            code (str): 종목 코드
            days (int): 가져올 일수
            
        Returns:
            pd.DataFrame: 주식 데이터
        """
        try:
            # 캐시 파일 확인
            csv_path = os.path.join(self.cache_dir, f'{code}_stock_data.csv')
            
            if not os.path.exists(csv_path):
                logger.warning(f"{code} 주가 데이터 파일이 없습니다. 데이터를 먼저 수집해주세요.")
                return pd.DataFrame()
            
            # CSV 파일에서 데이터 읽기
            df = pd.read_csv(csv_path)
            
            # 날짜 형식 변환
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                
                # 최근 days일치 데이터만 사용
                end_date = df['date'].max()
                start_date = end_date - pd.Timedelta(days=days)
                df = df[df['date'] >= start_date]
            
            logger.info(f"{code} 주가 데이터 로드 완료: {len(df)}개 행")
            return df
            
        except Exception as e:
            logger.error(f"주가 데이터 로드 실패: {e}", exc_info=True)
            return pd.DataFrame()
    
    def load_realtime_data(self, code: str) -> Dict[str, Any]:
        """
        실시간 데이터 로드
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict: 실시간 데이터
        """
        try:
            # 실시간 API 데이터 가져오기
            realtime_path = os.path.join(self.cache_dir, f'realtime_{code}.json')
            
            if not os.path.exists(realtime_path):
                logger.warning(f"{code} 실시간 데이터 파일이 없습니다.")
                return {}
            
            import json
            with open(realtime_path, 'r') as f:
                realtime_data = json.load(f)
            
            # 실시간 데이터의 유효성 확인
            cache_time = datetime.datetime.fromtimestamp(os.path.getmtime(realtime_path))
            current_time = datetime.datetime.now()
            
            # 1시간 이내의 데이터인 경우에만 사용
            if (current_time - cache_time).total_seconds() < 3600:
                logger.info(f"{code} 실시간 데이터 로드 완료 (캐시 시간: {cache_time.strftime('%Y-%m-%d %H:%M:%S')})")
                return realtime_data
            else:
                logger.warning(f"{code} 실시간 데이터가 오래되었습니다. (캐시 시간: {cache_time.strftime('%Y-%m-%d %H:%M:%S')})")
                return {}
                
        except Exception as e:
            logger.error(f"실시간 데이터 로드 실패: {e}", exc_info=True)
            return {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정제 (결측치 및 이상치 처리)
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 정제된 데이터프레임
        """
        if df.empty:
            return df
        
        try:
            # 원본 데이터 복사
            df_clean = df.copy()
            
            # 결측치 처리
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
            
            # 전진 채우기 -> 후진 채우기 -> 0 채우기 순서로 적용
            df_clean = df_clean.fillna(method='ffill')
            df_clean = df_clean.fillna(method='bfill')
            df_clean = df_clean.fillna(0)
            
            # 날짜 정렬 (있는 경우)
            if 'date' in df_clean.columns:
                df_clean = df_clean.sort_values('date')
            
            logger.info(f"데이터 정제 완료: {len(df_clean)}개 행")
            return df_clean
            
        except Exception as e:
            logger.error(f"데이터 정제 실패: {e}", exc_info=True)
            return df
    
    def merge_with_realtime(self, df: pd.DataFrame, realtime_data: Dict[str, Any], code: str) -> pd.DataFrame:
        """
        과거 데이터와 실시간 데이터 통합
        
        Args:
            df (pd.DataFrame): 과거 데이터
            realtime_data (dict): 실시간 데이터
            code (str): 종목 코드
            
        Returns:
            pd.DataFrame: 통합된 데이터프레임
        """
        if df.empty or not realtime_data:
            return df
        
        try:
            # 원본 데이터 복사
            merged_df = df.copy()
            
            # 오늘 날짜
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            
            # 날짜 컬럼이 있는지 확인
            if 'date' not in merged_df.columns:
                logger.warning("날짜 컬럼이 없어 실시간 데이터를 통합할 수 없습니다.")
                return merged_df
            
            # 오늘 데이터가 이미 있는지 확인
            today_mask = merged_df['date'].dt.strftime('%Y-%m-%d') == today
            
            if today_mask.any():
                # 오늘 데이터 업데이트
                merged_df.loc[today_mask, 'close'] = realtime_data.get('current_price', merged_df.loc[today_mask, 'close'].values[0])
                merged_df.loc[today_mask, 'volume'] = realtime_data.get('volume', merged_df.loc[today_mask, 'volume'].values[0])
                
                # 추가 API 지표 업데이트
                if 'bid_ask_ratio' not in merged_df.columns and 'bid_ask_ratio' in realtime_data:
                    merged_df.loc[today_mask, 'bid_ask_ratio'] = realtime_data.get('bid_ask_ratio', 1.0)
                    
                if 'market_pressure' not in merged_df.columns and 'market_pressure' in realtime_data:
                    merged_df.loc[today_mask, 'market_pressure'] = realtime_data.get('market_pressure', 0.5)
            else:
                # 오늘 데이터 추가
                today_data = {
                    'date': pd.to_datetime(today),
                    'code': code,
                    'open': realtime_data.get('current_price', 0),  # 실시간이므로 현재가를 시가로도 사용
                    'high': realtime_data.get('current_price', 0),
                    'low': realtime_data.get('current_price', 0),
                    'close': realtime_data.get('current_price', 0),
                    'volume': realtime_data.get('volume', 0),
                }
                
                # 추가 API 지표
                if 'bid_ask_ratio' in realtime_data:
                    today_data['bid_ask_ratio'] = realtime_data.get('bid_ask_ratio', 1.0)
                    
                if 'market_pressure' in realtime_data:
                    today_data['market_pressure'] = realtime_data.get('market_pressure', 0.5)
                
                # 오늘 데이터 추가
                merged_df = pd.concat([merged_df, pd.DataFrame([today_data])], ignore_index=True)
            
            logger.info(f"{code} 과거 데이터와 실시간 데이터 통합 완료")
            return merged_df
            
        except Exception as e:
            logger.error(f"데이터 통합 실패: {e}", exc_info=True)
            return df
    
    def save_processed_data(self, df: pd.DataFrame, code: str, suffix: str = 'preprocessed') -> str:
        """
        처리된 데이터 저장
        
        Args:
            df (pd.DataFrame): 저장할 데이터프레임
            code (str): 종목 코드
            suffix (str): 파일명 접미사
            
        Returns:
            str: 저장된 파일 경로 또는 빈 문자열
        """
        if df.empty:
            logger.warning(f"{code} 저장할 데이터가 없습니다.")
            return ""
        
        try:
            # 저장 경로
            save_path = os.path.join(self.cache_dir, f'{code}_{suffix}.csv')
            
            # 데이터 저장
            df.to_csv(save_path, index=False)
            
            logger.info(f"{code} {suffix} 데이터 저장 완료: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"데이터 저장 실패: {e}", exc_info=True)
            return "" 