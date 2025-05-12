"""
한국 주식 자동매매 - 시장 데이터 통합 모듈
강화학습 모델을 위한 다양한 시장 데이터 통합 기능
"""

import os
import json
import pandas as pd
import numpy as np
import datetime
import logging
from typing import Optional, Dict, List, Any, Union, Tuple

logger = logging.getLogger("stock_auto")

class MarketDataIntegrator:
    """시장 데이터 통합 클래스"""
    
    def __init__(self, cache_dir=None):
        """
        시장 데이터 통합기 초기화
        
        Args:
            cache_dir (str): 캐시 디렉토리 경로
        """
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '../../../data/cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def combine_market_data(self, stock_data: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        주가 데이터와 시장 데이터 통합하여 강화학습 입력으로 사용
        
        Args:
            stock_data (pd.DataFrame): 주가 데이터
            code (str): 종목 코드
            
        Returns:
            pd.DataFrame: 시장 데이터가 포함된 강화학습용 데이터프레임
        """
        if stock_data.empty:
            logger.warning(f"{code} 주가 데이터가 없습니다.")
            return pd.DataFrame()
        
        try:
            # 원본 데이터 복사
            merged_df = stock_data.copy()
            
            # 거래량 상위 종목 데이터 추가
            merged_df = self._add_volume_rank_data(merged_df, code)
            
            # 거래량 급증 종목 데이터 추가
            merged_df = self._add_volume_increasing_data(merged_df, code)
            
            # 업종 지수 데이터 추가
            merged_df = self._add_sector_index_data(merged_df, code)
            
            # 투자자별 매매 데이터 추가
            merged_df = self._add_investor_data(merged_df, code)
            
            logger.info(f"{code} 주가 데이터와 시장 데이터 통합 완료: {len(merged_df.columns)}개 특성")
            return merged_df
            
        except Exception as e:
            logger.error(f"시장 데이터 통합 실패: {e}", exc_info=True)
            return stock_data
    
    def _add_volume_rank_data(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        거래량 상위 종목 데이터 추가
        
        Args:
            df (pd.DataFrame): 대상 데이터프레임
            code (str): 종목 코드
            
        Returns:
            pd.DataFrame: 거래량 상위 데이터가 추가된 데이터프레임
        """
        # 원본 데이터 복사
        df_result = df.copy()
        
        try:
            # 거래량 상위 종목 데이터 가져오기
            volume_rank_path = os.path.join(self.cache_dir, 'volume_rank_data.json')
            
            if not os.path.exists(volume_rank_path):
                return df_result
            
            with open(volume_rank_path, 'r') as f:
                volume_rank_data = json.load(f)
            
            if not volume_rank_data:
                return df_result
            
            # 대상 종목이 상위 종목에 포함되는지 확인
            codes = [item.get('code', '') for item in volume_rank_data]
            
            if code in codes:
                rank = codes.index(code) + 1
                df_result['volume_rank'] = rank
                df_result['is_top_volume'] = 1
            else:
                df_result['volume_rank'] = len(volume_rank_data) + 1  # 목록에 없으면 최하위 취급
                df_result['is_top_volume'] = 0
            
            # 거래량 순위 정규화 (0~1 범위)
            if 'volume_rank' in df_result.columns:
                # 0에 가까울수록 상위 순위
                df_result['volume_rank_norm'] = 1.0 / (df_result['volume_rank'] + 1)
            
            return df_result
            
        except Exception as e:
            logger.error(f"거래량 상위 데이터 추가 실패: {e}", exc_info=True)
            return df
    
    def _add_volume_increasing_data(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        거래량 급증 종목 데이터 추가
        
        Args:
            df (pd.DataFrame): 대상 데이터프레임
            code (str): 종목 코드
            
        Returns:
            pd.DataFrame: 거래량 급증 데이터가 추가된 데이터프레임
        """
        # 원본 데이터 복사
        df_result = df.copy()
        
        try:
            # 거래량 급증 종목 데이터 가져오기
            volume_increasing_path = os.path.join(self.cache_dir, 'volume_increasing_data.json')
            
            if not os.path.exists(volume_increasing_path):
                return df_result
            
            with open(volume_increasing_path, 'r') as f:
                volume_increasing_data = json.load(f)
            
            if not volume_increasing_data:
                return df_result
            
            # 대상 종목이 급증 종목에 포함되는지 확인
            codes = [item.get('code', '') for item in volume_increasing_data]
            
            if code in codes:
                index = codes.index(code)
                df_result['volume_increase_rank'] = index + 1
                df_result['volume_increase_ratio'] = volume_increasing_data[index].get('volume_ratio', 1.0)
                df_result['is_volume_increasing'] = 1
            else:
                df_result['volume_increase_rank'] = len(volume_increasing_data) + 1  # 목록에 없으면 최하위 취급
                df_result['volume_increase_ratio'] = 1.0  # 기본값
                df_result['is_volume_increasing'] = 0
            
            # 거래량 급증 순위 정규화 (0~1 범위)
            if 'volume_increase_rank' in df_result.columns:
                # 0에 가까울수록 상위 순위
                df_result['volume_increase_rank_norm'] = 1.0 / (df_result['volume_increase_rank'] + 1)
            
            # 거래량 급증 비율 정규화 (로그 스케일)
            if 'volume_increase_ratio' in df_result.columns:
                # 1.0이 기준(정상), 1.0보다 크면 증가, 작으면 감소
                df_result['volume_increase_ratio_norm'] = np.log10(df_result['volume_increase_ratio'].clip(0.1, 10)) + 1.0
                df_result['volume_increase_ratio_norm'] = df_result['volume_increase_ratio_norm'] / 2.0  # 0~1 범위로 조정
            
            return df_result
            
        except Exception as e:
            logger.error(f"거래량 급증 데이터 추가 실패: {e}", exc_info=True)
            return df
    
    def _add_sector_index_data(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        업종 지수 데이터 추가
        
        Args:
            df (pd.DataFrame): 대상 데이터프레임
            code (str): 종목 코드
            
        Returns:
            pd.DataFrame: 업종 지수 데이터가 추가된 데이터프레임
        """
        # 원본 데이터 복사
        df_result = df.copy()
        
        try:
            # 업종 정보 가져오기
            stock_info_path = os.path.join(self.cache_dir, f'{code}_info.json')
            
            if not os.path.exists(stock_info_path):
                return df_result
            
            with open(stock_info_path, 'r') as f:
                stock_info = json.load(f)
            
            # 업종 코드 확인
            sector_code = stock_info.get('sector_code', '')
            
            if not sector_code:
                return df_result
            
            # 업종 지수 데이터 가져오기
            sector_path = os.path.join(self.cache_dir, f'sector_{sector_code}.csv')
            
            if not os.path.exists(sector_path):
                return df_result
            
            sector_df = pd.read_csv(sector_path)
            
            if sector_df.empty:
                return df_result
            
            # 날짜 형식 변환
            if 'date' in sector_df.columns and 'date' in df_result.columns:
                sector_df['date'] = pd.to_datetime(sector_df['date'])
                df_result['date'] = pd.to_datetime(df_result['date'])
                
                # 날짜 기준으로 데이터 합치기
                df_result = pd.merge(
                    df_result, 
                    sector_df[['date', 'close', 'change_rate']], 
                    on='date', 
                    how='left',
                    suffixes=('', '_sector')
                )
                
                # NaN 값 처리
                df_result = df_result.fillna(method='ffill')
                df_result = df_result.fillna(method='bfill')
                df_result = df_result.fillna(0)
                
                # 업종 대비 상대 강도 (종목 등락률 - 업종 등락률)
                if 'change_rate' in df_result.columns and 'change_rate_sector' in df_result.columns:
                    df_result['relative_strength'] = df_result['change_rate'] - df_result['change_rate_sector']
                    
                    # 상대 강도 정규화 (-1 ~ 1 범위)
                    df_result['relative_strength_norm'] = df_result['relative_strength'].clip(-10, 10) / 10
            
            return df_result
            
        except Exception as e:
            logger.error(f"업종 지수 데이터 추가 실패: {e}", exc_info=True)
            return df
    
    def _add_investor_data(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        투자자별 매매 데이터 추가
        
        Args:
            df (pd.DataFrame): 대상 데이터프레임
            code (str): 종목 코드
            
        Returns:
            pd.DataFrame: 투자자별 매매 데이터가 추가된 데이터프레임
        """
        # 원본 데이터 복사
        df_result = df.copy()
        
        try:
            # 투자자별 매매 데이터 가져오기
            investor_path = os.path.join(self.cache_dir, f'{code}_investor.csv')
            
            if not os.path.exists(investor_path):
                return df_result
            
            investor_df = pd.read_csv(investor_path)
            
            if investor_df.empty:
                return df_result
            
            # 날짜 형식 변환
            if 'date' in investor_df.columns and 'date' in df_result.columns:
                investor_df['date'] = pd.to_datetime(investor_df['date'])
                df_result['date'] = pd.to_datetime(df_result['date'])
                
                # 날짜 기준으로 데이터 합치기
                for investor_type in ['individual', 'foreign', 'institution']:
                    if f'{investor_type}_net' in investor_df.columns:
                        temp_df = investor_df[['date', f'{investor_type}_net']]
                        df_result = pd.merge(df_result, temp_df, on='date', how='left')
                        
                        # NaN 값 처리
                        df_result[f'{investor_type}_net'] = df_result[f'{investor_type}_net'].fillna(0)
                        
                        # 거래 대금 대비 순매수 비율 계산
                        if 'volume' in df_result.columns and 'close' in df_result.columns:
                            trade_amount = df_result['volume'] * df_result['close']
                            df_result[f'{investor_type}_net_ratio'] = df_result[f'{investor_type}_net'] / trade_amount
                            
                            # -1 ~ 1 범위로 클리핑 및 정규화
                            df_result[f'{investor_type}_net_ratio'] = df_result[f'{investor_type}_net_ratio'].clip(-1, 1)
                            df_result[f'{investor_type}_net_ratio_norm'] = (df_result[f'{investor_type}_net_ratio'] + 1) / 2
            
            return df_result
            
        except Exception as e:
            logger.error(f"투자자별 매매 데이터 추가 실패: {e}", exc_info=True)
            return df 