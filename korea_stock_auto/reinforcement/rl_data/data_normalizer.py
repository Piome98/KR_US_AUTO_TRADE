"""
한국 주식 자동매매 - 데이터 정규화 모듈
강화학습 모델을 위한 데이터 정규화 기능
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Optional, Dict, List, Any, Union, Tuple

logger = logging.getLogger("stock_auto")

class DataNormalizer:
    """데이터 정규화 클래스"""
    
    def __init__(self, scaler_path=None, feature_range=(0, 1)):
        """
        데이터 정규화 초기화
        
        Args:
            scaler_path (str): 스케일러 저장 경로
            feature_range (tuple): 정규화 범위
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.scaler_path = scaler_path or 'models/feature_scaler.pkl'
        
        # 스케일러 불러오기 시도
        self._load_scaler()
    
    def _load_scaler(self):
        """저장된 스케일러 불러오기"""
        try:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("스케일러 로드 성공")
        except Exception as e:
            logger.error(f"스케일러 로드 실패: {e}")
    
    def _save_scaler(self):
        """스케일러 저장"""
        try:
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"스케일러 저장 완료: {self.scaler_path}")
        except Exception as e:
            logger.error(f"스케일러 저장 실패: {e}")
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정규화
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 정규화된 데이터프레임
        """
        # 원본 데이터 복사
        df_normalized = df.copy()
        
        try:
            # 가격 관련 컬럼 (Min-Max 정규화)
            price_columns = ['open', 'high', 'low', 'close']
            available_columns = [col for col in price_columns if col in df_normalized.columns]
            
            if available_columns:
                # 윈도우 크기 설정 (이동 정규화)
                window_size = 20
                
                for col in available_columns:
                    # 이동 최대값, 최소값 계산
                    df_normalized[f'{col}_roll_max'] = df_normalized[col].rolling(window=window_size).max()
                    df_normalized[f'{col}_roll_min'] = df_normalized[col].rolling(window=window_size).min()
                    
                    # 첫 window_size 행은 전체 값으로 계산
                    df_normalized.loc[df_normalized.index[:window_size], f'{col}_roll_max'] = df_normalized[col][:window_size].max()
                    df_normalized.loc[df_normalized.index[:window_size], f'{col}_roll_min'] = df_normalized[col][:window_size].min()
                    
                    # Min-Max 정규화 적용
                    df_normalized[f'{col}_norm'] = (df_normalized[col] - df_normalized[f'{col}_roll_min']) / \
                                                 (df_normalized[f'{col}_roll_max'] - df_normalized[f'{col}_roll_min'])
                    
                    # NaN, inf 처리
                    df_normalized[f'{col}_norm'] = df_normalized[f'{col}_norm'].replace([np.inf, -np.inf], np.nan)
                    df_normalized[f'{col}_norm'] = df_normalized[f'{col}_norm'].fillna(0.5)  # 중간값으로 대체
                    
                    # 임시 컬럼 삭제
                    df_normalized = df_normalized.drop([f'{col}_roll_max', f'{col}_roll_min'], axis=1)
            
            # 거래량 정규화 (로그 스케일링)
            if 'volume' in df_normalized.columns:
                self._normalize_volume(df_normalized)
            
            # 기술적 지표 정규화
            self._normalize_technical_indicators(df_normalized)
            
            # 실시간 API 데이터 지표 처리
            self._normalize_api_indicators(df_normalized)
            
            # NaN 값 최종 처리
            df_normalized = df_normalized.replace([np.inf, -np.inf], np.nan)
            df_normalized = df_normalized.fillna(method='ffill')
            df_normalized = df_normalized.fillna(method='bfill')
            df_normalized = df_normalized.fillna(0.5)  # 중간값으로 최종 대체
            
            logger.info(f"데이터 정규화 완료")
            return df_normalized
            
        except Exception as e:
            logger.error(f"데이터 정규화 실패: {e}", exc_info=True)
            return df
    
    def _normalize_volume(self, df: pd.DataFrame) -> None:
        """
        거래량 정규화
        
        Args:
            df (pd.DataFrame): 정규화할 데이터프레임 (inplace)
        """
        try:
            # 윈도우 크기 설정
            window_size = 20
            
            # 로그 스케일링
            df['volume_norm'] = np.log1p(df['volume'])
            
            # 이동 윈도우에서 최대, 최소 계산
            df['volume_roll_max'] = df['volume_norm'].rolling(window=window_size).max()
            df['volume_roll_min'] = df['volume_norm'].rolling(window=window_size).min()
            
            # 첫 window_size 행은 전체 값으로 계산
            df.loc[df.index[:window_size], 'volume_roll_max'] = df['volume_norm'][:window_size].max()
            df.loc[df.index[:window_size], 'volume_roll_min'] = df['volume_norm'][:window_size].min()
            
            # Min-Max 정규화 적용
            df['volume_norm'] = (df['volume_norm'] - df['volume_roll_min']) / \
                              (df['volume_roll_max'] - df['volume_roll_min'])
            
            # NaN, inf 처리
            df['volume_norm'] = df['volume_norm'].replace([np.inf, -np.inf], np.nan)
            df['volume_norm'] = df['volume_norm'].fillna(0.5)
            
            # 임시 컬럼 삭제
            df.drop(['volume_roll_max', 'volume_roll_min'], axis=1, inplace=True)
            
        except Exception as e:
            logger.error(f"거래량 정규화 실패: {e}", exc_info=True)
    
    def _normalize_technical_indicators(self, df: pd.DataFrame) -> None:
        """
        기술적 지표 정규화
        
        Args:
            df (pd.DataFrame): 정규화할 데이터프레임 (inplace)
        """
        try:
            # 윈도우 크기 설정
            window_size = 20
            
            # 정규화할 기술적 지표 목록
            tech_columns = [
                'sma5', 'sma10', 'sma20', 'sma60', 'sma120',
                'ema5', 'ema10', 'ema20',
                'bb_upper', 'bb_middle', 'bb_lower',
                'rsi', 'adx', 'macd', 'macd_signal', 'macd_hist',
                'volatility', 'volume_ma5', 'volume_ma20', 'volume_ratio', 'obv',
                'atr', 'price_momentum', 'ma_cross', 'bb_width'
            ]
            
            # 실제 존재하는 컬럼만 필터링
            available_tech_columns = [col for col in tech_columns if col in df.columns]
            
            for col in available_tech_columns:
                # 특수 케이스 처리
                if col == 'rsi':
                    # RSI는 이미 0-100 범위라서 /100으로 정규화
                    df[f'{col}_norm'] = df[col] / 100
                elif col in ['macd', 'macd_signal', 'macd_hist', 'obv', 'ma_cross']:
                    # 이동 윈도우에서 최대, 최소 계산
                    df[f'{col}_roll_max'] = df[col].rolling(window=window_size).max()
                    df[f'{col}_roll_min'] = df[col].rolling(window=window_size).min()
                    
                    # 첫 window_size 행은 전체 값으로 계산
                    df.loc[df.index[:window_size], f'{col}_roll_max'] = df[col][:window_size].max()
                    df.loc[df.index[:window_size], f'{col}_roll_min'] = df[col][:window_size].min()
                    
                    # Min-Max 정규화 적용 (0으로 나누는 것 방지)
                    range_value = df[f'{col}_roll_max'] - df[f'{col}_roll_min']
                    where_zero = range_value == 0
                    range_value[where_zero] = 1  # 0인 경우 1로 대체
                    
                    df[f'{col}_norm'] = (df[col] - df[f'{col}_roll_min']) / range_value
                    
                    # 임시 컬럼 삭제
                    df.drop([f'{col}_roll_max', f'{col}_roll_min'], axis=1, inplace=True)
                else:
                    # 나머지는 가격 관련 항목이므로 가격 정규화 기준 사용
                    if 'close' in df.columns:
                        df[f'{col}_norm'] = (df[col] - df['close']) / df['close'] + 0.5
                    else:
                        # close가 없는 경우 자체 Min-Max 정규화
                        df[f'{col}_roll_max'] = df[col].rolling(window=window_size).max()
                        df[f'{col}_roll_min'] = df[col].rolling(window=window_size).min()
                        
                        df.loc[df.index[:window_size], f'{col}_roll_max'] = df[col][:window_size].max()
                        df.loc[df.index[:window_size], f'{col}_roll_min'] = df[col][:window_size].min()
                        
                        # Min-Max 정규화 적용 (0으로 나누는 것 방지)
                        range_value = df[f'{col}_roll_max'] - df[f'{col}_roll_min']
                        where_zero = range_value == 0
                        range_value[where_zero] = 1  # 0인 경우 1로 대체
                        
                        df[f'{col}_norm'] = (df[col] - df[f'{col}_roll_min']) / range_value
                        
                        # 임시 컬럼 삭제
                        df.drop([f'{col}_roll_max', f'{col}_roll_min'], axis=1, inplace=True)
            
            # 수익률 관련 컬럼 처리
            if 'daily_return' in df.columns:
                # 이동 윈도우에서 최대, 최소 계산 (±10% 범위로 클리핑)
                df['daily_return_clipped'] = df['daily_return'].clip(-0.1, 0.1)
                df['daily_return_norm'] = (df['daily_return_clipped'] + 0.1) / 0.2
                df.drop(['daily_return_clipped'], axis=1, inplace=True)
                
        except Exception as e:
            logger.error(f"기술적 지표 정규화 실패: {e}", exc_info=True)
    
    def _normalize_api_indicators(self, df: pd.DataFrame) -> None:
        """
        실시간 API 데이터 지표 정규화
        
        Args:
            df (pd.DataFrame): 정규화할 데이터프레임 (inplace)
        """
        try:
            # bid_ask_ratio 정규화
            if 'bid_ask_ratio' in df.columns:
                # 범위가 0~무한대이므로 로그 스케일 후 0~1 범위로 변환
                # 일반적으로 0.5~2.0 사이가 정상적인 범위라고 가정
                df['bid_ask_ratio_norm'] = df['bid_ask_ratio'].clip(0.1, 10)
                df['bid_ask_ratio_norm'] = (np.log10(df['bid_ask_ratio_norm']) + 1) / 2
            
            # market_pressure 정규화
            if 'market_pressure' in df.columns:
                # 이미 0~1 범위로 정규화되어 있음
                df['market_pressure_norm'] = df['market_pressure']
                
        except Exception as e:
            logger.error(f"API 지표 정규화 실패: {e}", exc_info=True)
    
    def create_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        정규화된 특성만 선택하여 반환
        
        Args:
            df (pd.DataFrame): 정규화된 데이터프레임
            
        Returns:
            pd.DataFrame: 정규화된 특성 컬럼만 선택된 데이터프레임
        """
        # 원본 데이터 복사
        df_features = df.copy()
        
        try:
            # 원본 데이터에서 date, code와 같은 비수치 데이터 유지
            metadata_cols = ['date', 'code', 'stock_name', 'market']
            available_metadata = [col for col in metadata_cols if col in df_features.columns]
            
            # 정규화된 특성 컬럼 선택
            norm_cols = [col for col in df_features.columns if col.endswith('_norm')]
            
            # 필요한 컬럼만 선택
            selected_cols = available_metadata + norm_cols
            
            # 결과 반환
            df_selected = df_features[selected_cols].copy()
            logger.info(f"정규화된 특성 {len(norm_cols)}개 선택 완료")
            
            return df_selected
            
        except Exception as e:
            logger.error(f"정규화된 특성 선택 실패: {e}", exc_info=True)
            return df 