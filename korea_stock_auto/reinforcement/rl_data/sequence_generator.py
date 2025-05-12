"""
한국 주식 자동매매 - 시퀀스 생성 모듈
강화학습 모델을 위한 시계열 시퀀스 생성 기능
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, List, Any, Union, Tuple

logger = logging.getLogger("stock_auto")

class SequenceGenerator:
    """강화학습용 시퀀스 생성 클래스"""
    
    def __init__(self, lookback=20, feature_columns=None):
        """
        시퀀스 생성기 초기화
        
        Args:
            lookback (int): 학습에 사용할 과거 데이터 길이
            feature_columns (list): 사용할 특성 컬럼 리스트
        """
        self.lookback = lookback
        
        # 기본 특성 컬럼 정의
        self.feature_columns = feature_columns or [
            'close_norm', 'open_norm', 'high_norm', 'low_norm', 'volume_norm',
            'rsi_norm', 'macd_norm', 'macd_signal_norm', 
            'bb_upper_norm', 'bb_middle_norm', 'bb_lower_norm',
            'sma5_norm', 'sma20_norm', 'sma60_norm'
        ]
    
    def create_sequences(self, df: pd.DataFrame, target_column='close') -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 시퀀스 데이터 생성
        
        Args:
            df (pd.DataFrame): 처리된 데이터프레임
            target_column (str): 타겟 컬럼 이름
            
        Returns:
            tuple: (X, y) 시퀀스 데이터와 타겟값
        """
        try:
            # 사용 가능한 특성 컬럼 필터링
            available_features = [col for col in self.feature_columns if col in df.columns]
            
            if not available_features:
                logger.warning("사용 가능한 특성 컬럼이 없습니다.")
                return np.array([]), np.array([])
            
            # 특성 컬럼만 선택
            feature_data = df[available_features].values
            
            # 시퀀스 및 타겟 데이터 초기화
            X, y = [], []
            
            # 시퀀스 생성
            for i in range(len(df) - self.lookback):
                # 시퀀스 데이터
                seq = feature_data[i:i+self.lookback]
                X.append(seq)
                
                # 타겟 데이터 (다음 종가의 방향)
                current_close = df[target_column].values[i+self.lookback-1]
                next_close = df[target_column].values[i+self.lookback]
                
                if next_close > current_close:
                    target = 1  # 상승 (매수)
                elif next_close < current_close:
                    target = 2  # 하락 (매도)
                else:
                    target = 0  # 유지
                
                y.append(target)
            
            logger.info(f"시퀀스 생성 완료: {len(X)}개 시퀀스, {len(available_features)}개 특성")
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"시퀀스 생성 실패: {e}", exc_info=True)
            return np.array([]), np.array([])
    
    def create_rl_dataset(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        강화학습용 데이터셋 생성
        
        Args:
            df (pd.DataFrame): 정규화된 데이터프레임
            
        Returns:
            dict: 강화학습용 데이터셋
        """
        try:
            # 사용 가능한 특성 컬럼 필터링
            available_features = [col for col in self.feature_columns if col in df.columns]
            
            if not available_features:
                logger.warning("사용 가능한 특성 컬럼이 없습니다.")
                return {}
            
            # 특성 컬럼만 선택
            feature_data = df[available_features].values
            
            # 시퀀스 생성
            states = []
            rewards = []
            next_states = []
            done_flags = []
            
            for i in range(len(df) - self.lookback - 1):
                # 현재 상태
                current_state = feature_data[i:i+self.lookback]
                states.append(current_state)
                
                # 다음 상태
                next_state = feature_data[i+1:i+self.lookback+1]
                next_states.append(next_state)
                
                # 보상 (다음 종가의 방향에 따른 보상)
                if 'close' in df.columns:
                    current_close = df['close'].values[i+self.lookback-1]
                    next_close = df['close'].values[i+self.lookback]
                    reward = (next_close - current_close) / current_close  # 수익률 기반 보상
                else:
                    # close가 없는 경우 기본값
                    reward = 0
                
                rewards.append(reward)
                
                # 완료 플래그 (마지막 시퀀스인 경우만 True)
                done = (i == len(df) - self.lookback - 2)
                done_flags.append(done)
            
            # 결과 데이터 구성
            rl_dataset = {
                'states': np.array(states),
                'rewards': np.array(rewards),
                'next_states': np.array(next_states),
                'done': np.array(done_flags)
            }
            
            logger.info(f"강화학습 데이터셋 생성 완료: {len(states)}개 샘플")
            return rl_dataset
            
        except Exception as e:
            logger.error(f"강화학습 데이터셋 생성 실패: {e}", exc_info=True)
            return {}
    
    def prepare_state_vector(self, current_data: Dict[str, Any], historical_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        현재 상태 벡터 생성
        
        Args:
            current_data (dict): 현재 시장 데이터
            historical_data (pd.DataFrame): 과거 시장 데이터
            
        Returns:
            numpy.ndarray: 현재 상태 벡터
        """
        try:
            # 과거 데이터가 있는 경우
            if historical_data is not None and not historical_data.empty:
                # 최신 데이터 사용
                if len(historical_data) >= self.lookback:
                    # 사용 가능한 특성 컬럼 필터링
                    available_features = [col for col in self.feature_columns if col in historical_data.columns]
                    
                    if available_features:
                        # 최근 lookback 일치 데이터 추출
                        recent_data = historical_data.iloc[-self.lookback:][available_features].values
                        return recent_data.flatten()  # 1차원 벡터로 변환
            
            # 과거 데이터가 없는 경우 현재 데이터만 사용하여 간소화된 상태 벡터 생성
            state_vector = []
            
            # 현재 가격
            current_price = float(current_data.get('close', 0))
            
            # 가격 정보 (정규화 없이 기본 비율만 계산)
            ma5 = float(current_data.get('ma5', current_price))
            ma20 = float(current_data.get('ma20', current_price))
            
            # 이평선 대비 가격 비율
            price_ma5_ratio = (current_price - ma5) / ma5 if ma5 > 0 else 0
            price_ma20_ratio = (current_price - ma20) / ma20 if ma20 > 0 else 0
            
            # 기본 상태 벡터
            state_vector = [
                price_ma5_ratio,
                price_ma20_ratio,
                float(current_data.get('rsi', 50)) / 100,  # RSI 0-1 정규화
                float(current_data.get('volume_ratio', 1))
            ]
            
            return np.array(state_vector)
            
        except Exception as e:
            logger.error(f"상태 벡터 생성 실패: {e}", exc_info=True)
            # 기본 상태 벡터 반환
            return np.array([0.0, 0.0, 0.5, 1.0]) 