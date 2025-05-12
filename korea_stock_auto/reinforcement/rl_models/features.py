"""
강화학습에 사용할 특성 관련 모듈
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger("stock_auto")

# 모든 사용 가능한 특성 목록
AVAILABLE_FEATURES = [
    # 가격 기반 정규화 특성
    'open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm',
    
    # 이동평균선 정규화 특성
    'sma5_norm', 'sma10_norm', 'sma20_norm', 'sma60_norm', 'sma120_norm',
    'ema5_norm', 'ema10_norm', 'ema20_norm',
    
    # 기술적 지표 정규화 특성
    'rsi_norm', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
    'bb_upper_norm', 'bb_middle_norm', 'bb_lower_norm', 
    'adx_norm', 'obv_norm',
    
    # 변동성, 거래량 관련 정규화 특성
    'volatility_norm', 'daily_return_norm',
    'volume_ma5_norm', 'volume_ma20_norm', 'volume_ratio_norm',
    
    # API에서 제공하는 추가 정규화 특성
    'bid_ask_ratio_norm', 'market_pressure_norm',
    
    # 시장 데이터에서 추가된 정규화 특성
    'volume_rank_norm', 'volume_increase_rank_norm', 'volume_ratio_norm',
    
    # 기타 바이너리 특성
    'is_top_volume', 'is_volume_increasing'
]

# 기본 특성 세트 (필수 특성)
BASE_FEATURES = [
    'close_norm', 'volume_norm', 
    'sma5_norm', 'sma20_norm',
    'rsi_norm', 'macd_norm',
    'daily_return_norm'
]

# 기술 분석 특성 세트
TECHNICAL_FEATURES = [
    'sma5_norm', 'sma10_norm', 'sma20_norm', 'sma60_norm',
    'ema5_norm', 'ema10_norm', 'ema20_norm',
    'bb_upper_norm', 'bb_middle_norm', 'bb_lower_norm',
    'rsi_norm', 'adx_norm', 
    'macd_norm', 'macd_signal_norm', 'macd_hist_norm'
]

# 거래량 관련 특성 세트
VOLUME_FEATURES = [
    'volume_norm', 'volume_ma5_norm', 'volume_ma20_norm', 
    'volume_ratio_norm', 'obv_norm',
    'volume_rank_norm', 'volume_increase_rank_norm'
]

# API 관련 특성 세트
API_FEATURES = [
    'bid_ask_ratio_norm', 'market_pressure_norm',
    'is_top_volume', 'is_volume_increasing'
]

# 모델 사전 정의 특성 세트
MODEL_FEATURE_SETS = {
    'minimal': BASE_FEATURES,
    'price_only': ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm'],
    'technical': BASE_FEATURES + TECHNICAL_FEATURES,
    'volume_focus': BASE_FEATURES + VOLUME_FEATURES,
    'api_enhanced': BASE_FEATURES + API_FEATURES,
    'full': AVAILABLE_FEATURES
}


def select_features(
    df: pd.DataFrame, 
    feature_set: str = 'technical', 
    custom_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    강화학습 모델에 사용할 특성 선택
    
    Args:
        df (pd.DataFrame): 전체 데이터프레임
        feature_set (str): 사전 정의된 특성 세트 이름 ('minimal', 'technical', 'volume_focus', 'api_enhanced', 'full')
        custom_features (list): 직접 선택한 특성 목록
        
    Returns:
        pd.DataFrame: 선택된 특성만 포함한 데이터프레임
    """
    try:
        # 원본 컬럼 목록
        original_cols = ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
        available_original_cols = [col for col in original_cols if col in df.columns]
        
        # 선택할 특성 결정
        if custom_features is not None:
            # 사용자 정의 특성 사용
            selected_features = [f for f in custom_features if f in df.columns]
        else:
            # 사전 정의된 특성 세트 사용
            if feature_set in MODEL_FEATURE_SETS:
                preset_features = MODEL_FEATURE_SETS[feature_set]
                selected_features = [f for f in preset_features if f in df.columns]
            else:
                # 기본값으로 technical 특성 사용
                preset_features = MODEL_FEATURE_SETS['technical']
                selected_features = [f for f in preset_features if f in df.columns]
        
        # 선택된 특성이 없는 경우
        if not selected_features:
            logger.warning(f"선택 가능한 특성이 없습니다. 원본 데이터만 반환합니다.")
            return df[available_original_cols]
        
        # 선택한 특성과 원본 컬럼 결합
        final_columns = available_original_cols + selected_features
        
        # 최종 데이터프레임 반환
        logger.info(f"강화학습용 특성 {len(selected_features)}개 선택 완료: {feature_set}")
        return df[final_columns]
        
    except Exception as e:
        logger.error(f"특성 선택 실패: {e}")
        # 오류 발생 시 원본 데이터 반환
        return df


def get_state_dim(features: pd.DataFrame) -> int:
    """
    강화학습 모델의 상태 차원 계산
    
    Args:
        features (pd.DataFrame): 강화학습에 사용할 특성 데이터프레임
        
    Returns:
        int: 상태 차원 크기
    """
    # 날짜, 코드, 원본 가격 데이터 제외
    exclude_cols = ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in features.columns if col not in exclude_cols]
    
    return len(feature_cols)


def get_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """
    모델에서 특성 중요도 추출 (현재는 더미 함수)
    
    Args:
        model: 강화학습 모델
        feature_names (list): 특성 이름 목록
        
    Returns:
        dict: 특성 중요도 사전
    """
    # 실제 구현에서는 모델에서 특성 중요도를 추출하는 로직 필요
    # 현재는 더미 값 반환
    return {name: 1.0/len(feature_names) for name in feature_names} 