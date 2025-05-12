"""
한국 주식 자동매매 - 강화학습 데이터 관리 모듈
다른 데이터 처리 모듈들을 통합하여 강화학습을 위한 데이터 파이프라인 관리
"""

import os
import numpy as np
import pandas as pd
import logging
import json
import datetime
from typing import Optional, Dict, List, Any, Union, Tuple

from .data_preprocessor import DataPreprocessor
from .technical_indicators import TechnicalIndicatorGenerator
from .data_normalizer import DataNormalizer
from .sequence_generator import SequenceGenerator
from .market_data_integrator import MarketDataIntegrator

logger = logging.getLogger("stock_auto")

class RLDataManager:
    """강화학습 데이터 파이프라인 관리 클래스"""
    
    def __init__(self, cache_dir=None, lookback=20, feature_columns=None):
        """
        강화학습 데이터 관리자 초기화
        
        Args:
            cache_dir (str): 캐시 디렉토리 경로
            lookback (int): 학습에 사용할 과거 데이터 길이
            feature_columns (list): 사용할 특성 컬럼 리스트
        """
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '../../../data/cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 하위 모듈 초기화
        self.preprocessor = DataPreprocessor(cache_dir=self.cache_dir)
        self.tech_indicator = TechnicalIndicatorGenerator()
        self.normalizer = DataNormalizer()
        self.sequence_generator = SequenceGenerator(lookback=lookback, feature_columns=feature_columns)
        self.market_integrator = MarketDataIntegrator(cache_dir=self.cache_dir)
    
    def prepare_api_data_for_rl(self, code: str, lookback_days: int = 30) -> pd.DataFrame:
        """
        API에서 가져온 데이터를 강화학습에 사용할 수 있도록 준비
        
        Args:
            code (str): 종목 코드
            lookback_days (int): 과거 데이터 확인 일수
            
        Returns:
            pd.DataFrame: 강화학습용으로 처리된 데이터프레임
        """
        try:
            # 1. 주가 데이터 로드
            df = self.preprocessor.load_stock_data(code, days=lookback_days)
            
            if df.empty:
                logger.warning(f"{code} 주가 데이터가 없습니다.")
                return pd.DataFrame()
            
            # 2. 실시간 데이터 로드 및 통합
            realtime_data = self.preprocessor.load_realtime_data(code)
            if realtime_data:
                df = self.preprocessor.merge_with_realtime(df, realtime_data, code)
            
            # 3. 데이터 정제
            df = self.preprocessor.clean_data(df)
            
            # 4. 기술적 지표 추가
            df = self.tech_indicator.add_all_indicators(df)
            
            # 5. 데이터 정규화
            df = self.normalizer.normalize_data(df)
            
            # 6. 시장 데이터 통합
            df = self.market_integrator.combine_market_data(df, code)
            
            # 7. 강화학습용 데이터 저장
            self.preprocessor.save_processed_data(df, code, suffix='rl_processed')
            
            logger.info(f"{code} 주가 데이터 강화학습용 전처리 완료: {len(df)}개 행, {len(df.columns)}개 컬럼")
            return df
            
        except Exception as e:
            logger.error(f"강화학습용 데이터 준비 실패: {e}", exc_info=True)
            return pd.DataFrame()
    
    def create_training_dataset(self, code: str) -> Dict[str, np.ndarray]:
        """
        강화학습 모델 훈련을 위한 데이터셋 생성
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict: 강화학습 훈련 데이터셋
        """
        try:
            # 1. 처리된 데이터 로드 또는 새로 생성
            rl_data_path = os.path.join(self.cache_dir, f'{code}_rl_processed.csv')
            
            if os.path.exists(rl_data_path):
                df = pd.read_csv(rl_data_path)
                logger.info(f"{code} 기존 처리 데이터 로드 완료: {len(df)}개 행")
            else:
                df = self.prepare_api_data_for_rl(code)
            
            if df.empty:
                logger.warning(f"{code} 처리된 데이터가 없습니다.")
                return {}
            
            # 2. 정규화된 특성만 선택
            df_features = self.normalizer.create_normalized_features(df)
            
            # 3. 강화학습 데이터셋 생성
            rl_dataset = self.sequence_generator.create_rl_dataset(df_features)
            
            if not rl_dataset:
                logger.warning(f"{code} 강화학습 데이터셋 생성 실패")
                return {}
            
            # 4. 데이터셋 정보 추가
            rl_dataset['info'] = {
                'code': code,
                'data_length': len(df),
                'features': len(df_features.columns) - 1 if 'date' in df_features.columns else len(df_features.columns),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 5. 데이터셋 저장
            try:
                np.savez(
                    os.path.join(self.cache_dir, f'{code}_rl_dataset.npz'),
                    states=rl_dataset['states'],
                    rewards=rl_dataset['rewards'],
                    next_states=rl_dataset['next_states'],
                    done=rl_dataset['done']
                )
                
                # 정보 파일 저장
                with open(os.path.join(self.cache_dir, f'{code}_rl_dataset_info.json'), 'w') as f:
                    json.dump(rl_dataset['info'], f, indent=2)
                
                logger.info(f"{code} 강화학습 데이터셋 저장 완료")
            except Exception as e:
                logger.error(f"데이터셋 저장 실패: {e}", exc_info=True)
            
            return rl_dataset
            
        except Exception as e:
            logger.error(f"훈련 데이터셋 생성 실패: {e}", exc_info=True)
            return {}
    
    def get_current_state(self, code: str) -> np.ndarray:
        """
        현재 상태 벡터 생성
        
        Args:
            code (str): 종목 코드
            
        Returns:
            numpy.ndarray: 현재 상태 벡터
        """
        try:
            # 1. 실시간 데이터 로드
            realtime_data = self.preprocessor.load_realtime_data(code)
            
            if not realtime_data:
                logger.warning(f"{code} 실시간 데이터가 없습니다.")
                return np.array([])
            
            # 2. 과거 처리된 데이터 로드
            rl_data_path = os.path.join(self.cache_dir, f'{code}_rl_processed.csv')
            
            historical_data = None
            if os.path.exists(rl_data_path):
                historical_data = pd.read_csv(rl_data_path)
                if 'date' in historical_data.columns:
                    historical_data['date'] = pd.to_datetime(historical_data['date'])
                    historical_data = historical_data.sort_values('date')
            
            # 3. 현재 상태 벡터 생성
            state_vector = self.sequence_generator.prepare_state_vector(realtime_data, historical_data)
            
            return state_vector
            
        except Exception as e:
            logger.error(f"현재 상태 벡터 생성 실패: {e}", exc_info=True)
            return np.array([])
    
    def update_datasets(self, codes: List[str]) -> bool:
        """
        여러 종목의 데이터셋 일괄 업데이트
        
        Args:
            codes (list): 종목 코드 리스트
            
        Returns:
            bool: 성공 여부
        """
        try:
            for code in codes:
                logger.info(f"{code} 강화학습 데이터셋 업데이트 시작")
                self.create_training_dataset(code)
            
            logger.info(f"{len(codes)}개 종목 데이터셋 업데이트 완료")
            return True
            
        except Exception as e:
            logger.error(f"데이터셋 일괄 업데이트 실패: {e}", exc_info=True)
            return False 