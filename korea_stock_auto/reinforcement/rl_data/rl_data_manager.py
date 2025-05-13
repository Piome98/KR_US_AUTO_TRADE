"""
한국 주식 자동매매 - 강화학습 데이터 관리 모듈 (리팩토링 버전)

다른 데이터 처리 모듈들을 통합하여 강화학습을 위한 데이터 파이프라인 관리
data/database 모듈을 활용하여 데이터 접근
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

from korea_stock_auto.data.database import (
    PriceDataManager, 
    TechnicalDataManager, 
    MarketDataManager
)

logger = logging.getLogger("stock_auto")

class RLDataManager:
    """강화학습 데이터 파이프라인 관리 클래스"""
    
    def __init__(self, db_path: str = "stock_data.db", cache_dir=None, lookback=20, feature_columns=None):
        """
        강화학습 데이터 관리자 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
            cache_dir: 캐시 디렉토리 경로
            lookback: 학습에 사용할 과거 데이터 길이
            feature_columns: 사용할 특성 컬럼 리스트
        """
        self.db_path = db_path
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '../../../data/cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # data/database 모듈 초기화
        self.price_manager = PriceDataManager(db_path)
        self.tech_manager = TechnicalDataManager(db_path)
        self.market_manager = MarketDataManager(db_path)
        
        # 강화학습 특화 모듈 초기화
        self.preprocessor = DataPreprocessor(cache_dir=self.cache_dir)
        self.tech_indicator = TechnicalIndicatorGenerator(db_path=db_path)
        self.normalizer = DataNormalizer()
        self.sequence_generator = SequenceGenerator(lookback=lookback, feature_columns=feature_columns)
        
        # 시장 데이터 통합 모듈 - 데이터베이스 매니저 전달
        self.market_integrator = MarketDataIntegrator(
            cache_dir=self.cache_dir,
            price_manager=self.price_manager,
            market_manager=self.market_manager
        )
    
    def prepare_api_data_for_rl(self, code: str, lookback_days: int = 30) -> pd.DataFrame:
        """
        API에서 가져온 데이터를 강화학습에 사용할 수 있도록 준비
        
        Args:
            code: 종목 코드
            lookback_days: 과거 데이터 확인 일수
            
        Returns:
            pd.DataFrame: 강화학습용으로 처리된 데이터프레임
        """
        try:
            # 1. 데이터베이스에서 주가 데이터 로드
            df = self.price_manager.get_price_history(code, days=lookback_days)
            
            if df.empty:
                logger.warning(f"{code} 주가 데이터가 없습니다.")
                return pd.DataFrame()
            
            # 2. 실시간 데이터 로드 및 통합 (기존 코드 유지)
            realtime_data = self.preprocessor.load_realtime_data(code)
            if realtime_data:
                df = self.preprocessor.merge_with_realtime(df, realtime_data, code)
            
            # 3. 데이터 정제
            df = self.preprocessor.clean_data(df)
            
            # 4. 기술적 지표 추가 (데이터베이스 우선 활용)
            df = self.tech_indicator.add_all_indicators(df, use_db_first=True, code=code)
            
            # 5. 데이터 정규화
            df = self.normalizer.normalize_data(df)
            
            # 6. 시장 데이터 통합 (통합된 인터페이스 사용)
            df = self.market_integrator.combine_market_data(df, code)
            
            # 7. 강화학습용 데이터 저장
            self.preprocessor.save_processed_data(df, code, suffix='rl_processed')
            
            logger.info(f"{code} 주가 데이터 강화학습용 전처리 완료: {len(df)}개 행, {len(df.columns)}개 컬럼")
            return df
            
        except Exception as e:
            logger.error(f"강화학습용 데이터 준비 실패: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _calculate_rolling_correlation(self, series1: pd.Series, series2: pd.Series, window: int = 20) -> pd.Series:
        """
        두 시계열 간의 이동 상관계수 계산
        
        Args:
            series1: 첫 번째 시계열
            series2: 두 번째 시계열
            window: 윈도우 크기
            
        Returns:
            pd.Series: 상관계수 시리즈
        """
        return series1.rolling(window=window).corr(series2)
    
    def create_training_dataset(self, code: str) -> Dict[str, np.ndarray]:
        """
        강화학습 모델 훈련을 위한 데이터셋 생성
        
        Args:
            code: 종목 코드
            
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
                
                with open(os.path.join(self.cache_dir, f'{code}_rl_dataset_info.json'), 'w') as f:
                    json.dump(rl_dataset['info'], f, indent=2)
                
                logger.info(f"{code} 강화학습 데이터셋 저장 완료")
                
            except Exception as e:
                logger.error(f"데이터셋 저장 실패: {e}", exc_info=True)
            
            return rl_dataset
            
        except Exception as e:
            logger.error(f"강화학습 데이터셋 생성 실패: {e}", exc_info=True)
            return {}
    
    def get_current_state(self, code: str) -> np.ndarray:
        """
        현재 거래 상태 가져오기 (최신 데이터)
        
        Args:
            code: 종목 코드
            
        Returns:
            np.ndarray: 강화학습 환경 상태 (feature vector)
        """
        try:
            # 처리된 데이터 가져오기
            df = self.prepare_api_data_for_rl(code, lookback_days=self.sequence_generator.lookback + 5)
            
            if df.empty:
                logger.warning(f"{code} 현재 상태 데이터 없음")
                return np.array([])
                
            # 정규화된 특성 생성
            df_features = self.normalizer.create_normalized_features(df)
            
            # 가장 최근 상태 반환
            state = self.sequence_generator.create_state(df_features)
            
            if state is None or len(state) == 0:
                logger.warning(f"{code} 현재 상태 생성 실패")
                return np.array([])
                
            return state
            
        except Exception as e:
            logger.error(f"현재 상태 가져오기 실패: {e}", exc_info=True)
            return np.array([])
    
    def update_datasets(self, codes: List[str]) -> bool:
        """
        여러 종목의 데이터셋 업데이트
        
        Args:
            codes: 종목 코드 리스트
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            results = []
            
            for code in codes:
                logger.info(f"{code} 데이터셋 업데이트 시작")
                
                # 데이터셋 생성
                dataset = self.create_training_dataset(code)
                
                # 성공 여부 저장
                results.append(bool(dataset))
                
                # 로그 메시지
                if dataset:
                    logger.info(f"{code} 데이터셋 업데이트 완료: {dataset['info']['data_length']}개 데이터, {dataset['info']['features']}개 특성")
                else:
                    logger.warning(f"{code} 데이터셋 업데이트 실패")
            
            # 모든 코드에 대해 성공했는지 확인
            success_rate = sum(results) / len(results) if results else 0
            
            logger.info(f"데이터셋 업데이트 완료: {len(codes)}개 종목 중 {sum(results)}개 성공 ({success_rate:.1%})")
            
            return all(results)
            
        except Exception as e:
            logger.error(f"데이터셋 업데이트 실패: {e}", exc_info=True)
            return False 