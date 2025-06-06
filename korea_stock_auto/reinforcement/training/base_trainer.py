"""
한국 주식 자동매매 - 강화학습 기본 트레이너 모듈

강화학습 모델 훈련을 위한 기본 클래스 및 함수 정의
"""

import os
import json
import pandas as pd
import numpy as np
import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple, Union

from korea_stock_auto.reinforcement.rl_data.rl_data_manager import RLDataManager
from korea_stock_auto.reinforcement.rl_data.technical_indicators import TechnicalIndicatorGenerator
from korea_stock_auto.reinforcement.rl_data.data_normalizer import DataNormalizer
from korea_stock_auto.reinforcement.rl_models import TradingEnvironment, RLModel
from korea_stock_auto.utils import send_message
from korea_stock_auto.data.database import DatabaseManager

logger = logging.getLogger("stock_auto")

class BaseTrainer:
    """강화학습 모델 훈련 기본 클래스"""
    
    def __init__(self, db_manager=None, data_manager=None, output_dir="models"):
        """
        모델 트레이너 초기화
        
        Args:
            db_manager (DatabaseManager): 데이터베이스 관리자
            data_manager (RLDataManager): 데이터 관리자
            output_dir (str): 출력 디렉토리
        """
        self.db_manager = db_manager or DatabaseManager()
        self.data_manager = data_manager or RLDataManager()
        self.tech_indicator = TechnicalIndicatorGenerator()
        self.data_normalizer = DataNormalizer()
        self.output_dir = output_dir
        self.cache_dir = os.path.join(os.path.dirname(__file__), '../../../data/cache')
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'figures'), exist_ok=True)
    
    def _generate_model_id(self, model_type):
        """
        모델 ID 생성
        
        Args:
            model_type (str): 모델 유형
            
        Returns:
            str: 생성된 모델 ID
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{model_type}_{timestamp}"
    
    def load_training_data(self, code, start_date=None, end_date=None):
        """
        학습 데이터 로드
        
        Args:
            code (str): 종목 코드
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        try:
            # 기본 날짜 설정
            if end_date is None:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                # 기본적으로 2년치 데이터 사용
                start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
            
            # 데이터베이스에서 데이터 로드
            query = """
            SELECT date, open, high, low, close, volume
            FROM daily_price
            WHERE code = ? AND date >= ? AND date <= ?
            ORDER BY date
            """
            
            data = self.db_manager.execute_query(query, (code, start_date, end_date))
            
            if data is None or len(data) == 0:
                send_message(f"데이터 로드 실패: 해당 기간에 {code} 데이터가 없습니다.", config.notification.discord_webhook_url)
                logger.warning(f"데이터 로드 실패: 해당 기간에 {code} 데이터가 없습니다.")
                return None
            
            # 데이터프레임 변환
            df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            
            # 숫자 컬럼으로 변환
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            logger.info(f"{code} 데이터 로드 완료: {len(df)}개 데이터")
            return df
            
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}", exc_info=True)
            return None
    
    def prepare_dataset(self, data, feature_set='technical'):
        """
        데이터셋 준비
        
        Args:
            data (pd.DataFrame): 원본 OHLCV 데이터
            feature_set (str): 사용할 특성 세트
            
        Returns:
            tuple: (학습용 데이터, 테스트용 데이터)
        """
        if data is None or len(data) == 0:
            return None, None
        
        try:
            # 기술적 지표 추가
            processed_data = self.tech_indicator.add_all_indicators(data)
            
            # 결측값 제거
            processed_data = processed_data.dropna()
            
            # 데이터 분할 (테스트 데이터는 최근 20%)
            train_size = int(len(processed_data) * 0.8)
            train_data = processed_data.iloc[:train_size]
            test_data = processed_data.iloc[train_size:]
            
            # 데이터 정규화
            train_data_scaled = self.data_normalizer.normalize_data(train_data)
            
            # 학습 데이터의 통계로 테스트 데이터 정규화
            test_data_scaled = self.data_normalizer.normalize_data(
                test_data, 
                scaler=self.data_normalizer.get_fitted_scaler()
            )
            
            logger.info(f"데이터셋 준비 완료: 학습 {len(train_data_scaled)}개, 테스트 {len(test_data_scaled)}개")
            return train_data_scaled, test_data_scaled
            
        except Exception as e:
            logger.error(f"데이터셋 준비 실패: {e}", exc_info=True)
            return None, None
    
    def create_environment(self, data, **env_params):
        """
        트레이딩 환경 생성
        
        Args:
            data (pd.DataFrame): 처리된 데이터
            **env_params: 환경 파라미터
            
        Returns:
            TradingEnvironment: 생성된 환경
        """
        try:
            # 기본 파라미터 설정
            default_params = {
                'initial_balance': 10000000,  # 1천만원
                'commission': 0.00015,        # 수수료 0.015%
                'window_size': 20,            # 20일 관찰 기간
                'reward_scaling': 0.01,       # 보상 스케일
                'holding_penalty': 0.0001     # 홀딩 패널티
            }
            
            # 사용자 정의 파라미터로 업데이트
            params = {**default_params, **env_params}
            
            # 환경 생성
            env = TradingEnvironment(
                df=data,
                initial_balance=params['initial_balance'],
                commission=params['commission'],
                window_size=params['window_size'],
                reward_scaling=params['reward_scaling'],
                holding_penalty=params['holding_penalty']
            )
            
            logger.info(f"거래 환경 생성 완료")
            return env
            
        except Exception as e:
            logger.error(f"환경 생성 실패: {e}", exc_info=True)
            return None
            
    def _split_data(self, data: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        학습용/테스트용 데이터 분할
        
        Args:
            data (pd.DataFrame): 원본 데이터
            test_ratio (float): 테스트 데이터 비율
            
        Returns:
            tuple: (학습용 데이터, 테스트용 데이터)
        """
        if data is None or len(data) == 0:
            return None, None
        
        try:
            # 시간 순서 유지하며 분할
            train_size = int(len(data) * (1 - test_ratio))
            
            # 데이터 분할
            train_data = data.iloc[:train_size].copy()
            test_data = data.iloc[train_size:].copy()
            
            logger.info(f"데이터 분할 완료: 학습 {len(train_data)}개, 테스트 {len(test_data)}개")
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"데이터 분할 실패: {e}", exc_info=True)
            return None, None
            
    def save_model_info(self, model_id, model_type, hyperparams, train_period, results):
        """
        모델 정보 저장
        
        Args:
            model_id (str): 모델 ID
            model_type (str): 모델 유형
            hyperparams (dict): 하이퍼파라미터
            train_period (dict): 학습 기간
            results (dict): 평가 결과
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 모델 메타데이터
            metadata = {
                'model_id': model_id,
                'model_type': model_type,
                'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'hyperparameters': hyperparams,
                'training_period': train_period,
                'evaluation_results': results
            }
            
            # 메타데이터 저장 경로
            metadata_path = os.path.join(self.output_dir, f"{model_id}_metadata.json")
            
            # JSON 파일로 저장
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"모델 정보 저장 완료: {metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"모델 정보 저장 실패: {e}", exc_info=True)
            return False 