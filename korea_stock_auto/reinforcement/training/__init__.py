"""
강화학습 모델 학습 패키지

강화학습 모델 학습, 평가, 파이프라인 관련 모듈 모음
"""

from .base_trainer import BaseTrainer
from .callbacks import SaveOnBestTrainingRewardCallback, EarlyStoppingCallback
from .evaluation import ModelEvaluator
from .pipeline import TrainingPipeline

# 편의를 위한 간소화된 인터페이스
def train_model(code, model_type='ppo', timesteps=100000, start_date=None, end_date=None, feature_set='technical'):
    """
    주식 종목에 대한 강화학습 모델 학습 간소화 함수
    
    Args:
        code (str): 종목 코드
        model_type (str): 모델 유형 ('ppo', 'a2c', 'dqn')
        timesteps (int): 학습 스텝 수
        start_date (str): 시작 날짜
        end_date (str): 종료 날짜
        feature_set (str): 사용할 특성 세트
        
    Returns:
        tuple: (모델 ID, 평가 결과)
    """
    pipeline = TrainingPipeline()
    return pipeline.run_training_pipeline(
        code=code,
        model_type=model_type,
        timesteps=timesteps,
        start_date=start_date,
        end_date=end_date,
        feature_set=feature_set
    )

__all__ = [
    'BaseTrainer',
    'SaveOnBestTrainingRewardCallback',
    'EarlyStoppingCallback',
    'ModelEvaluator',
    'TrainingPipeline',
    'train_model'
] 