"""
RL 시스템 설정 관리
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """모델 설정"""
    model_type: str = "ppo"
    timesteps: int = 100000
    learning_rate: float = 0.0003
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2


@dataclass
class EnvironmentConfig:
    """환경 설정"""
    initial_balance: float = 10000000.0
    commission: float = 0.00015
    window_size: int = 20
    reward_scaling: float = 0.01
    holding_penalty: float = 0.0001


@dataclass
class TrainingConfig:
    """훈련 설정"""
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    min_episodes: int = 10
    max_episodes: int = 1000
    early_stopping_patience: int = 50
    save_interval: int = 10


@dataclass
class DataConfig:
    """데이터 설정"""
    korea_stock_enabled: bool = True
    us_stock_enabled: bool = True
    technical_indicators: list = None
    normalization_method: str = "minmax"
    sequence_length: int = 60


@dataclass
class RLConfig:
    """통합 RL 시스템 설정"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.environment = EnvironmentConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        
        # 기본 기술적 지표 설정
        if self.data.technical_indicators is None:
            self.data.technical_indicators = [
                'sma_5', 'sma_10', 'sma_20', 'sma_60',
                'ema_12', 'ema_26',
                'rsi_14', 'rsi_30',
                'macd', 'macd_signal', 'macd_histogram',
                'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                'stoch_k', 'stoch_d',
                'williams_r',
                'atr_14',
                'volume_sma_20'
            ]
        
        # 경로 설정
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "models", "saved_models")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.data_dir = os.path.join(self.base_dir, "data", "cache")
        
        # 디렉터리 생성
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델 설정 반환"""
        return {
            'model_type': self.model.model_type,
            'timesteps': self.model.timesteps,
            'learning_rate': self.model.learning_rate,
            'batch_size': self.model.batch_size,
            'n_epochs': self.model.n_epochs,
            'gamma': self.model.gamma,
            'gae_lambda': self.model.gae_lambda,
            'clip_range': self.model.clip_range
        }
    
    def get_environment_config(self) -> Dict[str, Any]:
        """환경 설정 반환"""
        return {
            'initial_balance': self.environment.initial_balance,
            'commission': self.environment.commission,
            'window_size': self.environment.window_size,
            'reward_scaling': self.environment.reward_scaling,
            'holding_penalty': self.environment.holding_penalty
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """훈련 설정 반환"""
        return {
            'train_ratio': self.training.train_ratio,
            'validation_ratio': self.training.validation_ratio,
            'test_ratio': self.training.test_ratio,
            'min_episodes': self.training.min_episodes,
            'max_episodes': self.training.max_episodes,
            'early_stopping_patience': self.training.early_stopping_patience,
            'save_interval': self.training.save_interval
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """데이터 설정 반환"""
        return {
            'korea_stock_enabled': self.data.korea_stock_enabled,
            'us_stock_enabled': self.data.us_stock_enabled,
            'technical_indicators': self.data.technical_indicators,
            'normalization_method': self.data.normalization_method,
            'sequence_length': self.data.sequence_length
        }
    
    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """설정 업데이트"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    section = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(section, sub_key):
                            setattr(section, sub_key, sub_value)
                else:
                    setattr(self, key, value)