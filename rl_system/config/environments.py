"""
환경별 설정 관리
"""

import os
from typing import Dict, Any, Optional


class EnvironmentConfig:
    """환경별 설정 관리 클래스"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """환경별 설정 로드"""
        configs = {
            "development": {
                "model": {
                    "timesteps": 50000,
                    "learning_rate": 0.0003,
                    "batch_size": 32,
                    "save_interval": 5
                },
                "environment": {
                    "initial_balance": 1000000.0,
                    "commission": 0.00015,
                    "reward_scaling": 0.01
                },
                "training": {
                    "max_episodes": 100,
                    "early_stopping_patience": 20
                },
                "data": {
                    "sequence_length": 30
                }
            },
            "production": {
                "model": {
                    "timesteps": 200000,
                    "learning_rate": 0.0001,
                    "batch_size": 64,
                    "save_interval": 10
                },
                "environment": {
                    "initial_balance": 10000000.0,
                    "commission": 0.00015,
                    "reward_scaling": 0.01
                },
                "training": {
                    "max_episodes": 1000,
                    "early_stopping_patience": 50
                },
                "data": {
                    "sequence_length": 60
                }
            },
            "testing": {
                "model": {
                    "timesteps": 1000,
                    "learning_rate": 0.001,
                    "batch_size": 16,
                    "save_interval": 1
                },
                "environment": {
                    "initial_balance": 100000.0,
                    "commission": 0.0,
                    "reward_scaling": 0.1
                },
                "training": {
                    "max_episodes": 10,
                    "early_stopping_patience": 5
                },
                "data": {
                    "sequence_length": 10
                }
            }
        }
        
        return configs.get(self.environment, configs["development"])
    
    def get_config(self) -> Dict[str, Any]:
        """설정 반환"""
        return self._config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """특정 섹션 설정 반환"""
        return self._config.get(section, {})
    
    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """특정 값 반환"""
        return self._config.get(section, {}).get(key, default)
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """설정 업데이트"""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    @classmethod
    def get_environment_from_env(cls) -> str:
        """환경 변수에서 환경 설정 가져오기"""
        return os.getenv("RL_ENVIRONMENT", "development")
    
    @classmethod
    def create_from_env(cls) -> 'EnvironmentConfig':
        """환경 변수 기반 설정 생성"""
        environment = cls.get_environment_from_env()
        return cls(environment)