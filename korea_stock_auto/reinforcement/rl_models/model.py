"""
한국 주식 자동매매 - 강화학습 모델 모듈

강화학습 모델 래퍼 클래스 제공
"""

import os
import pickle
import numpy as np
import datetime
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import logging

logger = logging.getLogger("stock_auto")

class RLModel:
    """강화학습 모델 래퍼 클래스"""
    
    def __init__(self, model_type: str = "ppo", model_kwargs: Dict[str, Any] = None, model_path: str = None):
        """
        강화학습 모델 초기화
        
        Args:
            model_type (str): 모델 유형 ('ppo', 'a2c', 'dqn')
            model_kwargs (dict): 모델 생성 파라미터
            model_path (str): 모델 저장 경로
        """
        self.model_type = model_type.lower()
        self.model_kwargs = model_kwargs or {}
        self.model_path = model_path
        self.sb3_model = None  # 실제 Stable Baselines3 모델 인스턴스
        
        # 모델 인스턴스가 미리 제공된 경우
        if isinstance(model_kwargs, (PPO, A2C, DQN)):
            self.sb3_model = model_kwargs
            self.model_path = model_path
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, np.ndarray]:
        """
        모델 예측 수행
        
        Args:
            observation (numpy.ndarray): 관찰 벡터
            deterministic (bool): 결정적 액션 여부
            
        Returns:
            tuple: (예측 액션, 추가 정보)
        """
        if self.sb3_model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
            
        action, states = self.sb3_model.predict(observation, deterministic=deterministic)
        return action, states
    
    def train(self, env, total_timesteps: int = 100000, save: bool = True) -> bool:
        """
        모델 학습 수행
        
        Args:
            env: 학습할 환경
            total_timesteps (int): 총 학습 단계 수
            save (bool): 저장 여부
            
        Returns:
            bool: 학습 성공 여부
        """
        try:
            # 환경 래핑
            env = DummyVecEnv([lambda: env])
            
            # 모델 타입에 따라 모델 생성
            if self.model_type == "ppo":
                self.sb3_model = PPO("MlpPolicy", env, verbose=1, **self.model_kwargs)
            elif self.model_type == "a2c":
                self.sb3_model = A2C("MlpPolicy", env, verbose=1, **self.model_kwargs)
            elif self.model_type == "dqn":
                self.sb3_model = DQN("MlpPolicy", env, verbose=1, **self.model_kwargs)
            else:
                raise ValueError(f"지원하지 않는 모델 유형: {self.model_type}")
            
            # 모델 학습
            self.sb3_model.learn(total_timesteps=total_timesteps)
            
            # 학습된 모델 저장
            if save and self.model_path:
                self.save(self.model_path)
                
            return True
            
        except Exception as e:
            logger.error(f"모델 학습 실패: {e}")
            return False
    
    def save(self, path: str = None) -> None:
        """
        모델 저장
        
        Args:
            path (str): 저장 경로
        """
        if self.sb3_model is None:
            raise ValueError("저장할 모델이 없습니다.")
            
        # 경로가 지정되지 않은 경우 기본 경로 사용
        save_path = path or self.model_path
        
        if save_path:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 모델 저장
            self.sb3_model.save(save_path)
            logger.info(f"모델 저장 완료: {save_path}")
        else:
            raise ValueError("저장 경로가 지정되지 않았습니다.")
    
    @classmethod
    def load(cls, path: str, model_type: str = "ppo") -> 'RLModel':
        """
        저장된 모델 로드
        
        Args:
            path (str): 모델 경로
            model_type (str): 모델 유형
            
        Returns:
            RLModel: 로드된 모델 인스턴스
        """
        try:
            # 모델 타입에 따라 로드 함수 선택
            if model_type.lower() == "ppo":
                sb3_model = PPO.load(path)
            elif model_type.lower() == "a2c":
                sb3_model = A2C.load(path)
            elif model_type.lower() == "dqn":
                sb3_model = DQN.load(path)
            else:
                raise ValueError(f"지원하지 않는 모델 유형: {model_type}")
            
            # 래퍼 클래스 인스턴스 생성
            instance = cls(model_type=model_type, model_kwargs=sb3_model, model_path=path)
            logger.info(f"모델 로드 완료: {path}")
            
            return instance
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return None 