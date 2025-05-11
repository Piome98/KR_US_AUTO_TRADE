"""
강화학습 모델 및 학습 환경 모듈
"""

import os
import pickle
import numpy as np
import datetime
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from korea_stock_auto.utils import send_message
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import logging

logger = logging.getLogger("stock_auto")


class TradingEnvironment(gym.Env):
    """
    주식 거래 강화학습 환경
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        initial_balance: float = 10000000, 
        commission: float = 0.00015, 
        window_size: int = 20,
        reward_scaling: float = 0.01
    ):
        """
        초기화
        
        Args:
            df (pd.DataFrame): 주가 데이터
            initial_balance (float): 초기 자본
            commission (float): 거래 수수료
            window_size (int): 관찰 창 크기
            reward_scaling (float): 보상 스케일링 계수
        """
        super(TradingEnvironment, self).__init__()
        
        # 데이터프레임 설정
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        
        # 현재 위치 및 보유 주식
        self.current_step = 0
        self.shares_held = 0
        self.current_price = 0
        self.last_trade_price = 0
        self.avg_buy_price = 0
        
        # 행동 공간: 0=관망, 1=매수, 2=매도
        self.action_space = spaces.Discrete(3)
        
        # 관찰 공간: 가격 데이터 + 포트폴리오 상태
        # 가격 정규화 특성 + 포트폴리오 특성(보유 비율, 수익률, 현금 비율)
        feature_columns = self._get_feature_columns()
        self.feature_cols = len(feature_columns)
        
        # 관찰 공간 정의
        price_features_dim = len(feature_columns)
        portfolio_features_dim = 3  # 보유 비율, 현재 수익률, 현금 비율
        
        self.observation_space = spaces.Box(
            low=-1, high=1, 
            shape=(price_features_dim + portfolio_features_dim,), 
            dtype=np.float32
        )
        
        # 거래 이력
        self.history = []
        
        # 보상 계산용 변수
        self.initial_portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
    
    def _get_feature_columns(self) -> List[str]:
        """
        데이터프레임에서 특성 컬럼 추출
        
        Returns:
            list: 특성 컬럼 목록
        """
        # 날짜, 코드, 원본 가격 데이터 제외
        exclude_cols = ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
        # '_norm'으로 끝나는 정규화된 특성만 선택
        feature_cols = [col for col in self.df.columns if col.endswith('_norm') and col not in exclude_cols]
        
        return feature_cols
    
    def reset(self) -> np.ndarray:
        """
        환경 초기화
        
        Returns:
            np.ndarray: 초기 관찰
        """
        # 상태 초기화
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_price = 0
        self.last_trade_price = 0
        self.avg_buy_price = 0
        
        # 이력 초기화
        self.history = []
        
        # 포트폴리오 가치 초기화
        self.initial_portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        환경 단계 실행
        
        Args:
            action (int): 0=관망, 1=매수, 2=매도
            
        Returns:
            tuple: (관찰, 보상, 종료 여부, 추가 정보)
        """
        # 현재 상태 저장
        self.current_price = self._get_current_price()
        previous_portfolio_value = self._calculate_portfolio_value()
        
        # 행동 실행
        reward = 0
        
        if action == 1:  # 매수
            # 모든 현금으로 매수
            # 수수료 고려하여 구매 가능한 최대 주식 수 계산
            max_shares = int(self.balance / (self.current_price * (1 + self.commission)))
            
            if max_shares > 0:
                # 구매 가능한 경우
                cost = max_shares * self.current_price * (1 + self.commission)
                
                # 평균 매수가 계산
                if self.shares_held > 0:
                    self.avg_buy_price = (self.avg_buy_price * self.shares_held + self.current_price * max_shares) / (self.shares_held + max_shares)
                else:
                    self.avg_buy_price = self.current_price
                
                self.balance -= cost
                self.shares_held += max_shares
                self.last_trade_price = self.current_price
                
                # 매수 시 소액의 즉시 보상 (잦은 매매 방지)
                reward = -0.001  # 약간의 페널티로 불필요한 매매 억제
                
        elif action == 2:  # 매도
            if self.shares_held > 0:
                # 모든 주식 매도
                # 수수료 고려한 수익 계산
                proceeds = self.shares_held * self.current_price * (1 - self.commission)
                
                # 손익 계산 (수수료 포함)
                profit = proceeds - (self.shares_held * self.avg_buy_price)
                
                self.balance += proceeds
                self.shares_held = 0
                self.last_trade_price = self.current_price
                
                # 매도 시 즉시 보상 (수익률에 따라)
                # 수익/손실 보상 조정 (수익 = 양수 보상, 손실 = 음수 보상)
                if self.avg_buy_price > 0:
                    profit_rate = (self.current_price - self.avg_buy_price) / self.avg_buy_price
                    reward = profit_rate * self.reward_scaling
                
                self.avg_buy_price = 0
        
        # 다음 단계로 이동
        self.current_step += 1
        
        # 현재 포트폴리오 가치 계산
        current_portfolio_value = self._calculate_portfolio_value()
        
        # 이력 업데이트
        self.history.append({
            'step': self.current_step,
            'price': self.current_price,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'action': action,
            'portfolio_value': current_portfolio_value
        })
        
        # 단계별 포트폴리오 변화율로 보상 조정
        # 0 행동(관망)인 경우에도 포트폴리오 가치 변화에 따른 보상 부여
        if action == 0 and self.previous_portfolio_value > 0:
            value_change_rate = (current_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
            reward += value_change_rate * self.reward_scaling
        
        # 다음 단계 포트폴리오 가치 비교를 위해 현재 가치 저장
        self.previous_portfolio_value = current_portfolio_value
        
        # 종료 조건: 데이터 끝에 도달
        done = self.current_step >= len(self.df) - 1
        
        # 에피소드 종료 시 추가 보상
        if done:
            # 전체 포트폴리오 수익률에 따른 최종 보상
            total_return = (current_portfolio_value / self.initial_portfolio_value) - 1
            reward += total_return * self.reward_scaling * 10  # 최종 수익률에 대한 보상 강화
            
            # 에피소드 종료 시 여전히 주식을 보유하고 있으면, 매도하도록 유도
            if self.shares_held > 0:
                reward -= 0.01  # 작은 페널티
        
        # 추가 정보
        info = {
            'portfolio_value': current_portfolio_value,
            'price': self.current_price,
            'shares_held': self.shares_held
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_current_price(self) -> float:
        """
        현재 가격 반환
        
        Returns:
            float: 현재 가격
        """
        if 'close' in self.df.columns:
            return self.df.loc[self.current_step, 'close']
        return 0
    
    def _calculate_portfolio_value(self) -> float:
        """
        현재 포트폴리오 가치 계산
        
        Returns:
            float: 포트폴리오 가치
        """
        return self.balance + self.shares_held * self.current_price
    
    def _get_observation(self) -> np.ndarray:
        """
        현재 관찰 벡터 생성
        
        Returns:
            np.ndarray: 관찰 벡터
        """
        # 특성 컬럼 가져오기
        feature_columns = self._get_feature_columns()
        
        # 가격 특성 추출
        price_features = self.df.loc[self.current_step, feature_columns].values
        
        # 포트폴리오 특성 계산
        # 1. 보유 주식 비율 (0~1 사이)
        portfolio_value = self._calculate_portfolio_value()
        shares_value = self.shares_held * self.current_price
        shares_ratio = min(1.0, shares_value / portfolio_value) if portfolio_value > 0 else 0
        
        # 2. 현재 손익률 (-1~1 사이)
        if self.shares_held > 0 and self.avg_buy_price > 0:
            profit_ratio = (self.current_price - self.avg_buy_price) / self.avg_buy_price
            profit_ratio = np.clip(profit_ratio, -1, 1)  # -100% ~ +100%로 클리핑
        else:
            profit_ratio = 0
        
        # 3. 현금 비율 (0~1 사이)
        cash_ratio = min(1.0, self.balance / portfolio_value) if portfolio_value > 0 else 1
        
        # 모든 특성 결합
        portfolio_features = np.array([shares_ratio, profit_ratio, cash_ratio], dtype=np.float32)
        observation = np.concatenate([price_features, portfolio_features])
        
        return observation
    
    def render(self, mode='human'):
        """
        환경 시각화 (미구현)
        """
        pass


class RLModel:
    """강화학습 모델 래퍼"""
    
    def __init__(self, sb3_model, model_type: str = "ppo"):
        """
        초기화
        
        Args:
            sb3_model: Stable Baselines3 모델
            model_type (str): 모델 유형 (ppo, a2c, dqn)
        """
        self.sb3_model = sb3_model
        self.model_type = model_type.lower()
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, np.ndarray]:
        """
        행동 예측
        
        Args:
            observation (np.ndarray): 관찰 벡터
            deterministic (bool): 결정론적 예측 여부
            
        Returns:
            tuple: (행동, 상태 값)
        """
        action, state = self.sb3_model.predict(observation, deterministic=deterministic)
        return action, state
    
    def save(self, path: str) -> None:
        """
        모델 저장
        
        Args:
            path (str): 저장 경로
        """
        self.sb3_model.save(path)
    
    @classmethod
    def load(cls, path: str, model_type: str = "ppo") -> 'RLModel':
        """
        모델 로드
        
        Args:
            path (str): 모델 경로
            model_type (str): 모델 유형
            
        Returns:
            RLModel: 로드된 모델
        """
        if model_type.lower() == "ppo":
            sb3_model = PPO.load(path)
        elif model_type.lower() == "a2c":
            sb3_model = A2C.load(path)
        elif model_type.lower() == "dqn":
            sb3_model = DQN.load(path)
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {model_type}")
        
        return cls(sb3_model, model_type)


class ModelEnsemble:
    """모델 앙상블"""
    
    def __init__(self, models: List[RLModel], weights: Optional[List[float]] = None):
        """
        초기화
        
        Args:
            models (list): RLModel 목록
            weights (list): 각 모델의 가중치
        """
        self.models = models
        
        # 가중치 설정 (기본값: 동일 가중치)
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # 가중치 정규화
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        앙상블 행동 예측
        
        Args:
            observation (np.ndarray): 관찰 벡터
            deterministic (bool): 결정론적 예측 여부
            
        Returns:
            tuple: (행동, 기타 정보)
        """
        # 각 모델의 예측 수집
        action_votes = {0: 0.0, 1: 0.0, 2: 0.0}
        
        for i, model in enumerate(self.models):
            action, _ = model.predict(observation, deterministic=deterministic)
            action_votes[action] += self.weights[i]
        
        # 가장 높은 투표를 받은 행동 선택
        max_vote = -1
        chosen_action = 0
        
        for action, vote in action_votes.items():
            if vote > max_vote:
                max_vote = vote
                chosen_action = action
        
        # 결과 반환
        return chosen_action, {'votes': action_votes}
    
    def save_ensemble(self, path: str) -> None:
        """
        앙상블 모델 저장
        
        Args:
            path (str): 저장 경로
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_ensemble(cls, path: str) -> 'ModelEnsemble':
        """
        앙상블 모델 로드
        
        Args:
            path (str): 모델 경로
            
        Returns:
            ModelEnsemble: 로드된 앙상블
        """
        with open(path, 'rb') as f:
            ensemble = pickle.load(f)
        
        return ensemble 