"""
한국 주식 자동매매 - 강화학습 환경 모듈

주식 거래를 위한 OpenAI Gym 기반 강화학습 환경 제공
"""

import os
import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import List, Dict, Any, Optional, Tuple, Union
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
        reward_scaling: float = 0.01,
        holding_penalty: float = 0.0001
    ):
        """
        초기화
        
        Args:
            df (pd.DataFrame): 주가 데이터
            initial_balance (float): 초기 자본
            commission (float): 거래 수수료
            window_size (int): 관찰 창 크기
            reward_scaling (float): 보상 스케일링 계수
            holding_penalty (float): 주식 보유 페널티
        """
        super(TradingEnvironment, self).__init__()
        
        # 데이터프레임 설정
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.holding_penalty = holding_penalty
        
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
            
            # 주식 보유 시 홀딩 패널티 적용 (장기 보유 억제)
            if self.shares_held > 0:
                reward -= self.holding_penalty
        
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
        # 가격 특성 추출
        features = self._get_feature_columns()
        price_features = self.df.loc[self.current_step, features].values
        
        # 포트폴리오 상태 특성 계산
        portfolio_value = self._calculate_portfolio_value()
        
        # 1. 보유 비율: 현재 보유 주식 가치 / 포트폴리오 가치
        if portfolio_value > 0:
            holdings_ratio = (self.shares_held * self.current_price) / portfolio_value
        else:
            holdings_ratio = 0
            
        # 2. 현재 수익률: (현재가 - 평균매수가) / 평균매수가
        if self.avg_buy_price > 0 and self.shares_held > 0:
            profit_ratio = (self.current_price - self.avg_buy_price) / self.avg_buy_price
        else:
            profit_ratio = 0
            
        # 3. 현금 비율: 현재 현금 / 초기 자본
        cash_ratio = self.balance / self.initial_balance
        
        # 최종 관찰 벡터 (가격 특성 + 포트폴리오 특성)
        portfolio_features = np.array([holdings_ratio, profit_ratio, cash_ratio])
        
        return np.concatenate([price_features, portfolio_features]).astype(np.float32)
    
    def render(self, mode='human'):
        """
        환경 시각화 (현재는 간단한 콘솔 출력)
        
        Args:
            mode (str): 렌더링 모드
        """
        profit = self._calculate_portfolio_value() - self.initial_balance
        print(f"Step: {self.current_step}, Price: {self.current_price:.2f}, "
              f"Shares: {self.shares_held}, Balance: {self.balance:.2f}, "
              f"Profit: {profit:.2f}") 