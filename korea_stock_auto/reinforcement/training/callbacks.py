"""
한국 주식 자동매매 - 강화학습 콜백 모듈

학습 과정에서 사용되는 콜백 클래스 제공
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import logging

logger = logging.getLogger("stock_auto")

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    학습 중 최고 보상 달성 시 모델 저장 콜백
    """
    
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        """
        콜백 초기화
        
        Args:
            check_freq (int): 체크 주기
            save_path (str): 저장 경로
            verbose (int): 상세 출력 수준
        """
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
    def _init_callback(self) -> None:
        """콜백 초기화"""
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
    
    def _on_step(self) -> bool:
        """
        각 스텝마다 호출되는 메서드
        
        Returns:
            bool: 계속 학습 진행 여부
        """
        if self.n_calls % self.check_freq == 0:
            # 최근 100개 에피소드의 평균 보상 계산
            rewards = self.model.ep_info_buffer
            if len(rewards) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in rewards])
                
                # 최고 보상 달성 시 모델 저장
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path} (Reward: {mean_reward:.2f})")
                        logger.info(f"새로운 최고 모델 저장: {self.save_path} (보상: {mean_reward:.2f})")
                    self.model.save(self.save_path)
        
        return True


class EarlyStoppingCallback(BaseCallback):
    """
    조기 중단 콜백 - 일정 에피소드 동안 개선이 없으면 학습 중단
    """
    
    def __init__(self, check_freq: int, patience: int = 10, min_delta: float = 0.1, verbose: int = 1):
        """
        콜백 초기화
        
        Args:
            check_freq (int): 체크 주기
            patience (int): 개선 없이 대기할 에피소드 수
            min_delta (float): 개선으로 간주할 최소 보상 증가량
            verbose (int): 상세 출력 수준
        """
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
    
    def _on_step(self) -> bool:
        """
        각 스텝마다 호출되는 메서드
        
        Returns:
            bool: 계속 학습 진행 여부
        """
        if self.n_calls % self.check_freq == 0:
            # 최근 에피소드의 평균 보상 계산
            rewards = self.model.ep_info_buffer
            if len(rewards) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in rewards])
                
                # 개선 여부 확인
                if mean_reward > self.best_mean_reward + self.min_delta:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        logger.info(f"보상 개선: {mean_reward:.2f}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        logger.info(f"보상 정체: {mean_reward:.2f}, 카운트: {self.no_improvement_count}/{self.patience}")
                
                # 조기 중단 여부 결정
                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        logger.info(f"조기 중단: {self.patience}회 동안 개선 없음")
                    return False
        
        return True 