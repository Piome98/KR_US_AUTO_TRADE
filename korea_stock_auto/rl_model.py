"""
한국 주식 자동매매 - 강화학습 모델 모듈
다양한 강화학습 모델 구현 및 인터페이스 정의
"""

import os
import pickle
import numpy as np
import datetime
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from korea_stock_auto.utils import send_message

class TradingEnvironment(gym.Env):
    """주식 트레이딩 환경 (OpenAI Gym 기반)"""
    
    def __init__(self, df, initial_balance=10000000, commission=0.00015, 
                window_size=20, reward_scale=0.01, holding_penalty=0.0001):
        """
        트레이딩 환경 초기화
        
        Args:
            df (pd.DataFrame): 주가 데이터
            initial_balance (float): 초기 자산
            commission (float): 수수료 비율
            window_size (int): 관찰 창 크기
            reward_scale (float): 보상 스케일링 계수
            holding_penalty (float): 홀딩에 대한 패널티
        """
        super(TradingEnvironment, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.reward_scale = reward_scale
        self.holding_penalty = holding_penalty
        
        # 액션 및 관찰 공간 정의
        self.action_space = spaces.Discrete(3)  # 0: 홀드, 1: 매수, 2: 매도
        
        # 관찰 공간 (가격 데이터 + 포지션 정보)
        n_features = len(df.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, n_features + 2), dtype=np.float32
        )
        
        # 환경 변수 초기화
        self.reset()
    
    def reset(self):
        """환경 초기화"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_position = 0  # 0: 중립, 1: 롱
        self.cost_basis = 0
        self.total_realized_pnl = 0
        self.history = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """현재 관찰 상태 반환"""
        # 관찰 창 내의 데이터 추출
        frame = self.df.iloc[self.current_step - self.window_size:self.current_step].copy()
        
        # 포지션 정보 추가
        position = np.zeros((len(frame), 2))
        position[:, 0] = self.current_position
        position[:, 1] = self.shares_held / 100  # 스케일링
        
        # 관찰 데이터 구성
        obs = np.column_stack([frame.values, position])
        
        return obs
    
    def step(self, action):
        """
        환경에서 한 스텝 진행
        
        Args:
            action (int): 0: 홀드, 1: 매수, 2: 매도
            
        Returns:
            tuple: (관찰, 보상, 종료 여부, 추가 정보)
        """
        # 현재 가격 가져오기
        current_price = self.df.iloc[self.current_step]['close']
        
        # 액션에 따른 보상 계산
        reward = 0
        done = False
        info = {}
        
        # 매수 액션
        if action == 1 and self.current_position == 0:
            max_shares = int(self.balance / (current_price * (1 + self.commission)))
            if max_shares > 0:
                self.shares_held = max_shares
                self.cost_basis = current_price
                self.balance -= self.shares_held * current_price * (1 + self.commission)
                self.current_position = 1
                
        # 매도 액션
        elif action == 2 and self.current_position == 1:
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price * (1 - self.commission)
                realized_pnl = self.shares_held * (current_price - self.cost_basis)
                self.total_realized_pnl += realized_pnl
                self.shares_held = 0
                self.current_position = 0
                
                # 매도 시 실현 수익률에 따른 보상
                profit_percent = (current_price / self.cost_basis - 1) - self.commission * 2
                reward = profit_percent * self.reward_scale
        
        # 홀딩 중인 경우 미실현 손익 계산
        if self.current_position == 1:
            unrealized_pnl = self.shares_held * (current_price - self.cost_basis)
            profit_percent = current_price / self.cost_basis - 1
            
            # 보유 패널티 적용 (장기 홀딩 방지)
            if action == 0:
                reward = profit_percent * self.reward_scale - self.holding_penalty
            else:
                reward = profit_percent * self.reward_scale
        
        # 포트폴리오 가치 계산
        portfolio_value = self.balance
        if self.shares_held > 0:
            portfolio_value += self.shares_held * current_price
        
        # 거래 이력 기록
        self.history.append({
            'step': self.current_step,
            'price': current_price,
            'action': action,
            'shares': self.shares_held,
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'reward': reward
        })
        
        # 다음 스텝으로 진행
        self.current_step += 1
        
        # 데이터 종료 확인
        if self.current_step >= len(self.df) - 1:
            done = True
            
            # 포지션이 열려있으면 마지막에 정산
            if self.current_position == 1:
                self.balance += self.shares_held * current_price * (1 - self.commission)
                realized_pnl = self.shares_held * (current_price - self.cost_basis)
                self.total_realized_pnl += realized_pnl
                
                # 최종 성과 정보
                info['final_portfolio_value'] = self.balance
                info['total_pnl'] = self.total_realized_pnl
                info['return'] = self.balance / self.initial_balance - 1
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """환경 시각화 (현재 미구현)"""
        pass


class RLModel:
    """강화학습 모델 인터페이스"""
    
    def __init__(self, model_type='ppo', model_kwargs=None, model_path=None):
        """
        RL 모델 초기화
        
        Args:
            model_type (str): 모델 유형 ('ppo', 'a2c', 'dqn')
            model_kwargs (dict): 모델 파라미터
            model_path (str): 모델 저장 경로
        """
        self.model_type = model_type.lower()
        self.model_kwargs = model_kwargs or {}
        self.model_path = model_path
        self.model = None
        
        # 모델 디렉토리 생성
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def _create_model(self, env):
        """모델 타입에 따른 인스턴스 생성"""
        if self.model_type == 'ppo':
            model = PPO('MlpPolicy', env, verbose=1, **self.model_kwargs)
        elif self.model_type == 'a2c':
            model = A2C('MlpPolicy', env, verbose=1, **self.model_kwargs)
        elif self.model_type == 'dqn':
            model = DQN('MlpPolicy', env, verbose=1, **self.model_kwargs)
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {self.model_type}")
        
        return model
    
    def train(self, env, total_timesteps=100000, save=True):
        """
        모델 학습
        
        Args:
            env (gym.Env): 학습 환경
            total_timesteps (int): 총 학습 스텝
            save (bool): 모델 저장 여부
            
        Returns:
            bool: 학습 성공 여부
        """
        try:
            # 환경을 DummyVecEnv로 래핑
            if not isinstance(env, DummyVecEnv):
                env = DummyVecEnv([lambda: env])
            
            # 모델 생성
            self.model = self._create_model(env)
            
            # 학습 시작
            start_time = datetime.datetime.now()
            send_message(f"모델 학습 시작 (타입: {self.model_type}, 스텝: {total_timesteps})")
            
            self.model.learn(total_timesteps=total_timesteps)
            
            end_time = datetime.datetime.now()
            train_time = (end_time - start_time).total_seconds() / 60
            send_message(f"모델 학습 완료 (소요 시간: {train_time:.2f}분)")
            
            # 모델 저장
            if save and self.model_path:
                self.save()
            
            return True
            
        except Exception as e:
            send_message(f"모델 학습 실패: {e}")
            return False
    
    def predict(self, observation, deterministic=True):
        """
        행동 예측
        
        Args:
            observation (numpy.ndarray): 환경 관찰값
            deterministic (bool): 결정론적 예측 여부
            
        Returns:
            tuple: (action, _state)
        """
        if self.model is None:
            send_message("모델이 초기화되지 않았습니다.")
            return 0, None
        
        try:
            return self.model.predict(observation, deterministic=deterministic)
        except Exception as e:
            send_message(f"예측 실패: {e}")
            return 0, None
    
    def save(self, path=None):
        """모델 저장"""
        if self.model is None:
            send_message("저장할 모델이 없습니다.")
            return False
        
        try:
            save_path = path or self.model_path
            if save_path:
                self.model.save(save_path)
                send_message(f"모델 저장 완료: {save_path}")
                return True
            else:
                send_message("저장 경로가 지정되지 않았습니다.")
                return False
                
        except Exception as e:
            send_message(f"모델 저장 실패: {e}")
            return False
    
    def load(self, path=None):
        """모델 로드"""
        try:
            load_path = path or self.model_path
            
            if not load_path or not os.path.exists(load_path):
                send_message(f"모델 파일이 존재하지 않습니다: {load_path}")
                return False
            
            # 모델 타입에 따라 로드
            if self.model_type == 'ppo':
                self.model = PPO.load(load_path)
            elif self.model_type == 'a2c':
                self.model = A2C.load(load_path)
            elif self.model_type == 'dqn':
                self.model = DQN.load(load_path)
            else:
                send_message(f"지원하지 않는 모델 유형: {self.model_type}")
                return False
            
            send_message(f"모델 로드 성공: {load_path}")
            return True
            
        except Exception as e:
            send_message(f"모델 로드 실패: {e}")
            return False


class ModelEnsemble:
    """모델 앙상블 클래스"""
    
    def __init__(self, models=None, weights=None):
        """
        모델 앙상블 초기화
        
        Args:
            models (dict): 모델 인스턴스 딕셔너리 (id: model)
            weights (dict): 모델 가중치 딕셔너리 (id: weight)
        """
        self.models = models or {}
        self.weights = weights or {}
        
        # 가중치가 없는 모델은 기본값 1.0 설정
        for model_id in self.models:
            if model_id not in self.weights:
                self.weights[model_id] = 1.0
    
    def add_model(self, model_id, model, weight=1.0):
        """
        앙상블에 모델 추가
        
        Args:
            model_id (str): 모델 ID
            model (RLModel): 모델 인스턴스
            weight (float): 모델 가중치
        """
        self.models[model_id] = model
        self.weights[model_id] = weight
    
    def remove_model(self, model_id):
        """앙상블에서 모델 제거"""
        if model_id in self.models:
            del self.models[model_id]
        if model_id in self.weights:
            del self.weights[model_id]
    
    def predict(self, observation):
        """
        앙상블 예측 수행
        
        Args:
            observation (numpy.ndarray): 환경 관찰값
            
        Returns:
            tuple: (action, confidence)
        """
        if not self.models:
            return 0, 0.5  # 기본값: 홀드, 낮은 확신도
        
        # 각 모델의 예측 및 가중치 계산
        total_weight = sum(self.weights.values())
        action_scores = {0: 0.0, 1: 0.0, 2: 0.0}
        
        for model_id, model in self.models.items():
            weight = self.weights.get(model_id, 1.0)
            normalized_weight = weight / total_weight
            
            action, _ = model.predict(observation)
            action_scores[action] += normalized_weight
        
        # 최대 점수 액션 선택
        best_action = max(action_scores, key=action_scores.get)
        confidence = action_scores[best_action]
        
        return best_action, confidence
    
    def update_weights(self, new_weights):
        """
        모델 가중치 업데이트
        
        Args:
            new_weights (dict): 새로운 가중치 딕셔너리
        """
        for model_id, weight in new_weights.items():
            if model_id in self.models:
                self.weights[model_id] = weight
    
    def save_ensemble(self, path="models/ensemble.pkl"):
        """앙상블 설정 저장"""
        try:
            # 저장할 데이터 (모델 인스턴스 제외)
            data = {
                'weights': self.weights,
                'model_types': {model_id: model.model_type for model_id, model in self.models.items()},
                'model_paths': {model_id: model.model_path for model_id, model in self.models.items()}
            }
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(data, f)
                
            send_message(f"앙상블 설정 저장 완료: {path}")
            return True
            
        except Exception as e:
            send_message(f"앙상블 저장 실패: {e}")
            return False
    
    @classmethod
    def load_ensemble(cls, path="models/ensemble.pkl"):
        """앙상블 설정 로드"""
        try:
            if not os.path.exists(path):
                send_message(f"앙상블 설정 파일이 존재하지 않습니다: {path}")
                return None
            
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            weights = data.get('weights', {})
            model_types = data.get('model_types', {})
            model_paths = data.get('model_paths', {})
            
            # 앙상블 인스턴스 생성
            ensemble = cls(weights=weights)
            
            # 모델 로드
            for model_id in weights:
                if model_id in model_types and model_id in model_paths:
                    model = RLModel(
                        model_type=model_types[model_id],
                        model_path=model_paths[model_id]
                    )
                    
                    if model.load():
                        ensemble.add_model(model_id, model, weights[model_id])
            
            send_message(f"앙상블 설정 로드 완료: {len(ensemble.models)}개 모델")
            return ensemble
            
        except Exception as e:
            send_message(f"앙상블 로드 실패: {e}")
            return None 