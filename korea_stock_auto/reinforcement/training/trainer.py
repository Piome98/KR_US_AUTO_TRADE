"""
한국 주식 자동매매 - 학습 모듈
강화학습 모델 학습 및 평가 기능
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
from sklearn.model_selection import train_test_split
from korea_stock_auto.reinforcement.rl_data.data_processor import DataProcessor
from korea_stock_auto.reinforcement.rl_models.rl_model import TradingEnvironment, RLModel, ModelEnsemble
from korea_stock_auto.reinforcement.rl_models.features import select_features, get_state_dim
from korea_stock_auto.utils import send_message
from korea_stock_auto.data.database import DatabaseManager
from stable_baselines3 import PPO, A2C, DQN # 강화학습 모델 추가 임포트 필요-모델 평가가
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

logger = logging.getLogger("stock_auto")

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    학습 중 최고 보상 달성 시 모델 저장 콜백
    """
    
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
    
    def _on_step(self) -> bool:
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
                    self.model.save(self.save_path)
        
        return True

class ModelTrainer:
    """강화학습 모델 학습 및 평가 클래스"""
    
    def __init__(self, db_manager=None, data_processor=None, output_dir="models"):
        """
        모델 트레이너 초기화
        
        Args:
            db_manager (DatabaseManager): 데이터베이스 관리자
            data_processor (DataProcessor): 데이터 전처리기
            output_dir (str): 출력 디렉토리
        """
        self.db_manager = db_manager or DatabaseManager()
        self.data_processor = data_processor or DataProcessor()
        self.output_dir = output_dir
        self.cache_dir = os.path.join(os.path.dirname(__file__), '../../../data/cache')
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'figures'), exist_ok=True)
    
    def _generate_model_id(self, model_type):
        """모델 ID 생성"""
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
                send_message(f"데이터 로드 실패: 해당 기간에 {code} 데이터가 없습니다.")
                return None
            
            # 데이터프레임 변환
            df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 숫자 컬럼으로 변환
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            send_message(f"{code} 데이터 로드 완료: {len(df)}개 데이터")
            return df
            
        except Exception as e:
            send_message(f"데이터 로드 실패: {e}")
            return None
    
    def prepare_dataset(self, data):
        """
        데이터셋 준비
        
        Args:
            data (pd.DataFrame): 원본 OHLCV 데이터
            
        Returns:
            tuple: (학습용 데이터, 테스트용 데이터)
        """
        if data is None or len(data) == 0:
            return None, None
        
        try:
            # 기술적 지표 추가
            processed_data = self.data_processor.add_technical_indicators(data)
            
            # 결측값 제거
            processed_data = processed_data.dropna()
            
            # 시간 순서대로 데이터 분할 (테스트 데이터는 최근 20%)
            train_size = int(len(processed_data) * 0.8)
            train_data = processed_data.iloc[:train_size]
            test_data = processed_data.iloc[train_size:]
            
            # 학습 데이터로 스케일러 학습 및 변환
            train_data_scaled = self.data_processor.normalize_data(train_data, fit=True)
            test_data_scaled = self.data_processor.normalize_data(test_data, fit=False)
            
            return train_data_scaled, test_data_scaled
            
        except Exception as e:
            send_message(f"데이터셋 준비 실패: {e}")
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
                'reward_scale': 0.01,         # 보상 스케일
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
                reward_scale=params['reward_scale'],
                holding_penalty=params['holding_penalty']
            )
            
            return env
            
        except Exception as e:
            send_message(f"환경 생성 실패: {e}")
            return None
    
    def train_model(self, env, model_type='ppo', timesteps=100000, **model_params):
        """
        모델 학습
        
        Args:
            env (TradingEnvironment): 학습 환경
            model_type (str): 모델 유형 ('ppo', 'a2c', 'dqn')
            timesteps (int): 학습 스텝 수
            **model_params: 모델 파라미터
            
        Returns:
            tuple: (모델 ID, 모델 인스턴스)
        """
        try:
            # 모델 ID 생성
            model_id = self._generate_model_id(model_type)
            model_path = os.path.join(self.output_dir, f"{model_id}.zip")
            
            # 모델 인스턴스 생성
            model = RLModel(
                model_type=model_type,
                model_kwargs=model_params,
                model_path=model_path
            )
            
            # 학습 시작
            send_message(f"모델 학습 시작 (ID: {model_id})")
            success = model.train(env, total_timesteps=timesteps, save=True)
            
            if success:
                send_message(f"모델 학습 완료: {model_id}")
                return model_id, model
            else:
                send_message(f"모델 학습 실패: {model_id}")
                return None, None
                
        except Exception as e:
            send_message(f"모델 학습 실패: {e}")
            return None, None
    
    def evaluate_model(self, model, test_data, output_prefix=None):
        """
        모델 평가
        
        Args:
            model (RLModel): 평가할 모델
            test_data (pd.DataFrame): 테스트 데이터
            output_prefix (str): 출력 파일 접두사
            
        Returns:
            dict: 평가 결과
        """
        try:
            # 평가 환경 생성
            env = self.create_environment(test_data)
            
            # 기록 초기화
            observations = []
            actions = []
            rewards = []
            portfolio_values = []
            
            # 초기 관찰값 가져오기
            observation = env.reset()
            done = False
            total_reward = 0
            
            # 에피소드 실행
            while not done:
                observations.append(observation)
                
                # 모델 예측
                action, _state = model.predict(observation, deterministic=True)
                actions.append(action)
                
                # 환경 스텝
                observation, reward, done, info = env.step(action)
                rewards.append(reward)
                total_reward += reward
                
                # 포트폴리오 가치 기록
                portfolio_values.append(env.history[-1]['portfolio_value'])
            
            # 최종 결과 계산
            initial_balance = env.initial_balance
            final_portfolio_value = portfolio_values[-1]
            total_return = (final_portfolio_value / initial_balance - 1) * 100  # %로 변환
            
            # 거래 횟수 계산
            buy_count = actions.count(1)
            sell_count = actions.count(2)
            
            # 결과 저장
            results = {
                'model_id': model.model_path.split('/')[-1].split('.')[0],
                'total_reward': total_reward,
                'total_return': total_return,
                'initial_balance': initial_balance,
                'final_portfolio_value': final_portfolio_value,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'total_trades': buy_count + sell_count
            }
            
            # 성능 그래프 생성 및 저장
            if output_prefix:
                self._save_performance_plots(env.history, output_prefix)
            
            send_message(f"모델 평가 완료: 수익률 {total_return:.2f}%, 거래 횟수: {buy_count + sell_count}")
            return results
            
        except Exception as e:
            send_message(f"모델 평가 실패: {e}")
            return None
    
    def _save_performance_plots(self, history, prefix):
        """
        성능 그래프 저장
        
        Args:
            history (list): 거래 이력
            prefix (str): 출력 파일 접두사
        """
        try:
            # 거래 이력 데이터프레임 변환
            df = pd.DataFrame(history)
            
            # 포트폴리오 가치 그래프
            plt.figure(figsize=(14, 7))
            plt.plot(df['portfolio_value'])
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Step')
            plt.ylabel('Portfolio Value')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'figures', f"{prefix}_portfolio.png"))
            plt.close()
            
            # 가격 및 거래 시점 그래프
            plt.figure(figsize=(14, 7))
            plt.plot(df['price'], label='Price')
            
            # 매수 시점 표시
            buy_points = df[df['action'] == 1]
            plt.scatter(buy_points.index, buy_points['price'], color='green', label='Buy', marker='^', s=100)
            
            # 매도 시점 표시
            sell_points = df[df['action'] == 2]
            plt.scatter(sell_points.index, sell_points['price'], color='red', label='Sell', marker='v', s=100)
            
            plt.title('Price and Trading Actions')
            plt.xlabel('Step')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'figures', f"{prefix}_trades.png"))
            plt.close()
            
            # 누적 보상 그래프
            plt.figure(figsize=(14, 7))
            plt.plot(df['reward'].cumsum())
            plt.title('Cumulative Reward')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'figures', f"{prefix}_reward.png"))
            plt.close()
            
        except Exception as e:
            send_message(f"성능 그래프 저장 실패: {e}")
    
    def save_model_info(self, model_id, model_type, hyperparams, train_period, results):
        """
        모델 정보 저장
        
        Args:
            model_id (str): 모델 ID
            model_type (str): 모델 유형
            hyperparams (dict): 하이퍼파라미터
            train_period (dict): 학습 기간 정보
            results (dict): 평가 결과
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 모델 정보 생성
            model_info = {
                'model_id': model_id,
                'model_type': model_type,
                'hyperparameters': hyperparams,
                'training_period': train_period,
                'evaluation_results': results,
                'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 정보 저장 디렉토리 생성
            log_dir = os.path.join(self.output_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # 모델 정보 파일 경로
            log_path = os.path.join(log_dir, f"{model_id}_info.json")
            
            # 모델 정보 저장
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            send_message(f"모델 정보 저장 완료: {log_path}")
            return True
            
        except Exception as e:
            send_message(f"모델 정보 저장 실패: {e}")
            return False
    
    def run_training_pipeline(
        self,
        code: str,
        model_type: str = "ppo",
        timesteps: int = 100000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feature_set: str = "api_enhanced"  # API 데이터를 활용하는 특성 세트 기본값으로 사용
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        학습 파이프라인 실행
        
        Args:
            code (str): 종목 코드
            model_type (str): 모델 유형 (ppo, a2c, dqn)
            timesteps (int): 학습 스텝 수
            start_date (str): 시작 날짜
            end_date (str): 종료 날짜
            feature_set (str): 사용할 특성 세트
            
        Returns:
            tuple: (모델 ID, 테스트 결과) 또는 (None, None)
        """
        try:
            send_message(f"{code} {model_type.upper()} 모델 학습 시작")
            
            # 데이터 준비 (API 데이터 활용)
            data = self.data_processor.combine_market_data(code)
            
            if data.empty:
                send_message(f"{code} 데이터가 없습니다.")
                return None, None
            
            # 필요한 특성 선택
            features = select_features(data, feature_set=feature_set)
            
            # 학습/테스트 데이터 분할
            train_data, test_data = self._split_data(features, test_ratio=0.2)
            
            # 모델 생성 및 학습
            model = self._train_model(
                train_data, 
                model_type=model_type, 
                timesteps=timesteps
            )
            
            if model is None:
                send_message(f"{code} 모델 학습 실패")
                return None, None
            
            # 모델 평가
            test_results = self._evaluate_model(model, test_data)
            
            # 모델 저장
            model_id = self._save_model(model, code, model_type, test_results)
            if model_id is None:
                send_message(f"{code} 모델 저장 실패")
                return None, None
            
            # 모델 메타데이터 저장
            self._save_model_metadata(
                model_id, 
                code, 
                model_type, 
                test_results, 
                feature_set=feature_set
            )
            
            send_message(f"{code} {model_type.upper()} 모델 학습 완료 (ID: {model_id})")
            return model_id, test_results
            
        except Exception as e:
            send_message(f"{code} 모델 학습 중 오류 발생: {e}")
            logger.error(f"모델 학습 파이프라인 실패: {e}", exc_info=True)
            return None, None
    
    def _split_data(
        self, 
        data: pd.DataFrame, 
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        데이터를 학습/테스트 세트로 분할
        
        Args:
            data (pd.DataFrame): 전체 데이터
            test_ratio (float): 테스트 데이터 비율
            
        Returns:
            tuple: (학습 데이터, 테스트 데이터)
        """
        # 데이터 정렬 확인
        if 'date' in data.columns:
            data = data.sort_values('date')
        
        # 분할 인덱스 계산
        split_idx = int(len(data) * (1 - test_ratio))
        
        # 데이터 분할
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        send_message(f"데이터 분할 완료: 학습 {len(train_data)}개, 테스트 {len(test_data)}개")
        return train_data, test_data
    
    def _train_model(
        self, 
        train_data: pd.DataFrame, 
        model_type: str = "ppo", 
        timesteps: int = 100000
    ) -> Optional[RLModel]:
        """
        강화학습 모델 학습
        
        Args:
            train_data (pd.DataFrame): 학습 데이터
            model_type (str): 모델 유형 (ppo, a2c, dqn)
            timesteps (int): 학습 스텝 수
            
        Returns:
            RLModel or None: 학습된 모델 또는 None
        """
        try:
            # 학습 환경 생성
            env = TradingEnvironment(df=train_data)
            
            # 모니터링을 위한 환경 래핑
            log_dir = os.path.join(self.output_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir)
            
            # 상태 차원 계산
            state_dim = get_state_dim(train_data)
            send_message(f"상태 차원: {state_dim}")
            
            # 모델 유형에 따른 모델 생성
            if model_type.lower() == "ppo":
                model = PPO(
                    "MlpPolicy", 
                    env, 
                    verbose=0,
                    learning_rate=0.0003,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01
                )
            elif model_type.lower() == "a2c":
                model = A2C(
                    "MlpPolicy", 
                    env, 
                    verbose=0,
                    learning_rate=0.0007,
                    n_steps=5,
                    gamma=0.99,
                    ent_coef=0.01
                )
            elif model_type.lower() == "dqn":
                model = DQN(
                    "MlpPolicy", 
                    env, 
                    verbose=0,
                    learning_rate=0.0001,
                    buffer_size=10000,
                    learning_starts=1000,
                    batch_size=64,
                    gamma=0.99,
                    exploration_fraction=0.2,
                    exploration_final_eps=0.1
                )
            else:
                send_message(f"지원하지 않는 모델 유형: {model_type}")
                return None
            
            # 임시 모델 저장 경로
            temp_model_path = os.path.join(self.output_dir, "temp_model")
            
            # 콜백 설정
            callback = SaveOnBestTrainingRewardCallback(
                check_freq=1000, 
                save_path=temp_model_path,
                verbose=1
            )
            
            # 모델 학습
            send_message(f"{model_type.upper()} 모델 학습 진행 중... ({timesteps} 스텝)")
            model.learn(total_timesteps=timesteps, callback=callback)
            
            # 최고 성능 모델 로드
            if os.path.exists(temp_model_path + ".zip"):
                if model_type.lower() == "ppo":
                    best_model = PPO.load(temp_model_path)
                elif model_type.lower() == "a2c":
                    best_model = A2C.load(temp_model_path)
                elif model_type.lower() == "dqn":
                    best_model = DQN.load(temp_model_path)
                
                # RLModel 래퍼로 변환
                rl_model = RLModel(sb3_model=best_model, model_type=model_type)
                send_message(f"{model_type.upper()} 모델 학습 완료")
                return rl_model
            else:
                # 최고 성능 모델이 없는 경우 현재 모델 반환
                rl_model = RLModel(sb3_model=model, model_type=model_type)
                send_message(f"{model_type.upper()} 모델 학습 완료 (최고 성능 모델 없음)")
                return rl_model
            
        except Exception as e:
            send_message(f"모델 학습 실패: {e}")
            logger.error(f"모델 학습 실패: {e}", exc_info=True)
            return None
    
    def _evaluate_model(
        self, 
        model: RLModel, 
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        학습된 모델 평가
        
        Args:
            model (RLModel): 평가할 모델
            test_data (pd.DataFrame): 테스트 데이터
            
        Returns:
            dict: 평가 결과
        """
        try:
            # 테스트 환경 생성
            env = TradingEnvironment(df=test_data)
            
            # 초기 자본 저장
            initial_balance = env.initial_balance
            
            # 테스트 시뮬레이션
            observation = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(observation)
                observation, reward, done, info = env.step(action)
            
            # 최종 포트폴리오 가치 계산
            final_balance = env.balance
            if env.shares_held > 0:
                final_balance += env.shares_held * test_data.iloc[-1]['close']
            
            # 수익률 계산
            total_return = (final_balance / initial_balance - 1) * 100
            
            # 거래 횟수 계산
            actions = [h['action'] for h in env.history]
            buy_count = actions.count(1)  # 매수
            sell_count = actions.count(2)  # 매도
            hold_count = actions.count(0)  # 관망
            total_trades = buy_count + sell_count
            
            # 최대 손실, 최대 이익 계산
            max_drawdown = 0
            max_profit = 0
            peak_value = initial_balance
            
            for i, record in enumerate(env.history):
                portfolio_value = record['portfolio_value']
                # 최대 손실 계산
                if portfolio_value > peak_value:
                    peak_value = portfolio_value
                else:
                    drawdown = (peak_value - portfolio_value) / peak_value * 100
                    max_drawdown = max(max_drawdown, drawdown)
                    
                # 최대 이익 계산 (이전 값 대비)
                if i > 0:
                    prev_value = env.history[i-1]['portfolio_value']
                    profit = (portfolio_value - prev_value) / prev_value * 100
                    max_profit = max(max_profit, profit)
            
            # 결과 반환
            results = {
                'total_return': total_return,
                'total_trades': total_trades,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count,
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'max_drawdown': max_drawdown,
                'max_profit': max_profit
            }
            
            send_message(f"모델 평가 완료: 수익률 {total_return:.2f}%, 거래 횟수: {total_trades}")
            return results
            
        except Exception as e:
            send_message(f"모델 평가 실패: {e}")
            logger.error(f"모델 평가 실패: {e}", exc_info=True)
            return {'total_return': 0, 'total_trades': 0}
    
    def _save_model(
        self, 
        model: RLModel, 
        code: str, 
        model_type: str, 
        test_results: Dict[str, Any]
    ) -> Optional[str]:
        """
        모델 저장
        
        Args:
            model (RLModel): 저장할 모델
            code (str): 종목 코드
            model_type (str): 모델 유형
            test_results (dict): 테스트 결과
            
        Returns:
            str or None: 모델 ID 또는 None
        """
        try:
            # 모델 ID 생성
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_id = f"{model_type}_{code}_{timestamp}"
            
            # 모델 디렉토리 생성
            model_dir = os.path.join(self.output_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Stable Baselines3 모델 저장
            sb3_model_path = os.path.join(model_dir, "sb3_model.zip")
            model.sb3_model.save(sb3_model_path)
            
            # RLModel 래퍼 저장
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            send_message(f"모델 저장 완료: {model_id}")
            return model_id
            
        except Exception as e:
            send_message(f"모델 저장 실패: {e}")
            logger.error(f"모델 저장 실패: {e}", exc_info=True)
            return None
    
    def _save_model_metadata(
        self, 
        model_id: str, 
        code: str, 
        model_type: str, 
        test_results: Dict[str, Any],
        feature_set: str = "technical"
    ) -> bool:
        """
        모델 메타데이터 저장
        
        Args:
            model_id (str): 모델 ID
            code (str): 종목 코드
            model_type (str): 모델 유형
            test_results (dict): 테스트 결과
            feature_set (str): 사용된 특성 세트
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 모델 메타데이터 생성
            metadata = {
                'model_id': model_id,
                'training_code': code,
                'model_type': model_type,
                'created_at': datetime.datetime.now().isoformat(),
                'test_results': test_results,
                'feature_set': feature_set,
                'total_return': test_results.get('total_return', 0)
            }
            
            # 메타데이터 파일 경로
            metadata_path = os.path.join(self.output_dir, model_id, "metadata.json")
            
            # 메타데이터 저장
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # 모델 목록 갱신
            self._update_model_list(model_id, metadata)
            
            return True
            
        except Exception as e:
            send_message(f"메타데이터 저장 실패: {e}")
            logger.error(f"메타데이터 저장 실패: {e}", exc_info=True)
            return False
    
    def _update_model_list(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """
        모델 목록 갱신
        
        Args:
            model_id (str): 모델 ID
            metadata (dict): 모델 메타데이터
        """
        try:
            # 모델 목록 파일 경로
            models_list_path = os.path.join(self.output_dir, "models_list.json")
            
            # 기존 목록 로드 또는 새로 생성
            if os.path.exists(models_list_path):
                with open(models_list_path, 'r') as f:
                    models_list = json.load(f)
            else:
                models_list = []
            
            # 새 모델 추가
            models_list.append(metadata)
            
            # 성능 기준으로 정렬 (수익률 내림차순)
            models_list.sort(key=lambda x: x.get('total_return', 0), reverse=True)
            
            # 저장
            with open(models_list_path, 'w') as f:
                json.dump(models_list, f, indent=2)
                
        except Exception as e:
            logger.error(f"모델 목록 갱신 실패: {e}", exc_info=True)


def create_ensemble_from_best_models(trainer: ModelTrainer, top_n: int = 3) -> Optional[ModelEnsemble]:
    """
    최고 성능 모델들로 앙상블 생성
    
    Args:
        trainer (ModelTrainer): 모델 학습기
        top_n (int): 앙상블에 포함할 모델 수
        
    Returns:
        ModelEnsemble or None: 앙상블 모델 또는 None
    """
    try:
        # 모델 목록 로드
        models_list_path = os.path.join(trainer.output_dir, "models_list.json")
        
        if not os.path.exists(models_list_path):
            send_message("모델 목록이 없습니다.")
            return None
        
        with open(models_list_path, 'r') as f:
            models_list = json.load(f)
        
        # 성능 기준으로 정렬 (이미 정렬되어 있지만 확실하게)
        models_list.sort(key=lambda x: x.get('total_return', 0), reverse=True)
        
        # 상위 N개 모델 선택
        top_models_info = models_list[:top_n]
        
        if not top_models_info:
            send_message("사용 가능한 모델이 없습니다.")
            return None
        
        # 모델 로드
        models = []
        weights = []
        
        for model_info in top_models_info:
            model_id = model_info.get('model_id')
            model_path = os.path.join(trainer.output_dir, model_id, "model.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # 모델과 가중치 추가 (수익률 기반 가중치)
                models.append(model)
                weights.append(max(0.1, model_info.get('total_return', 0)))
        
        if not models:
            send_message("모델을 로드할 수 없습니다.")
            return None
        
        # 가중치 정규화
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # 동일 가중치
            weights = [1.0 / len(models) for _ in models]
        
        # 앙상블 생성
        ensemble = ModelEnsemble(models=models, weights=weights)
        
        # 앙상블 저장
        ensemble_path = os.path.join(trainer.output_dir, "ensemble.pkl")
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble, f)
        
        # 앙상블 메타데이터 생성
        ensemble_metadata = {
            'ensemble_id': f"ensemble_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'created_at': datetime.datetime.now().isoformat(),
            'model_ids': [model_info.get('model_id') for model_info in top_models_info],
            'weights': weights,
            'models_count': len(models)
        }
        
        # 메타데이터 저장
        metadata_path = os.path.join(trainer.output_dir, "ensemble_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)
        
        send_message(f"앙상블 생성 완료: {len(models)}개 모델")
        return ensemble
        
    except Exception as e:
        send_message(f"앙상블 생성 실패: {e}")
        logger.error(f"앙상블 생성 실패: {e}", exc_info=True)
        return None 