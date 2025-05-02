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
from sklearn.model_selection import train_test_split
from korea_stock_auto.reinforcement.rl_data.data_processor import DataProcessor
from korea_stock_auto.reinforcement.rl_models.rl_model import TradingEnvironment, RLModel, ModelEnsemble
from korea_stock_auto.utils import send_message
from korea_stock_auto.data.database import DatabaseManager

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
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
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
    
    def run_training_pipeline(self, code, model_type='ppo', timesteps=100000, 
                             start_date=None, end_date=None, **model_params):
        """
        전체 학습 파이프라인 실행
        
        Args:
            code (str): 종목 코드
            model_type (str): 모델 유형
            timesteps (int): 학습 스텝 수
            start_date (str): 시작 날짜
            end_date (str): 종료 날짜
            **model_params: 모델 파라미터
            
        Returns:
            tuple: (모델 ID, 평가 결과)
        """
        try:
            # 1. 데이터 로드
            data = self.load_training_data(code, start_date, end_date)
            if data is None:
                return None, None
            
            # 2. 데이터셋 준비
            train_data, test_data = self.prepare_dataset(data)
            if train_data is None or test_data is None:
                return None, None
            
            # 3. 학습 환경 생성
            train_env = self.create_environment(train_data)
            if train_env is None:
                return None, None
            
            # 4. 모델 학습
            model_id, model = self.train_model(
                train_env, 
                model_type=model_type,
                timesteps=timesteps,
                **model_params
            )
            
            if model_id is None or model is None:
                return None, None
            
            # 5. 모델 평가
            results = self.evaluate_model(
                model, 
                test_data,
                output_prefix=model_id
            )
            
            if results is None:
                return model_id, None
            
            # 6. 모델 정보 저장
            train_period = {
                'code': code,
                'start_date': start_date or 'default',
                'end_date': end_date or 'default',
                'train_size': len(train_data),
                'test_size': len(test_data)
            }
            
            self.save_model_info(
                model_id, 
                model_type, 
                model_params, 
                train_period, 
                results
            )
            
            return model_id, results
            
        except Exception as e:
            send_message(f"학습 파이프라인 실패: {e}")
            return None, None


def create_ensemble_from_best_models(trainer, top_n=3):
    """
    최고 성능 모델들로 앙상블 생성
    
    Args:
        trainer (ModelTrainer): 모델 트레이너 인스턴스
        top_n (int): 선택할 상위 모델 수
        
    Returns:
        ModelEnsemble: 생성된 앙상블
    """
    try:
        # 로그 디렉토리 경로
        log_dir = os.path.join(trainer.output_dir, 'logs')
        
        # 모델 정보 파일 목록
        model_files = [f for f in os.listdir(log_dir) if f.endswith('_info.json')]
        
        if not model_files:
            send_message("앙상블을 위한 모델이 없습니다.")
            return None
        
        # 모델 정보 및 성능 로드
        models_info = []
        for file_name in model_files:
            file_path = os.path.join(log_dir, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                
                # 모델 ID 및 성능 추출
                model_id = model_info.get('model_id')
                model_type = model_info.get('model_type')
                model_path = os.path.join(trainer.output_dir, f"{model_id}.zip")
                
                # 모델 성능 지표
                evaluation = model_info.get('evaluation_results', {})
                total_return = evaluation.get('total_return', 0)
                
                if os.path.exists(model_path):
                    models_info.append({
                        'model_id': model_id,
                        'model_type': model_type,
                        'model_path': model_path,
                        'total_return': total_return
                    })
            except Exception as e:
                send_message(f"모델 정보 로드 실패: {file_path}, {e}")
        
        if not models_info:
            send_message("유효한 모델이 없습니다.")
            return None
        
        # 성능순으로 정렬
        models_info.sort(key=lambda x: x['total_return'], reverse=True)
        
        # 상위 N개 모델 선택
        selected_models = models_info[:top_n]
        
        # 앙상블 생성
        ensemble = ModelEnsemble()
        
        # 각 모델 로드 및 앙상블에 추가
        for model_info in selected_models:
            model_id = model_info['model_id']
            model_type = model_info['model_type']
            model_path = model_info['model_path']
            
            # 가중치 계산 (수익률 기반)
            total_return = model_info['total_return']
            weight = max(total_return, 1)  # 최소 가중치 1 보장
            
            # 모델 생성 및 로드
            model = RLModel(model_type=model_type, model_path=model_path)
            if model.load():
                ensemble.add_model(model_id, model, weight)
        
        if len(ensemble.models) == 0:
            send_message("앙상블 생성 실패: 모델을 로드할 수 없습니다.")
            return None
        
        # 앙상블 저장
        ensemble_path = os.path.join(trainer.output_dir, "ensemble.pkl")
        ensemble.save_ensemble(ensemble_path)
        
        send_message(f"앙상블 생성 완료: {len(ensemble.models)}개 모델, 저장 경로: {ensemble_path}")
        return ensemble
        
    except Exception as e:
        send_message(f"앙상블 생성 실패: {e}")
        return None 