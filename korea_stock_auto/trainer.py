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
from korea_stock_auto.data_processor import DataProcessor
from korea_stock_auto.rl_model import TradingEnvironment, RLModel, ModelEnsemble
from korea_stock_auto.utils import send_message
from korea_stock_auto.database import DatabaseManager

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
            
            # 결과 계산
            final_portfolio = env.history[-1]['portfolio_value']
            initial_portfolio = env.initial_balance
            total_return = final_portfolio / initial_portfolio - 1
            
            # 샤프 비율 계산 (일별 수익률 사용)
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            
            # 최대 낙폭 계산
            max_drawdown = 0
            peak = portfolio_values[0]
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # 승률 계산
            trade_count = sum(1 for i in range(len(env.history)-1) 
                             if env.history[i]['action'] == 2 and env.history[i]['shares'] > 0)
            win_count = sum(1 for i in range(len(env.history)-1) 
                           if env.history[i]['action'] == 2 and env.history[i]['shares'] > 0 
                           and env.history[i]['price'] > env.history[i-1]['price'])
            win_rate = win_count / trade_count if trade_count > 0 else 0
            
            # 결과 저장
            result = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'trade_count': trade_count,
                'final_portfolio': final_portfolio,
                'total_reward': total_reward
            }
            
            # 성능 그래프 저장
            if output_prefix:
                self._save_performance_plots(env.history, output_prefix)
            
            send_message(f"모델 평가 결과: 수익률 {total_return:.2%}, 샤프 비율 {sharpe_ratio:.2f}, 최대 낙폭 {max_drawdown:.2%}, 승률 {win_rate:.2%}")
            return result
            
        except Exception as e:
            send_message(f"모델 평가 실패: {e}")
            return None
    
    def _save_performance_plots(self, history, prefix):
        """
        성능 그래프 저장
        
        Args:
            history (list): 거래 이력
            prefix (str): 파일 접두사
        """
        try:
            # 데이터 추출
            steps = [h['step'] for h in history]
            prices = [h['price'] for h in history]
            actions = [h['action'] for h in history]
            portfolio_values = [h['portfolio_value'] for h in history]
            
            # 포트폴리오 가치 그래프
            plt.figure(figsize=(12, 6))
            plt.plot(steps, portfolio_values)
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Step')
            plt.ylabel('Portfolio Value')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'figures', f"{prefix}_portfolio.png"))
            
            # 가격 및 액션 그래프
            plt.figure(figsize=(12, 8))
            
            # 가격 그래프
            plt.subplot(2, 1, 1)
            plt.plot(steps, prices)
            plt.title('Price Over Time')
            plt.grid(True)
            
            # 액션 그래프
            plt.subplot(2, 1, 2)
            buy_steps = [steps[i] for i in range(len(steps)) if actions[i] == 1]
            sell_steps = [steps[i] for i in range(len(steps)) if actions[i] == 2]
            
            # 각 액션 지점에 점 표시
            plt.plot(steps, prices, color='blue', alpha=0.5)
            plt.scatter(buy_steps, [prices[steps.index(s)] for s in buy_steps], color='green', label='Buy')
            plt.scatter(sell_steps, [prices[steps.index(s)] for s in sell_steps], color='red', label='Sell')
            plt.title('Actions')
            plt.xlabel('Step')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'figures', f"{prefix}_actions.png"))
            plt.close('all')
            
        except Exception as e:
            send_message(f"그래프 저장 실패: {e}")
    
    def save_model_info(self, model_id, model_type, hyperparams, train_period, results):
        """
        모델 정보 저장
        
        Args:
            model_id (str): 모델 ID
            model_type (str): 모델 유형
            hyperparams (dict): 하이퍼파라미터
            train_period (str): 학습 기간
            results (dict): 평가 결과
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 모델 정보 파일 경로
            info_path = os.path.join(self.output_dir, f"{model_id}_info.json")
            
            # 모델 정보 구성
            info = {
                'model_id': model_id,
                'model_type': model_type,
                'creation_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'hyperparameters': hyperparams,
                'training_period': train_period,
                'evaluation_results': results
            }
            
            # JSON 파일로 저장
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4)
            
            send_message(f"모델 정보 저장 완료: {info_path}")
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
            tuple: (모델 ID, 모델 인스턴스, 평가 결과)
        """
        # 1. 데이터 로드
        data = self.load_training_data(code, start_date, end_date)
        if data is None:
            return None, None, None
        
        # 2. 데이터셋 준비
        train_data, test_data = self.prepare_dataset(data)
        if train_data is None or test_data is None:
            return None, None, None
        
        # 3. 학습 환경 생성
        train_env = self.create_environment(train_data)
        if train_env is None:
            return None, None, None
        
        # 4. 모델 학습
        model_id, model = self.train_model(
            train_env, 
            model_type=model_type, 
            timesteps=timesteps, 
            **model_params
        )
        if model_id is None or model is None:
            return None, None, None
        
        # 5. 모델 평가
        results = self.evaluate_model(
            model, 
            test_data, 
            output_prefix=model_id
        )
        if results is None:
            return model_id, model, None
        
        # 6. 모델 정보 저장
        train_period = f"{train_data.index[0].strftime('%Y-%m-%d')} ~ {train_data.index[-1].strftime('%Y-%m-%d')}"
        self.save_model_info(
            model_id, 
            model_type, 
            model_params, 
            train_period, 
            results
        )
        
        return model_id, model, results


def create_ensemble_from_best_models(trainer, top_n=3):
    """
    최고 성능의 모델로 앙상블 생성
    
    Args:
        trainer (ModelTrainer): 모델 트레이너
        top_n (int): 앙상블에 포함할 모델 수
        
    Returns:
        ModelEnsemble: 앙상블 모델
    """
    try:
        # 모델 정보 파일 로드
        model_files = [f for f in os.listdir(trainer.output_dir) if f.endswith('_info.json')]
        models_info = []
        
        for mf in model_files:
            try:
                with open(os.path.join(trainer.output_dir, mf), 'r') as f:
                    info = json.load(f)
                    
                # 평가 결과가 있는 모델만 추가
                if 'evaluation_results' in info and info['evaluation_results']:
                    models_info.append(info)
            except Exception:
                continue
        
        if not models_info:
            send_message("앙상블 생성에 사용할 모델이 없습니다.")
            return None
        
        # 수익률 기준으로 정렬
        sorted_models = sorted(
            models_info, 
            key=lambda x: x['evaluation_results'].get('total_return', 0), 
            reverse=True
        )
        
        # 상위 N개 모델 선택
        top_models = sorted_models[:top_n]
        
        # 앙상블 생성
        ensemble = ModelEnsemble()
        
        for model_info in top_models:
            model_id = model_info['model_id']
            model_type = model_info['model_type']
            model_path = os.path.join(trainer.output_dir, f"{model_id}.zip")
            
            # 모델 로드
            if os.path.exists(model_path):
                model = RLModel(model_type=model_type, model_path=model_path)
                if model.load():
                    # 수익률에 비례한 가중치 설정
                    weight = max(0.1, model_info['evaluation_results'].get('total_return', 0) * 10)
                    ensemble.add_model(model_id, model, weight)
        
        if not ensemble.models:
            send_message("앙상블에 추가된 모델이 없습니다.")
            return None
        
        # 앙상블 저장
        ensemble.save_ensemble(os.path.join(trainer.output_dir, "ensemble.pkl"))
        
        send_message(f"앙상블 생성 완료: {len(ensemble.models)}개 모델")
        return ensemble
        
    except Exception as e:
        send_message(f"앙상블 생성 실패: {e}")
        return None 