"""
강화학습 모델 관련 유틸리티 함수
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from korea_stock_auto.utils import send_message
from korea_stock_auto.reinforcement.rl_models.rl_model import RLModel, ModelEnsemble


def list_available_models(models_dir="models"):
    """
    사용 가능한 모델 목록 반환
    
    Args:
        models_dir (str): 모델 디렉토리 경로
    
    Returns:
        list: 모델 정보 리스트
    """
    try:
        models_list = []
        
        # 모델 파일 확인
        for filename in os.listdir(models_dir):
            if filename.endswith(".zip"):
                model_id = filename.split(".")[0]
                
                # 모델 정보 파일 경로
                info_path = os.path.join(models_dir, "logs", f"{model_id}_info.json")
                
                model_info = {
                    "model_id": model_id,
                    "model_path": os.path.join(models_dir, filename),
                    "created_at": ""
                }
                
                # 모델 정보 파일이 있는 경우 추가 정보 로드
                if os.path.exists(info_path):
                    try:
                        with open(info_path, "r", encoding="utf-8") as f:
                            info = json.load(f)
                            
                        model_info.update({
                            "model_type": info.get("model_type", "unknown"),
                            "created_at": info.get("created_at", ""),
                            "total_return": info.get("evaluation_results", {}).get("total_return", 0),
                            "training_code": info.get("training_period", {}).get("code", "")
                        })
                    except Exception as e:
                        send_message(f"모델 정보 파일 로드 실패: {info_path}, {e}")
                
                models_list.append(model_info)
        
        # 생성일 기준 내림차순 정렬
        models_list.sort(key=lambda x: x["created_at"], reverse=True)
        
        return models_list
        
    except Exception as e:
        send_message(f"모델 목록 조회 실패: {e}")
        return []


def load_model_by_id(model_id, models_dir="models"):
    """
    ID로 모델 로드
    
    Args:
        model_id (str): 모델 ID
        models_dir (str): 모델 디렉토리 경로
    
    Returns:
        RLModel: 로드된 모델 또는 None
    """
    try:
        # 모델 정보 파일 경로
        info_path = os.path.join(models_dir, "logs", f"{model_id}_info.json")
        model_path = os.path.join(models_dir, f"{model_id}.zip")
        
        # 모델 정보 파일이 없거나 모델 파일이 없는 경우
        if not os.path.exists(info_path) or not os.path.exists(model_path):
            send_message(f"모델 또는 모델 정보를 찾을 수 없습니다: {model_id}")
            return None
        
        # 모델 정보 파일에서 모델 유형 로드
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
            
        model_type = info.get("model_type", "ppo")
        
        # 모델 인스턴스 생성 및 로드
        model = RLModel(model_type=model_type, model_path=model_path)
        if model.load():
            send_message(f"모델 로드 성공: {model_id}")
            return model
        else:
            send_message(f"모델 로드 실패: {model_id}")
            return None
            
    except Exception as e:
        send_message(f"모델 로드 실패: {e}")
        return None


def compare_models(model_ids, test_data, models_dir="models", output_path=None):
    """
    여러 모델의 성능 비교
    
    Args:
        model_ids (list): 비교할 모델 ID 리스트
        test_data (pd.DataFrame): 테스트 데이터
        models_dir (str): 모델 디렉토리 경로
        output_path (str): 비교 결과 저장 경로
    
    Returns:
        pd.DataFrame: 비교 결과
    """
    try:
        results = []
        
        # 각 모델 로드 및 평가
        for model_id in model_ids:
            model = load_model_by_id(model_id, models_dir)
            
            if model is None:
                continue
                
            # 평가 환경 설정
            from korea_stock_auto.reinforcement.rl_models.rl_model import TradingEnvironment
            env = TradingEnvironment(df=test_data)
            
            # 에피소드 실행
            observation = env.reset()
            done = False
            total_reward = 0
            
            # 시뮬레이션 실행
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, done, info = env.step(action)
                total_reward += reward
            
            # 결과 계산
            initial_balance = env.initial_balance
            final_value = env.balance
            if env.shares_held > 0:
                final_value += env.shares_held * test_data.iloc[-1]['close']
                
            total_return = (final_value / initial_balance - 1) * 100
            
            # 거래 횟수 계산
            actions = [h['action'] for h in env.history]
            buy_count = actions.count(1)
            sell_count = actions.count(2)
            
            # 결과 저장
            results.append({
                'model_id': model_id,
                'total_reward': total_reward,
                'total_return': total_return,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'total_trades': buy_count + sell_count
            })
            
        # 결과를 DataFrame으로 변환
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('total_return', ascending=False)
            
            # 결과 저장
            if output_path:
                df_results.to_csv(output_path, index=False)
                
            return df_results
        else:
            send_message("비교할 모델이 없습니다.")
            return pd.DataFrame()
            
    except Exception as e:
        send_message(f"모델 비교 실패: {e}")
        return pd.DataFrame()


def visualize_model_performance(model_id, test_data, models_dir="models", output_dir=None):
    """
    모델 성능 시각화
    
    Args:
        model_id (str): 모델 ID
        test_data (pd.DataFrame): 테스트 데이터
        models_dir (str): 모델 디렉토리 경로
        output_dir (str): 시각화 결과 저장 디렉토리
    
    Returns:
        bool: 성공 여부
    """
    try:
        # 모델 로드
        model = load_model_by_id(model_id, models_dir)
        
        if model is None:
            return False
            
        # 평가 환경 설정
        from korea_stock_auto.reinforcement.rl_models.rl_model import TradingEnvironment
        env = TradingEnvironment(df=test_data)
        
        # 에피소드 실행
        observation = env.reset()
        done = False
        
        # 시뮬레이션 실행
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
        
        # 결과 시각화
        history = env.history
        df_history = pd.DataFrame(history)
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = os.path.join(models_dir, "figures")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 포트폴리오 가치 그래프
        plt.figure(figsize=(14, 7))
        plt.plot(df_history['portfolio_value'])
        plt.title(f'Portfolio Value Over Time - {model_id}')
        plt.xlabel('Step')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{model_id}_portfolio.png"))
        plt.close()
        
        # 2. 가격 및 거래 시점 그래프
        plt.figure(figsize=(14, 7))
        plt.plot(df_history['price'], label='Price')
        
        # 매수 시점 표시
        buy_points = df_history[df_history['action'] == 1]
        if not buy_points.empty:
            plt.scatter(buy_points.index, buy_points['price'], color='green', label='Buy', marker='^', s=100)
        
        # 매도 시점 표시
        sell_points = df_history[df_history['action'] == 2]
        if not sell_points.empty:
            plt.scatter(sell_points.index, sell_points['price'], color='red', label='Sell', marker='v', s=100)
        
        plt.title(f'Price and Trading Actions - {model_id}')
        plt.xlabel('Step')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{model_id}_trades.png"))
        plt.close()
        
        # 3. 누적 보상 그래프
        plt.figure(figsize=(14, 7))
        plt.plot(df_history['reward'].cumsum())
        plt.title(f'Cumulative Reward - {model_id}')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{model_id}_reward.png"))
        plt.close()
        
        send_message(f"모델 성능 시각화 완료: {model_id}")
        return True
        
    except Exception as e:
        send_message(f"모델 성능 시각화 실패: {e}")
        return False 