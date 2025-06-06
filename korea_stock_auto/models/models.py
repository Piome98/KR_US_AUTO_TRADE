"""
한국 주식 자동매매 - 강화학습 모델 모듈
모델 로드, 예측 및 관리 기능
"""

import os
import json
import numpy as np
import pickle
import datetime
from korea_stock_auto.utils.utils import send_message

class ModelManager:
    """강화학습 모델 관리 클래스"""
    
    def __init__(self, model_dir="models"):
        """
        모델 관리자 초기화
        
        Args:
            model_dir (str): 모델 파일 저장 디렉토리
        """
        self.model_dir = model_dir
        self.model = None
        self.fallback_mode = False
        
        # 모델 디렉토리 생성
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 기본 모델 로드 시도
        self.load_latest_model()
    
    def load_latest_model(self):
        """최신 모델 로드"""
        try:
            # 모델 디렉토리에서 가장 최신 파일 찾기
            model_files = [f for f in os.listdir(self.model_dir) 
                          if f.endswith('.pkl') and f.startswith('rl_model_')]
            
            if not model_files:
                send_message("사용 가능한 모델 파일이 없습니다. 기본 전략을 사용합니다.", config.notification.discord_webhook_url)
                self.fallback_mode = True
                return False
            
            # 가장 최신 파일 선택 (날짜 기준)
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(self.model_dir, latest_model)
            
            # 모델 로드
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            send_message(f"모델 로드 성공: {latest_model}", config.notification.discord_webhook_url)
            self.fallback_mode = False
            return True
            
        except Exception as e:
            send_message(f"모델 로드 실패: {e}", config.notification.discord_webhook_url)
            self.fallback_mode = True
            return False
    
    def predict(self, state):
        """
        상태 벡터로부터 행동 예측
        
        Args:
            state (numpy.ndarray): 상태 벡터
            
        Returns:
            tuple: (action, confidence)
                action: 0(홀드), 1(매수), 2(매도)
                confidence: 행동에 대한 확신도 (0.0~1.0)
        """
        if self.fallback_mode or self.model is None:
            return self._fallback_strategy(state)
        
        try:
            # 모델에 따라 입력 형태 조정이 필요할 수 있음
            state_input = np.array(state).reshape(1, -1)
            
            # 예측 수행 (모델에 따라 출력 형태가 다를 수 있음)
            prediction = self.model.predict(state_input)
            
            # 예시: 모델이 각 행동의 확률을 출력하는 경우
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(state_input)[0]
                action = np.argmax(probs)
                confidence = probs[action]
            else:
                # 단순 분류 모델인 경우
                action = prediction[0]
                confidence = 0.6  # 기본 확신도
            
            return int(action), float(confidence)
            
        except Exception as e:
            send_message(f"예측 실패: {e}", config.notification.discord_webhook_url)
            return self._fallback_strategy(state)
    
    def _fallback_strategy(self, state):
        """
        모델 실패 시 사용할 기본 전략
        
        Args:
            state (numpy.ndarray): 상태 벡터
            
        Returns:
            tuple: (action, confidence)
        """
        # 상태 벡터에서 필요한 값 추출 (인덱스는 상태 벡터 설계에 따라 달라질 수 있음)
        try:
            price_ma5_ratio = state[0]  # 현재가/5일 이평선 비율
            price_ma20_ratio = state[1]  # 현재가/20일 이평선 비율
            
            # 이동평균선 기반 전략 (보수적으로 설정)
            if price_ma5_ratio < -0.03 and price_ma20_ratio < -0.02:
                # 이평선보다 충분히 낮을 때만 매수
                return 1, 0.6  # 매수, 중간 확신도
            elif price_ma5_ratio > 0.03:
                # 5일선 대비 상승 시 매도
                return 2, 0.6  # 매도, 중간 확신도
            else:
                # 그 외에는 홀드
                return 0, 0.5  # 홀드, 낮은 확신도
        except Exception:
            # 안전하게 홀드 반환
            return 0, 0.5
    
    def log_prediction(self, code, state, action, confidence, price):
        """
        예측 결과 로깅 (학습 데이터로 활용)
        
        Args:
            code (str): 종목코드
            state (numpy.ndarray): 상태 벡터
            action (int): 예측된 행동
            confidence (float): 확신도
            price (float): 현재가
        """
        try:
            # 로그 디렉토리 생성
            log_dir = os.path.join(self.model_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # 오늘 날짜 로그 파일
            today = datetime.datetime.now().strftime('%Y%m%d')
            log_file = os.path.join(log_dir, f'prediction_log_{today}.jsonl')
            
            # 로그 데이터 구성
            log_data = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'code': code,
                'state': state.tolist() if isinstance(state, np.ndarray) else state,
                'action': int(action),
                'confidence': float(confidence),
                'price': float(price)
            }
            
            # 파일에 추가
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data) + '\n')
                
        except Exception as e:
            send_message(f"예측 로깅 실패: {e}", config.notification.discord_webhook_url) 