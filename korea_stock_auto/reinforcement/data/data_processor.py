"""
한국 주식 자동매매 - 데이터 전처리 모듈
강화학습 모델을 위한 데이터 준비 및 전처리 기능
"""

import os
import numpy as np
import pandas as pd
import talib
import pickle
from sklearn.preprocessing import MinMaxScaler
from korea_stock_auto.utils import send_message

class DataProcessor:
    """강화학습을 위한 데이터 전처리 클래스"""
    
    def __init__(self, lookback=20, feature_columns=None, scaler_path=None):
        """
        데이터 프로세서 초기화
        
        Args:
            lookback (int): 학습에 사용할 과거 데이터 길이
            feature_columns (list): 사용할 특성 컬럼 리스트
            scaler_path (str): 스케일러 저장 경로
        """
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_path = scaler_path or 'models/feature_scaler.pkl'
        
        # 기본 특성 컬럼 정의
        self.feature_columns = feature_columns or [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma20', 'ma60', 'rsi', 'macd', 'macd_signal', 'bbands_upper', 'bbands_middle', 'bbands_lower'
        ]
        
        # 스케일러 불러오기 시도
        self._load_scaler()
    
    def _load_scaler(self):
        """저장된 스케일러 불러오기"""
        try:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                send_message("스케일러 로드 성공")
        except Exception as e:
            send_message(f"스케일러 로드 실패: {e}")
    
    def _save_scaler(self):
        """스케일러 저장"""
        try:
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            send_message(f"스케일러 저장 실패: {e}")
    
    def add_technical_indicators(self, df):
        """
        기술적 지표 추가
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임 (OHLCV 데이터)
            
        Returns:
            pd.DataFrame: 기술적 지표가 추가된 데이터프레임
        """
        # 데이터 복사
        df_processed = df.copy()
        
        try:
            # 이동평균선
            df_processed['ma5'] = talib.SMA(df_processed['close'], timeperiod=5)
            df_processed['ma20'] = talib.SMA(df_processed['close'], timeperiod=20)
            df_processed['ma60'] = talib.SMA(df_processed['close'], timeperiod=60)
            
            # RSI
            df_processed['rsi'] = talib.RSI(df_processed['close'], timeperiod=14)
            
            # MACD
            macd, macd_signal, _ = talib.MACD(df_processed['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df_processed['macd'] = macd
            df_processed['macd_signal'] = macd_signal
            
            # 볼린저 밴드
            upper, middle, lower = talib.BBANDS(df_processed['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df_processed['bbands_upper'] = upper
            df_processed['bbands_middle'] = middle
            df_processed['bbands_lower'] = lower
            
            # 변동성 지표 (ATR)
            df_processed['atr'] = talib.ATR(df_processed['high'], df_processed['low'], df_processed['close'], timeperiod=14)
            
            # 모멘텀 지표 (ROC)
            df_processed['roc'] = talib.ROC(df_processed['close'], timeperiod=10)
            
            # 볼륨 관련 지표
            df_processed['volume_ma5'] = talib.SMA(df_processed['volume'], timeperiod=5)
            df_processed['volume_ratio'] = df_processed['volume'] / df_processed['volume_ma5']
            
            # 가격과 이평선 비율
            df_processed['price_ma5_ratio'] = (df_processed['close'] - df_processed['ma5']) / df_processed['ma5']
            df_processed['price_ma20_ratio'] = (df_processed['close'] - df_processed['ma20']) / df_processed['ma20']
            
            # 이평선 기울기
            df_processed['ma5_slope'] = talib.LINEARREG_SLOPE(df_processed['ma5'], timeperiod=5)
            df_processed['ma20_slope'] = talib.LINEARREG_SLOPE(df_processed['ma20'], timeperiod=5)
            
            # 결측값 처리
            df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
            df_processed = df_processed.dropna()
            
            return df_processed
            
        except Exception as e:
            send_message(f"기술적 지표 추가 실패: {e}")
            return df
    
    def normalize_data(self, df, fit=False):
        """
        데이터 정규화
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            fit (bool): 스케일러 학습 여부
            
        Returns:
            pd.DataFrame: 정규화된 데이터프레임
        """
        try:
            # 정규화할 컬럼 선택
            columns_to_scale = [col for col in self.feature_columns if col in df.columns]
            
            # 스케일러 적용
            if fit:
                scaled_data = self.scaler.fit_transform(df[columns_to_scale])
                # 학습된 스케일러 저장
                self._save_scaler()
            else:
                scaled_data = self.scaler.transform(df[columns_to_scale])
            
            # 정규화된 데이터프레임 생성
            scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=columns_to_scale)
            
            # 정규화되지 않은 컬럼 복원
            for col in df.columns:
                if col not in columns_to_scale:
                    scaled_df[col] = df[col]
            
            return scaled_df
            
        except Exception as e:
            send_message(f"데이터 정규화 실패: {e}")
            return df
    
    def create_sequences(self, df, target_column='close'):
        """
        시계열 시퀀스 데이터 생성
        
        Args:
            df (pd.DataFrame): 처리된 데이터프레임
            target_column (str): 타겟 컬럼 이름
            
        Returns:
            tuple: (X, y) 시퀀스 데이터와 타겟값
        """
        try:
            # 특성 컬럼만 선택
            feature_data = df[self.feature_columns].values
            
            # 시퀀스 및 타겟 데이터 초기화
            X, y = [], []
            
            # 시퀀스 생성
            for i in range(len(df) - self.lookback):
                # 시퀀스 데이터
                seq = feature_data[i:i+self.lookback]
                X.append(seq)
                
                # 타겟 데이터 (다음 종가의 방향)
                current_close = df[target_column].values[i+self.lookback-1]
                next_close = df[target_column].values[i+self.lookback]
                
                if next_close > current_close:
                    target = 1  # 상승 (매수)
                elif next_close < current_close:
                    target = 2  # 하락 (매도)
                else:
                    target = 0  # 유지
                
                y.append(target)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            send_message(f"시퀀스 생성 실패: {e}")
            return np.array([]), np.array([])
    
    def prepare_state_vector(self, current_data, historical_data=None):
        """
        현재 상태 벡터 생성
        
        Args:
            current_data (dict): 현재 시장 데이터
            historical_data (pd.DataFrame): 과거 시장 데이터
            
        Returns:
            numpy.ndarray: 현재 상태 벡터
        """
        try:
            # 과거 데이터가 있는 경우 기술적 지표 추가
            if historical_data is not None and len(historical_data) > 0:
                historical_data = self.add_technical_indicators(historical_data)
                historical_data = self.normalize_data(historical_data)
                
                # 최신 데이터 사용
                if len(historical_data) >= self.lookback:
                    # 최근 lookback 일치 데이터 추출
                    recent_data = historical_data.iloc[-self.lookback:][self.feature_columns].values
                    return recent_data.flatten()  # 1차원 벡터로 변환
            
            # 과거 데이터가 없는 경우 현재 데이터만 사용하여 간소화된 상태 벡터 생성
            state_vector = []
            
            # 현재 가격
            current_price = float(current_data.get('close', 0))
            
            # 가격 정보 (정규화 없이 기본 비율만 계산)
            ma5 = float(current_data.get('ma5', current_price))
            ma20 = float(current_data.get('ma20', current_price))
            
            # 이평선 대비 가격 비율
            price_ma5_ratio = (current_price - ma5) / ma5 if ma5 > 0 else 0
            price_ma20_ratio = (current_price - ma20) / ma20 if ma20 > 0 else 0
            
            # 기본 상태 벡터
            state_vector = [
                price_ma5_ratio,
                price_ma20_ratio,
                float(current_data.get('rsi', 50)) / 100,  # RSI 0-1 정규화
                float(current_data.get('volume_ratio', 1))
            ]
            
            return np.array(state_vector)
            
        except Exception as e:
            send_message(f"상태 벡터 생성 실패: {e}")
            # 기본 상태 벡터 반환
            return np.array([0.0, 0.0, 0.5, 1.0]) 