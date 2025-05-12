"""
한국 주식 자동매매 - 기술적 지표 생성 모듈
강화학습 모델을 위한 기술적 지표 계산 기능
"""

import pandas as pd
import numpy as np
import talib
import logging
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger("stock_auto")

class TechnicalIndicatorGenerator:
    """기술적 지표 생성 클래스"""
    
    def __init__(self):
        """초기화"""
        # 기본 기술적 지표 컬럼 정의
        self.all_indicators = [
            'sma5', 'sma10', 'sma20', 'sma60', 'sma120',
            'ema5', 'ema10', 'ema20',
            'bb_upper', 'bb_middle', 'bb_lower',
            'rsi', 'adx', 'macd', 'macd_signal', 'macd_hist',
            'daily_return', 'volatility',
            'volume_ma5', 'volume_ma20', 'volume_ratio', 'obv'
        ]
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표 추가
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 기술적 지표가 추가된 데이터프레임
        """
        # 원본 데이터 복사
        df_processed = df.copy()
        
        # 필요한 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df_processed.columns]
        
        if missing_columns:
            logger.warning(f"누락된 컬럼: {missing_columns}")
            return df_processed
        
        try:
            # 이동평균선 추가
            df_processed = self.add_moving_averages(df_processed)
            
            # 볼린저 밴드 추가
            df_processed = self.add_bollinger_bands(df_processed)
            
            # 모멘텀 지표 추가
            df_processed = self.add_momentum_indicators(df_processed)
            
            # 거래량 지표 추가
            df_processed = self.add_volume_indicators(df_processed)
            
            # 변동성 지표 추가
            df_processed = self.add_volatility_indicators(df_processed)
            
            # NaN 값 처리
            df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
            df_processed = df_processed.fillna(method='ffill')
            df_processed = df_processed.fillna(method='bfill')
            df_processed = df_processed.fillna(0)
            
            logger.info(f"기술적 지표 {len(df_processed.columns) - len(df.columns)}개 추가 완료")
            return df_processed
            
        except Exception as e:
            logger.error(f"기술적 지표 추가 실패: {e}", exc_info=True)
            return df_processed
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        이동평균선 추가
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 이동평균이 추가된 데이터프레임
        """
        df_result = df.copy()
        
        try:
            # 단순 이동평균 (SMA)
            df_result['sma5'] = talib.SMA(df_result['close'].values, timeperiod=5)
            df_result['sma10'] = talib.SMA(df_result['close'].values, timeperiod=10)
            df_result['sma20'] = talib.SMA(df_result['close'].values, timeperiod=20)
            df_result['sma60'] = talib.SMA(df_result['close'].values, timeperiod=60)
            df_result['sma120'] = talib.SMA(df_result['close'].values, timeperiod=120)
            
            # 지수이동평균 (EMA)
            df_result['ema5'] = talib.EMA(df_result['close'].values, timeperiod=5)
            df_result['ema10'] = talib.EMA(df_result['close'].values, timeperiod=10)
            df_result['ema20'] = talib.EMA(df_result['close'].values, timeperiod=20)
            
            return df_result
            
        except Exception as e:
            logger.error(f"이동평균 추가 실패: {e}", exc_info=True)
            return df
    
    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        볼린저 밴드 추가
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 볼린저 밴드가 추가된 데이터프레임
        """
        df_result = df.copy()
        
        try:
            # 볼린저 밴드 (Bollinger Bands)
            upperband, middleband, lowerband = talib.BBANDS(
                df_result['close'].values, 
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            df_result['bb_upper'] = upperband
            df_result['bb_middle'] = middleband
            df_result['bb_lower'] = lowerband
            
            return df_result
            
        except Exception as e:
            logger.error(f"볼린저 밴드 추가 실패: {e}", exc_info=True)
            return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모멘텀 지표 추가 (RSI, MACD, ADX 등)
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 모멘텀 지표가 추가된 데이터프레임
        """
        df_result = df.copy()
        
        try:
            # 상대강도지수 (RSI)
            df_result['rsi'] = talib.RSI(df_result['close'].values, timeperiod=14)
            
            # 평균 방향성 지수 (ADX)
            df_result['adx'] = talib.ADX(
                df_result['high'].values,
                df_result['low'].values,
                df_result['close'].values,
                timeperiod=14
            )
            
            # MACD (Moving Average Convergence Divergence)
            macd, macd_signal, macd_hist = talib.MACD(
                df_result['close'].values,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            df_result['macd'] = macd
            df_result['macd_signal'] = macd_signal
            df_result['macd_hist'] = macd_hist
            
            # 일중 가격 변동률
            df_result['daily_return'] = df_result['close'].pct_change()
            
            return df_result
            
        except Exception as e:
            logger.error(f"모멘텀 지표 추가 실패: {e}", exc_info=True)
            return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        거래량 지표 추가
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 거래량 지표가 추가된 데이터프레임
        """
        df_result = df.copy()
        
        try:
            # 거래량 이동평균
            df_result['volume_ma5'] = talib.SMA(df_result['volume'].values, timeperiod=5)
            df_result['volume_ma20'] = talib.SMA(df_result['volume'].values, timeperiod=20)
            
            # 거래량 비율 (현재 거래량 / 5일 평균 거래량)
            df_result['volume_ratio'] = df_result['volume'] / df_result['volume_ma5']
            
            # OBV (On Balance Volume)
            df_result['obv'] = talib.OBV(df_result['close'].values, df_result['volume'].values)
            
            return df_result
            
        except Exception as e:
            logger.error(f"거래량 지표 추가 실패: {e}", exc_info=True)
            return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        변동성 지표 추가
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 변동성 지표가 추가된 데이터프레임
        """
        df_result = df.copy()
        
        try:
            # 변동성 (Volatility) - 20일 표준편차
            df_result['volatility'] = df_result['daily_return'].rolling(window=20).std()
            
            # ATR (Average True Range)
            df_result['atr'] = talib.ATR(
                df_result['high'].values,
                df_result['low'].values,
                df_result['close'].values,
                timeperiod=14
            )
            
            return df_result
            
        except Exception as e:
            logger.error(f"변동성 지표 추가 실패: {e}", exc_info=True)
            return df
    
    def add_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        사용자 정의 지표 추가
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 사용자 정의 지표가 추가된 데이터프레임
        """
        df_result = df.copy()
        
        try:
            # 가격 모멘텀: 현재가 / 20일 이동평균
            if 'close' in df_result.columns and 'sma20' in df_result.columns:
                df_result['price_momentum'] = df_result['close'] / df_result['sma20']
            
            # 이동평균 교차: 5일 이동평균 - 20일 이동평균
            if 'sma5' in df_result.columns and 'sma20' in df_result.columns:
                df_result['ma_cross'] = df_result['sma5'] - df_result['sma20']
            
            # 볼린저 밴드 폭: (상단 - 하단) / 중간
            if all(col in df_result.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                df_result['bb_width'] = (df_result['bb_upper'] - df_result['bb_lower']) / df_result['bb_middle']
            
            # 실시간 API 데이터 관련 추가 지표 (있는 경우)
            if 'bid_ask_ratio' in df_result.columns:
                pass  # 이미 API 데이터에서 계산되어 있음
            
            if 'market_pressure' in df_result.columns:
                pass  # 이미 API 데이터에서 계산되어 있음
            
            return df_result
            
        except Exception as e:
            logger.error(f"사용자 정의 지표 추가 실패: {e}", exc_info=True)
            return df 