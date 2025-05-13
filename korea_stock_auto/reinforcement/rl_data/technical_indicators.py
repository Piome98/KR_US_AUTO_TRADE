"""
한국 주식 자동매매 - 기술적 지표 생성 모듈 (리팩토링 버전)

강화학습 모델을 위한 기술적 지표 계산 기능
database.technical_data 모듈과 연동하여 기존 저장된 지표 활용 및 추가 지표 계산
"""

import pandas as pd
import numpy as np
import talib
import logging
from typing import Optional, List, Dict, Any, Union

from korea_stock_auto.data.database import TechnicalDataManager

logger = logging.getLogger("stock_auto")

class TechnicalIndicatorGenerator:
    """기술적 지표 생성 클래스"""
    
    def __init__(self, db_path: str = "stock_data.db"):
        """
        초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        # 기본 기술적 지표 컬럼 정의
        self.all_indicators = [
            'sma5', 'sma10', 'sma20', 'sma60', 'sma120',
            'ema5', 'ema10', 'ema20',
            'bb_upper', 'bb_middle', 'bb_lower',
            'rsi', 'adx', 'macd', 'macd_signal', 'macd_hist',
            'daily_return', 'volatility',
            'volume_ma5', 'volume_ma20', 'volume_ratio', 'obv'
        ]
        
        # 데이터베이스 연동을 위한 기술적 지표 매니저
        self.tech_manager = TechnicalDataManager(db_path)
    
    def get_indicators_from_database(self, code: str, days: int = 30) -> pd.DataFrame:
        """
        데이터베이스에서 기술적 지표 가져오기
        
        Args:
            code: 종목 코드
            days: 조회 기간(일)
            
        Returns:
            pd.DataFrame: 기술적 지표 데이터프레임
        """
        try:
            # 데이터베이스에서 저장된 기술적 지표 조회
            indicators_df = self.tech_manager.get_technical_indicators(code, days)
            
            if indicators_df.empty:
                logger.warning(f"{code} 기술적 지표 데이터가 데이터베이스에 없습니다.")
                return pd.DataFrame()
            
            logger.info(f"{code} 기술적 지표 데이터 조회 성공: {len(indicators_df)} 행")
            return indicators_df
            
        except Exception as e:
            logger.error(f"기술적 지표 데이터 조회 실패: {e}", exc_info=True)
            return pd.DataFrame()
    
    def add_all_indicators(self, df: pd.DataFrame, use_db_first: bool = True, code: str = None) -> pd.DataFrame:
        """
        모든 기술적 지표 추가
        
        Args:
            df: 원본 데이터프레임
            use_db_first: 데이터베이스 데이터 우선 활용 여부
            code: 종목 코드 (use_db_first=True인 경우 필요)
            
        Returns:
            pd.DataFrame: 기술적 지표가 추가된 데이터프레임
        """
        # 원본 데이터 복사
        df_processed = df.copy()
        
        try:
            # 데이터베이스 우선 활용 옵션이 켜져있고 종목 코드가 제공된 경우
            if use_db_first and code:
                # 데이터베이스에서 기술적 지표 조회
                db_indicators = self.get_indicators_from_database(code, len(df))
                
                # 데이터베이스에 지표가 있는 경우, 해당 데이터 활용
                if not db_indicators.empty:
                    # 날짜 기준 병합
                    if 'date' in df_processed.columns and 'date' in db_indicators.columns:
                        # 날짜 형식 통일
                        df_processed['date'] = pd.to_datetime(df_processed['date'])
                        db_indicators['date'] = pd.to_datetime(db_indicators['date'])
                        
                        # 데이터 병합
                        df_merged = pd.merge(
                            df_processed,
                            db_indicators.drop(['code'], axis=1, errors='ignore'),
                            on='date',
                            how='left'
                        )
                        
                        # 병합 후 NaN 값이 있는 경우에만 직접 계산
                        missing_indicators = [col for col in self.all_indicators 
                                            if col in db_indicators.columns and df_merged[col].isna().any()]
                        
                        if missing_indicators:
                            logger.info(f"일부 누락된 지표 직접 계산: {missing_indicators}")
                            # 누락된 행에 대해서만 지표 계산
                            df_with_missing = df_processed[df_processed['date'].isin(
                                df_merged[df_merged[missing_indicators[0]].isna()]['date']
                            )]
                            
                            if not df_with_missing.empty:
                                df_calculated = self._calculate_indicators(df_with_missing)
                                
                                # 계산된 값으로 누락된 부분 채우기
                                for date in df_calculated['date'].unique():
                                    mask = df_merged['date'] == date
                                    calc_mask = df_calculated['date'] == date
                                    
                                    for indicator in missing_indicators:
                                        if indicator in df_calculated.columns:
                                            df_merged.loc[mask, indicator] = df_calculated.loc[calc_mask, indicator].values[0]
                            
                            return df_merged
                        
                        logger.info(f"데이터베이스 기술적 지표 활용 완료")
                        return df_merged
            
            # 데이터베이스 활용 불가 또는 옵션 꺼짐 - 직접 계산
            return self._calculate_indicators(df_processed)
            
        except Exception as e:
            logger.error(f"기술적 지표 추가 실패: {e}", exc_info=True)
            return df_processed
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표 직접 계산
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 기술적 지표가 추가된 데이터프레임
        """
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
            
            logger.info(f"기술적 지표 직접 계산 완료: {len(df_processed.columns) - len(df.columns)}개 지표")
            return df_processed
            
        except Exception as e:
            logger.error(f"기술적 지표 계산 실패: {e}", exc_info=True)
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