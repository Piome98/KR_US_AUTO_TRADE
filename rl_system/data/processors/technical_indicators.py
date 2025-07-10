"""
기술적 지표 계산 모듈
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """기술적 지표 계산 클래스"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """
        단순이동평균(SMA) 계산
        
        Args:
            data: 가격 데이터
            window: 이동평균 기간
            
        Returns:
            pd.Series: SMA 값
        """
        return data.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """
        지수이동평균(EMA) 계산
        
        Args:
            data: 가격 데이터
            window: 이동평균 기간
            
        Returns:
            pd.Series: EMA 값
        """
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        상대강도지수(RSI) 계산
        
        Args:
            data: 가격 데이터
            window: RSI 기간
            
        Returns:
            pd.Series: RSI 값
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        MACD 계산
        
        Args:
            data: 가격 데이터
            fast_period: 빠른 EMA 기간
            slow_period: 느린 EMA 기간
            signal_period: 신호선 EMA 기간
            
        Returns:
            Dict[str, pd.Series]: MACD, 신호선, 히스토그램
        """
        fast_ema = TechnicalIndicators.calculate_ema(data, fast_period)
        slow_ema = TechnicalIndicators.calculate_ema(data, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(
        data: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        볼린저 밴드 계산
        
        Args:
            data: 가격 데이터
            window: 이동평균 기간
            num_std: 표준편차 배수
            
        Returns:
            Dict[str, pd.Series]: 상단밴드, 중간밴드, 하단밴드
        """
        middle_band = TechnicalIndicators.calculate_sma(data, window)
        std = data.rolling(window=window).std()
        
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return {
            'bollinger_upper': upper_band,
            'bollinger_middle': middle_band,
            'bollinger_lower': lower_band
        }
    
    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """
        스토캐스틱 계산
        
        Args:
            high: 고가 데이터
            low: 저가 데이터
            close: 종가 데이터
            k_period: %K 기간
            d_period: %D 기간
            
        Returns:
            Dict[str, pd.Series]: %K, %D
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    @staticmethod
    def calculate_williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """
        윌리엄스 %R 계산
        
        Args:
            high: 고가 데이터
            low: 저가 데이터
            close: 종가 데이터
            window: 계산 기간
            
        Returns:
            pd.Series: 윌리엄스 %R 값
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """
        평균실체범위(ATR) 계산
        
        Args:
            high: 고가 데이터
            low: 저가 데이터
            close: 종가 데이터
            window: ATR 기간
            
        Returns:
            pd.Series: ATR 값
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def calculate_momentum(data: pd.Series, window: int = 10) -> pd.Series:
        """
        모멘텀 계산
        
        Args:
            data: 가격 데이터
            window: 모멘텀 기간
            
        Returns:
            pd.Series: 모멘텀 값
        """
        return data - data.shift(window)
    
    @staticmethod
    def calculate_roc(data: pd.Series, window: int = 10) -> pd.Series:
        """
        변화율(ROC) 계산
        
        Args:
            data: 가격 데이터
            window: ROC 기간
            
        Returns:
            pd.Series: ROC 값
        """
        return ((data - data.shift(window)) / data.shift(window)) * 100
    
    @staticmethod
    def calculate_cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        상품채널지수(CCI) 계산
        
        Args:
            high: 고가 데이터
            low: 저가 데이터
            close: 종가 데이터
            window: CCI 기간
            
        Returns:
            pd.Series: CCI 값
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mean_dev = typical_price.rolling(window=window).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_dev)
        
        return cci
    
    @staticmethod
    def calculate_volume_indicators(
        volume: pd.Series,
        price: pd.Series,
        window: int = 20
    ) -> Dict[str, pd.Series]:
        """
        거래량 지표들 계산
        
        Args:
            volume: 거래량 데이터
            price: 가격 데이터
            window: 계산 기간
            
        Returns:
            Dict[str, pd.Series]: 거래량 지표들
        """
        volume_sma = TechnicalIndicators.calculate_sma(volume, window)
        volume_ratio = volume / volume_sma
        
        # 거래량 가격 추세 (VPT)
        vpt = ((price.pct_change() * volume).cumsum())
        
        # 온밸런스볼륨 (OBV)
        obv = np.where(price > price.shift(1), volume, 
                      np.where(price < price.shift(1), -volume, 0)).cumsum()
        
        return {
            'volume_sma': volume_sma,
            'volume_ratio': volume_ratio,
            'vpt': vpt,
            'obv': pd.Series(obv, index=volume.index)
        }
    
    @staticmethod
    def calculate_price_ratios(
        data: pd.DataFrame,
        base_column: str = 'close'
    ) -> pd.DataFrame:
        """
        가격 비율들 계산
        
        Args:
            data: OHLCV 데이터
            base_column: 기준 컬럼
            
        Returns:
            pd.DataFrame: 가격 비율 데이터
        """
        result = data.copy()
        base_price = data[base_column]
        
        # 고가/저가 비율
        if 'high' in data.columns and 'low' in data.columns:
            result['high_low_ratio'] = data['high'] / data['low']
            result['high_close_ratio'] = data['high'] / base_price
            result['low_close_ratio'] = data['low'] / base_price
        
        # 시가/종가 비율
        if 'open' in data.columns:
            result['open_close_ratio'] = data['open'] / base_price
        
        # 전일 대비 비율
        result['price_change_ratio'] = base_price / base_price.shift(1)
        
        return result
    
    @staticmethod
    def add_all_indicators(
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        모든 기술적 지표 추가
        
        Args:
            data: OHLCV 데이터
            indicators: 추가할 지표 목록 (None이면 모든 지표)
            
        Returns:
            pd.DataFrame: 지표가 추가된 데이터
        """
        try:
            result = data.copy()
            
            # 필수 컬럼 확인
            required_columns = ['close']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Required columns missing: {required_columns}")
                return result
            
            close = data['close']
            high = data.get('high', close)
            low = data.get('low', close)
            open_price = data.get('open', close)
            volume = data.get('volume', pd.Series(1, index=data.index))
            
            # 기본 지표들
            basic_indicators = {
                'sma_5': lambda: TechnicalIndicators.calculate_sma(close, 5),
                'sma_10': lambda: TechnicalIndicators.calculate_sma(close, 10),
                'sma_20': lambda: TechnicalIndicators.calculate_sma(close, 20),
                'sma_60': lambda: TechnicalIndicators.calculate_sma(close, 60),
                'ema_12': lambda: TechnicalIndicators.calculate_ema(close, 12),
                'ema_26': lambda: TechnicalIndicators.calculate_ema(close, 26),
                'rsi_14': lambda: TechnicalIndicators.calculate_rsi(close, 14),
                'rsi_30': lambda: TechnicalIndicators.calculate_rsi(close, 30),
                'williams_r': lambda: TechnicalIndicators.calculate_williams_r(high, low, close),
                'atr_14': lambda: TechnicalIndicators.calculate_atr(high, low, close, 14),
                'momentum_10': lambda: TechnicalIndicators.calculate_momentum(close, 10),
                'roc_10': lambda: TechnicalIndicators.calculate_roc(close, 10),
                'cci_20': lambda: TechnicalIndicators.calculate_cci(high, low, close, 20)
            }
            
            # MACD 지표
            macd_data = TechnicalIndicators.calculate_macd(close)
            basic_indicators.update({
                'macd': lambda: macd_data['macd'],
                'macd_signal': lambda: macd_data['macd_signal'],
                'macd_histogram': lambda: macd_data['macd_histogram']
            })
            
            # 볼린저 밴드
            bb_data = TechnicalIndicators.calculate_bollinger_bands(close)
            basic_indicators.update({
                'bollinger_upper': lambda: bb_data['bollinger_upper'],
                'bollinger_middle': lambda: bb_data['bollinger_middle'],
                'bollinger_lower': lambda: bb_data['bollinger_lower']
            })
            
            # 스토캐스틱
            stoch_data = TechnicalIndicators.calculate_stochastic(high, low, close)
            basic_indicators.update({
                'stoch_k': lambda: stoch_data['stoch_k'],
                'stoch_d': lambda: stoch_data['stoch_d']
            })
            
            # 거래량 지표
            volume_data = TechnicalIndicators.calculate_volume_indicators(volume, close)
            basic_indicators.update({
                'volume_sma_20': lambda: volume_data['volume_sma'],
                'volume_ratio': lambda: volume_data['volume_ratio'],
                'vpt': lambda: volume_data['vpt'],
                'obv': lambda: volume_data['obv']
            })
            
            # 지표 선택
            if indicators is None:
                indicators = list(basic_indicators.keys())
            
            # 지표 계산 및 추가
            for indicator in indicators:
                if indicator in basic_indicators:
                    try:
                        result[indicator] = basic_indicators[indicator]()
                        logger.debug(f"Added indicator: {indicator}")
                    except Exception as e:
                        logger.error(f"Failed to calculate {indicator}: {e}")
                        result[indicator] = np.nan
                else:
                    logger.warning(f"Unknown indicator: {indicator}")
            
            # 가격 비율 추가
            ratio_data = TechnicalIndicators.calculate_price_ratios(data)
            for col in ratio_data.columns:
                if col not in result.columns:
                    result[col] = ratio_data[col]
            
            # NaN 값 처리
            result = result.fillna(method='ffill').fillna(0)
            
            logger.info(f"Added {len(indicators)} technical indicators")
            return result
            
        except Exception as e:
            logger.error(f"Failed to add technical indicators: {e}")
            return data
    
    @staticmethod
    def normalize_indicators(
        data: pd.DataFrame,
        indicator_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        기술적 지표 정규화
        
        Args:
            data: 지표가 포함된 데이터
            indicator_columns: 정규화할 지표 컬럼 목록
            
        Returns:
            pd.DataFrame: 정규화된 데이터
        """
        try:
            result = data.copy()
            
            if indicator_columns is None:
                # 기술적 지표로 추정되는 컬럼들 자동 선택
                exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'date']
                indicator_columns = [col for col in data.columns if col not in exclude_columns]
            
            for col in indicator_columns:
                if col in result.columns:
                    try:
                        # Z-score 정규화
                        mean_val = result[col].mean()
                        std_val = result[col].std()
                        
                        if std_val > 0:
                            result[f"{col}_norm"] = (result[col] - mean_val) / std_val
                        else:
                            result[f"{col}_norm"] = 0
                            
                        logger.debug(f"Normalized indicator: {col}")
                    except Exception as e:
                        logger.error(f"Failed to normalize {col}: {e}")
                        result[f"{col}_norm"] = 0
            
            logger.info(f"Normalized {len(indicator_columns)} indicators")
            return result
            
        except Exception as e:
            logger.error(f"Failed to normalize indicators: {e}")
            return data