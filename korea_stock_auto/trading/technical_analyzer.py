"""
한국 주식 자동매매 - 기술적 분석 모듈
주가 데이터에 대한 기술적 분석 기능 제공
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.utils.utils import send_message

logger = logging.getLogger("stock_auto")

class TechnicalAnalyzer:
    """주식 기술적 분석 클래스"""
    
    def __init__(self, api_client: KoreaInvestmentApiClient):
        """
        기술적 분석기 초기화
        
        Args:
            api_client: API 클라이언트 인스턴스
        """
        self.api = api_client
    
    def calculate_macd(self, prices: List[float], short_period: int = 12, long_period: int = 26, signal_period: int = 9) -> Dict[str, float]:
        """
        MACD(Moving Average Convergence Divergence) 계산 함수
        
        Args:
            prices: 가격 데이터 리스트
            short_period: 단기 이동평균 기간
            long_period: 장기 이동평균 기간
            signal_period: 시그널 라인 기간
            
        Returns:
            dict: MACD 값, 시그널 라인 값, 히스토그램 값을 포함하는 딕셔너리
        """
        if len(prices) < long_period:
            return {
                "macd": 0,
                "signal": 0,
                "histogram": 0
            }
            
        # 단기 지수 이동평균
        short_ema = self._calculate_ema(prices, short_period)
        
        # 장기 지수 이동평균
        long_ema = self._calculate_ema(prices, long_period)
        
        # MACD 라인
        macd_line = short_ema - long_ema
        
        # 시그널 라인 (MACD의 지수 이동평균)
        signal_line = self._calculate_ema([macd_line], signal_period)
        
        # 히스토그램 (MACD - 시그널 라인)
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """
        지수 이동평균(EMA) 계산 함수
        
        Args:
            prices: 가격 데이터 리스트
            period: 이동평균 기간
            
        Returns:
            float: 지수 이동평균 값
        """
        if len(prices) < period:
            return np.mean(prices)
            
        # 단순 이동평균으로 초기화
        sma = np.mean(prices[:period])
        
        # 승수 계산 (일반적으로 2/(기간+1) 사용)
        multiplier = 2 / (period + 1)
        
        # EMA 계산
        ema = sma
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
            
        return ema
    
    def get_moving_average(self, code: str, period: int = 20) -> Optional[float]:
        """
        지정된 기간의 이동평균선 계산 함수
        
        Args:
            code: 종목 코드
            period: 이동평균 기간
            
        Returns:
            float or None: 이동평균 값 또는 데이터가 없는 경우 None
        """
        # 코드 포맷: 12자리로 변환
        formatted_code = code.zfill(12)
        daily_data = self.api.get_daily_data(formatted_code)
        
        if not daily_data or len(daily_data) < period:
            logger.warning(f"{code} 일봉 데이터가 부족하여 이동평균 계산 불가")
            return None
            
        try:
            # 종가 데이터 추출
            closes = [float(candle["stck_clpr"]) for candle in daily_data[:period]]
            
            # 단순 이동평균 계산
            ma = np.mean(closes)
            return ma
        except Exception as e:
            logger.error(f"{code} 이동평균 계산 실패: {e}")
            return None
    
    def calculate_rsi(self, code: str, period: int = 14) -> Optional[float]:
        """
        RSI(Relative Strength Index) 계산 함수
        
        Args:
            code: 종목 코드
            period: RSI 계산 기간
            
        Returns:
            float or None: RSI 값 또는 데이터가 없는 경우 None
        """
        formatted_code = code.zfill(12)
        daily_data = self.api.get_daily_data(formatted_code)
        
        if not daily_data or len(daily_data) <= period:
            logger.warning(f"{code} 일봉 데이터가 부족하여 RSI 계산 불가")
            return None
            
        try:
            # 종가 데이터 추출
            closes = [float(candle["stck_clpr"]) for candle in daily_data[:period+1]]
            closes.reverse()  # 최신 데이터가 앞에 오도록 순서 변경
            
            # 가격 변화 계산
            deltas = np.diff(closes)
            
            # 상승/하락 분리
            gains = np.clip(deltas, 0, np.inf)
            losses = np.clip(-deltas, 0, np.inf)
            
            # 평균 상승/하락 계산
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:  # 하락이 없는 경우
                return 100.0
                
            # RS(Relative Strength) 계산
            rs = avg_gain / avg_loss
            
            # RSI 계산
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception as e:
            logger.error(f"{code} RSI 계산 실패: {e}")
            return None
    
    def calculate_bollinger_bands(self, code: str, period: int = 20, std_dev: float = 2.0) -> Optional[Dict[str, float]]:
        """
        볼린저 밴드 계산 함수
        
        Args:
            code: 종목 코드
            period: 이동평균 기간
            std_dev: 표준편차 승수
            
        Returns:
            dict or None: 볼린저 밴드 값 또는 데이터가 없는 경우 None
        """
        formatted_code = code.zfill(12)
        daily_data = self.api.get_daily_data(formatted_code)
        
        if not daily_data or len(daily_data) < period:
            logger.warning(f"{code} 일봉 데이터가 부족하여 볼린저 밴드 계산 불가")
            return None
            
        try:
            # 종가 데이터 추출
            closes = [float(candle["stck_clpr"]) for candle in daily_data[:period]]
            
            # 이동평균 계산
            middle_band = np.mean(closes)
            
            # 표준편차 계산
            std = np.std(closes)
            
            # 상단/하단 밴드 계산
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)
            
            return {
                "upper": upper_band,
                "middle": middle_band,
                "lower": lower_band,
                "bandwidth": (upper_band - lower_band) / middle_band
            }
            
        except Exception as e:
            logger.error(f"{code} 볼린저 밴드 계산 실패: {e}")
            return None
    
    def calculate_volatility(self, code: str, period: int = 20) -> Optional[float]:
        """
        변동성 계산 함수
        
        Args:
            code: 종목 코드
            period: 변동성 계산 기간
            
        Returns:
            float or None: 변동성 값 또는 데이터가 없는 경우 None
        """
        formatted_code = code.zfill(12)
        daily_data = self.api.get_daily_data(formatted_code)
        
        if not daily_data or len(daily_data) < period:
            logger.warning(f"{code} 일봉 데이터가 부족하여 변동성 계산 불가")
            return None
            
        try:
            # 종가 데이터 추출
            closes = np.array([float(candle["stck_clpr"]) for candle in daily_data[:period]])
            
            # 일간 수익률 계산
            returns = np.diff(closes) / closes[:-1]
            
            # 변동성 (수익률의 표준편차)
            volatility = np.std(returns) * np.sqrt(252)  # 연율화
            
            return volatility
            
        except Exception as e:
            logger.error(f"{code} 변동성 계산 실패: {e}")
            return None
    
    def is_golden_cross(self, code: str, short_period: int = 5, long_period: int = 20) -> Optional[bool]:
        """
        골든 크로스 발생 여부 확인
        
        Args:
            code: 종목 코드
            short_period: 단기 이동평균 기간
            long_period: 장기 이동평균 기간
            
        Returns:
            bool or None: 골든 크로스 발생 여부 또는 데이터가 없는 경우 None
        """
        formatted_code = code.zfill(12)
        daily_data = self.api.get_daily_data(formatted_code)
        
        if not daily_data or len(daily_data) < long_period + 2:  # 이전 데이터까지 필요
            logger.warning(f"{code} 일봉 데이터가 부족하여 골든 크로스 확인 불가")
            return None
            
        try:
            # 종가 데이터 추출
            closes = [float(candle["stck_clpr"]) for candle in daily_data[:long_period+2]]
            
            # 현재 이동평균
            current_short_ma = np.mean(closes[:short_period])
            current_long_ma = np.mean(closes[:long_period])
            
            # 이전 이동평균
            prev_closes = closes[1:]  # 하루 전 데이터
            prev_short_ma = np.mean(prev_closes[:short_period])
            prev_long_ma = np.mean(prev_closes[:long_period])
            
            # 골든 크로스 확인 (단기 > 장기 교차)
            is_golden = (prev_short_ma <= prev_long_ma) and (current_short_ma > current_long_ma)
            
            return is_golden
            
        except Exception as e:
            logger.error(f"{code} 골든 크로스 확인 실패: {e}")
            return None 