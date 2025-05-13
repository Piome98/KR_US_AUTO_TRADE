"""
한국 주식 자동매매 - 기술적 분석 모듈
주가 데이터에 대한 기술적 분석 기능 제공
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.data.database import TechnicalDataManager
from korea_stock_auto.reinforcement.rl_data.technical_indicators import TechnicalIndicatorGenerator
from korea_stock_auto.utils.utils import send_message

logger = logging.getLogger("stock_auto")

class TechnicalAnalyzer(TechnicalIndicatorGenerator):
    """
    주식 기술적 분석 클래스
    TechnicalIndicatorGenerator를 상속받아 기술적 지표 생성 기능을 재사용합니다.
    """
    
    def __init__(self, api_client: KoreaInvestmentApiClient, db_path: str = "stock_data.db"):
        """
        기술적 분석기 초기화
        
        Args:
            api_client: API 클라이언트 인스턴스
            db_path: 데이터베이스 파일 경로
        """
        super().__init__(db_path=db_path)
        self.api = api_client
        self.tech_data_manager = TechnicalDataManager(db_path)
    
    def get_moving_average(self, code: str, period: int = 20) -> Optional[float]:
        """
        지정된 기간의 이동평균선 계산 함수
        
        Args:
            code: 종목 코드
            period: 이동평균 기간
            
        Returns:
            float or None: 이동평균 값 또는 데이터가 없는 경우 None
        """
        # 먼저 데이터베이스에서 지표 조회
        indicator_name = f"sma_{period}" if period in [5, 20, 60, 120] else "sma_20"  # 가장 가까운 값 사용
        latest_indicators = self.tech_data_manager.get_latest_indicators(code)
        
        if latest_indicators and indicator_name in latest_indicators:
            return latest_indicators[indicator_name]
        
        # 데이터베이스에 없으면 API에서 가져와 계산
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
        # 먼저 데이터베이스에서 지표 조회
        latest_indicators = self.tech_data_manager.get_latest_indicators(code)
        
        if latest_indicators and 'rsi' in latest_indicators:
            return latest_indicators['rsi']
            
        # 데이터베이스에 없으면 API에서 가져와 계산
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
        # 먼저 데이터베이스에서 지표 조회
        latest_indicators = self.tech_data_manager.get_latest_indicators(code)
        
        if latest_indicators and 'bollinger_upper' in latest_indicators:
            return {
                "upper": latest_indicators['bollinger_upper'],
                "middle": latest_indicators['bollinger_middle'],
                "lower": latest_indicators['bollinger_lower'],
                "bandwidth": (latest_indicators['bollinger_upper'] - latest_indicators['bollinger_lower']) / latest_indicators['bollinger_middle'] if latest_indicators['bollinger_middle'] != 0 else 0
            }
            
        # 데이터베이스에 없으면 API에서 가져와 계산
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
        # 먼저 데이터베이스에서 지표 조회
        latest_indicators = self.tech_data_manager.get_latest_indicators(code)
        
        if latest_indicators and 'atr' in latest_indicators:  # ATR로 변동성 대체
            return latest_indicators['atr']
            
        # 데이터베이스에 없으면 API에서 가져와 계산
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
        골든 크로스 확인 함수
        
        Args:
            code: 종목 코드
            short_period: 단기 이동평균 기간
            long_period: 장기 이동평균 기간
            
        Returns:
            bool or None: 골든 크로스 여부 또는 데이터가 없는 경우 None
        """
        try:
            # 필요한 이동평균 계산
            formatted_code = code.zfill(12)
            daily_data = self.api.get_daily_data(formatted_code)
            
            if not daily_data or len(daily_data) < long_period + 2:  # 최소 2일치 더 필요
                logger.warning(f"{code} 일봉 데이터가 부족하여 골든 크로스 확인 불가")
                return None
            
            # 전날과 오늘의 종가 데이터 추출
            closes_yesterday = [float(candle["stck_clpr"]) for candle in daily_data[1:]]
            closes_today = [float(candle["stck_clpr"]) for candle in daily_data[:-1]]
            
            # 이동평균 계산
            short_ma_yesterday = np.mean(closes_yesterday[:short_period])
            long_ma_yesterday = np.mean(closes_yesterday[:long_period])
            
            short_ma_today = np.mean(closes_today[:short_period])
            long_ma_today = np.mean(closes_today[:long_period])
            
            # 전날: 단기 < 장기, 오늘: 단기 > 장기 => 골든 크로스
            return (short_ma_yesterday < long_ma_yesterday) and (short_ma_today > long_ma_today)
            
        except Exception as e:
            logger.error(f"{code} 골든 크로스 확인 실패: {e}")
            return None 