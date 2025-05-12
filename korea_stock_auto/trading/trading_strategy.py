"""
한국 주식 자동매매 - 매매 전략 모듈
매매 전략 인터페이스 및 구현 클래스 정의
"""

import logging
import abc
from typing import Dict, List, Any, Optional, Tuple, Union

from korea_stock_auto.config import TRADE_CONFIG
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.trading.technical_analyzer import TechnicalAnalyzer

logger = logging.getLogger("stock_auto")

class TradingStrategy(abc.ABC):
    """매매 전략 인터페이스"""
    
    def __init__(self, api_client: KoreaInvestmentApiClient, analyzer: TechnicalAnalyzer):
        """
        매매 전략 초기화
        
        Args:
            api_client: API 클라이언트 인스턴스
            analyzer: 기술적 분석기 인스턴스
        """
        self.api = api_client
        self.analyzer = analyzer
        self.name = "기본 전략"
    
    @abc.abstractmethod
    def should_buy(self, code: str, current_price: float) -> bool:
        """
        매수 시그널 확인
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            
        Returns:
            bool: 매수 시그널 여부
        """
        pass
    
    @abc.abstractmethod
    def should_sell(self, code: str, current_price: float, entry_price: float) -> bool:
        """
        매도 시그널 확인
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            entry_price: 매수 가격
            
        Returns:
            bool: 매도 시그널 여부
        """
        pass
    
    def get_strategy_name(self) -> str:
        """
        전략 이름 반환
        
        Returns:
            str: 전략 이름
        """
        return self.name


class MACDStrategy(TradingStrategy):
    """MACD 기반 매매 전략"""
    
    def __init__(self, api_client: KoreaInvestmentApiClient, analyzer: TechnicalAnalyzer):
        """
        MACD 전략 초기화
        
        Args:
            api_client: API 클라이언트 인스턴스
            analyzer: 기술적 분석기 인스턴스
        """
        super().__init__(api_client, analyzer)
        self.name = "MACD 전략"
        self.macd_short = TRADE_CONFIG.get("macd_short", 12)
        self.macd_long = TRADE_CONFIG.get("macd_long", 26)
        self.macd_signal = TRADE_CONFIG.get("macd_signal", 9)
        self.profit_target_pct = TRADE_CONFIG.get("profit_target_pct", 5.0)
        self.stop_loss_pct = TRADE_CONFIG.get("stop_loss_pct", 3.0)
    
    def should_buy(self, code: str, current_price: float) -> bool:
        """
        MACD 매수 시그널 확인
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            
        Returns:
            bool: 매수 시그널 여부
        """
        try:
            # 일봉 데이터 조회
            daily_data = self.api.get_daily_data(code)
            if not daily_data or len(daily_data) < self.macd_long + self.macd_signal:
                logger.warning(f"{code} 일봉 데이터 부족으로 매수 시그널 확인 불가")
                return False
                
            # 종가 데이터 추출
            closes = [float(candle["stck_clpr"]) for candle in daily_data[:self.macd_long + self.macd_signal]]
            closes.reverse()  # 최신 데이터가 앞에 오도록 순서 변경
            
            # 현재 MACD 계산
            current_macd = self.analyzer.calculate_macd(
                closes, 
                short_period=self.macd_short, 
                long_period=self.macd_long, 
                signal_period=self.macd_signal
            )
            
            # 이전 MACD 계산 (1일 전)
            prev_closes = closes[1:]
            prev_macd = self.analyzer.calculate_macd(
                prev_closes, 
                short_period=self.macd_short, 
                long_period=self.macd_long, 
                signal_period=self.macd_signal
            )
            
            # MACD 골든 크로스 확인 (MACD 라인이 시그널 라인을 상향 돌파)
            is_golden_cross = (prev_macd["macd"] <= prev_macd["signal"]) and (current_macd["macd"] > current_macd["signal"])
            
            # 추가 조건: MACD 히스토그램이 양수로 전환
            is_histogram_positive = (prev_macd["histogram"] <= 0) and (current_macd["histogram"] > 0)
            
            # 매수 시그널
            if is_golden_cross or is_histogram_positive:
                logger.info(f"{code} MACD 매수 시그널 발생: 골든크로스={is_golden_cross}, 히스토그램전환={is_histogram_positive}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"{code} MACD 매수 시그널 확인 중 오류 발생: {e}")
            return False
    
    def should_sell(self, code: str, current_price: float, entry_price: float) -> bool:
        """
        MACD 매도 시그널 확인
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            entry_price: 매수 가격
            
        Returns:
            bool: 매도 시그널 여부
        """
        try:
            # 수익률 계산
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            
            # 익절 조건 확인
            if profit_pct >= self.profit_target_pct:
                logger.info(f"{code} 익절 조건 도달: 수익률 {profit_pct:.2f}% >= 목표 {self.profit_target_pct:.2f}%")
                return True
                
            # 손절 조건 확인
            if profit_pct <= -self.stop_loss_pct:
                logger.info(f"{code} 손절 조건 도달: 수익률 {profit_pct:.2f}% <= 손절선 -{self.stop_loss_pct:.2f}%")
                return True
                
            # 일봉 데이터 조회
            daily_data = self.api.get_daily_data(code)
            if not daily_data or len(daily_data) < self.macd_long + self.macd_signal:
                logger.warning(f"{code} 일봉 데이터 부족으로 매도 시그널 확인 불가")
                return False
                
            # 종가 데이터 추출
            closes = [float(candle["stck_clpr"]) for candle in daily_data[:self.macd_long + self.macd_signal]]
            closes.reverse()  # 최신 데이터가 앞에 오도록 순서 변경
            
            # 현재 MACD 계산
            current_macd = self.analyzer.calculate_macd(
                closes, 
                short_period=self.macd_short, 
                long_period=self.macd_long, 
                signal_period=self.macd_signal
            )
            
            # 이전 MACD 계산 (1일 전)
            prev_closes = closes[1:]
            prev_macd = self.analyzer.calculate_macd(
                prev_closes, 
                short_period=self.macd_short, 
                long_period=self.macd_long, 
                signal_period=self.macd_signal
            )
            
            # MACD 데드 크로스 확인 (MACD 라인이 시그널 라인을 하향 돌파)
            is_dead_cross = (prev_macd["macd"] >= prev_macd["signal"]) and (current_macd["macd"] < current_macd["signal"])
            
            # 추가 조건: MACD 히스토그램이 음수로 전환
            is_histogram_negative = (prev_macd["histogram"] >= 0) and (current_macd["histogram"] < 0)
            
            # 매도 시그널
            if is_dead_cross or is_histogram_negative:
                logger.info(f"{code} MACD 매도 시그널 발생: 데드크로스={is_dead_cross}, 히스토그램전환={is_histogram_negative}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"{code} MACD 매도 시그널 확인 중 오류 발생: {e}")
            return False


class MovingAverageStrategy(TradingStrategy):
    """이동평균선 기반 매매 전략"""
    
    def __init__(self, api_client: KoreaInvestmentApiClient, analyzer: TechnicalAnalyzer):
        """
        이동평균선 전략 초기화
        
        Args:
            api_client: API 클라이언트 인스턴스
            analyzer: 기술적 분석기 인스턴스
        """
        super().__init__(api_client, analyzer)
        self.name = "이동평균선 전략"
        self.short_period = TRADE_CONFIG.get("ma_short_period", 5)
        self.long_period = TRADE_CONFIG.get("ma_long_period", 20)
        self.profit_target_pct = TRADE_CONFIG.get("profit_target_pct", 5.0)
        self.stop_loss_pct = TRADE_CONFIG.get("stop_loss_pct", 3.0)
    
    def should_buy(self, code: str, current_price: float) -> bool:
        """
        이동평균선 매수 시그널 확인
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            
        Returns:
            bool: 매수 시그널 여부
        """
        try:
            # 골든 크로스 확인
            is_golden = self.analyzer.is_golden_cross(code, self.short_period, self.long_period)
            if is_golden:
                logger.info(f"{code} 이동평균선 골든크로스 매수 시그널 발생")
                return True
                
            # 현재가와 이동평균선 비교
            ma_long = self.analyzer.get_moving_average(code, self.long_period)
            if ma_long and current_price > ma_long * 1.01:  # 1% 이상 상회
                logger.info(f"{code} 현재가({current_price})가 {self.long_period}일 이동평균선({ma_long:.2f})을 1% 이상 상회")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"{code} 이동평균선 매수 시그널 확인 중 오류 발생: {e}")
            return False
    
    def should_sell(self, code: str, current_price: float, entry_price: float) -> bool:
        """
        이동평균선 매도 시그널 확인
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            entry_price: 매수 가격
            
        Returns:
            bool: 매도 시그널 여부
        """
        try:
            # 수익률 계산
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            
            # 익절 조건 확인
            if profit_pct >= self.profit_target_pct:
                logger.info(f"{code} 익절 조건 도달: 수익률 {profit_pct:.2f}% >= 목표 {self.profit_target_pct:.2f}%")
                return True
                
            # 손절 조건 확인
            if profit_pct <= -self.stop_loss_pct:
                logger.info(f"{code} 손절 조건 도달: 수익률 {profit_pct:.2f}% <= 손절선 -{self.stop_loss_pct:.2f}%")
                return True
            
            # 현재가와 이동평균선 비교
            ma_short = self.analyzer.get_moving_average(code, self.short_period)
            if ma_short and current_price < ma_short * 0.99:  # 1% 이상 하회
                logger.info(f"{code} 현재가({current_price})가 {self.short_period}일 이동평균선({ma_short:.2f})을 1% 이상 하회")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"{code} 이동평균선 매도 시그널 확인 중 오류 발생: {e}")
            return False


class RSIStrategy(TradingStrategy):
    """RSI 기반 매매 전략"""
    
    def __init__(self, api_client: KoreaInvestmentApiClient, analyzer: TechnicalAnalyzer):
        """
        RSI 전략 초기화
        
        Args:
            api_client: API 클라이언트 인스턴스
            analyzer: 기술적 분석기 인스턴스
        """
        super().__init__(api_client, analyzer)
        self.name = "RSI 전략"
        self.rsi_period = TRADE_CONFIG.get("rsi_period", 14)
        self.rsi_oversold = TRADE_CONFIG.get("rsi_oversold", 30)
        self.rsi_overbought = TRADE_CONFIG.get("rsi_overbought", 70)
        self.profit_target_pct = TRADE_CONFIG.get("profit_target_pct", 5.0)
        self.stop_loss_pct = TRADE_CONFIG.get("stop_loss_pct", 3.0)
    
    def should_buy(self, code: str, current_price: float) -> bool:
        """
        RSI 매수 시그널 확인
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            
        Returns:
            bool: 매수 시그널 여부
        """
        try:
            # RSI 계산
            rsi = self.analyzer.calculate_rsi(code, self.rsi_period)
            if rsi is None:
                return False
                
            # 과매도 상태에서 반등 시 매수
            if rsi <= self.rsi_oversold:
                logger.info(f"{code} RSI({rsi:.2f}) 과매도 상태로 매수 시그널 발생")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"{code} RSI 매수 시그널 확인 중 오류 발생: {e}")
            return False
    
    def should_sell(self, code: str, current_price: float, entry_price: float) -> bool:
        """
        RSI 매도 시그널 확인
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            entry_price: 매수 가격
            
        Returns:
            bool: 매도 시그널 여부
        """
        try:
            # 수익률 계산
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            
            # 익절 조건 확인
            if profit_pct >= self.profit_target_pct:
                logger.info(f"{code} 익절 조건 도달: 수익률 {profit_pct:.2f}% >= 목표 {self.profit_target_pct:.2f}%")
                return True
                
            # 손절 조건 확인
            if profit_pct <= -self.stop_loss_pct:
                logger.info(f"{code} 손절 조건 도달: 수익률 {profit_pct:.2f}% <= 손절선 -{self.stop_loss_pct:.2f}%")
                return True
                
            # RSI 계산
            rsi = self.analyzer.calculate_rsi(code, self.rsi_period)
            if rsi is None:
                return False
                
            # 과매수 상태에서 매도
            if rsi >= self.rsi_overbought:
                logger.info(f"{code} RSI({rsi:.2f}) 과매수 상태로 매도 시그널 발생")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"{code} RSI 매도 시그널 확인 중 오류 발생: {e}")
            return False 