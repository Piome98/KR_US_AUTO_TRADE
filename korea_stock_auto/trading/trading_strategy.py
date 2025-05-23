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
            # 일봉 기준 MACD 데이터 가져오기 (TechnicalAnalyzer 사용)
            macd_daily = self.analyzer.get_macd(code, interval='D', short_window=self.macd_short, long_window=self.macd_long, signal_window=self.macd_signal)

            if not macd_daily or \
               macd_daily.get('macd') is None or macd_daily.get('prev_macd') is None or \
               macd_daily.get('signal') is None or macd_daily.get('prev_signal') is None or \
               macd_daily.get('histogram') is None or macd_daily.get('prev_histogram') is None:
                logger.warning(f"{code} 일봉 MACD 데이터 부족 또는 형식 오류로 매수 시그널 확인 불가. Data: {macd_daily}")
                return False

            # MACD 골든 크로스 확인 (MACD 라인이 시그널 라인을 상향 돌파)
            is_golden_cross = (macd_daily['prev_macd'] <= macd_daily['prev_signal'] and 
                               macd_daily['macd'] > macd_daily['signal'])
            
            # 추가 조건: MACD 히스토그램이 양수로 전환
            is_histogram_positive = (macd_daily['prev_histogram'] <= 0 and macd_daily['histogram'] > 0)
            
            # 매수 시그널
            if is_golden_cross or is_histogram_positive:
                logger.info(f"{code} MACD 매수 시그널 발생: 골든크로스={is_golden_cross}, 히스토그램전환={is_histogram_positive}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"{code} MACD 매수 시그널 확인 중 오류 발생: {e}", exc_info=True)
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
            profit_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            
            # 익절 조건 확인
            if profit_pct >= self.profit_target_pct:
                logger.info(f"{code} 익절 조건 도달: 수익률 {profit_pct:.2f}% >= 목표 {self.profit_target_pct:.2f}%")
                return True
                
            # 손절 조건 확인
            if profit_pct <= -self.stop_loss_pct:
                logger.info(f"{code} 손절 조건 도달: 수익률 {profit_pct:.2f}% <= 손절선 -{self.stop_loss_pct:.2f}%")
                return True
                
            # 일봉 기준 MACD 데이터 가져오기 (TechnicalAnalyzer 사용)
            macd_daily = self.analyzer.get_macd(code, interval='D', short_window=self.macd_short, long_window=self.macd_long, signal_window=self.macd_signal)

            if not macd_daily or \
               macd_daily.get('macd') is None or macd_daily.get('prev_macd') is None or \
               macd_daily.get('signal') is None or macd_daily.get('prev_signal') is None or \
               macd_daily.get('histogram') is None or macd_daily.get('prev_histogram') is None:
                logger.warning(f"{code} 일봉 MACD 데이터 부족 또는 형식 오류로 매도 시그널 확인 불가. Data: {macd_daily}")
                return False
            
            # MACD 데드 크로스 확인 (MACD 라인이 시그널 라인을 하향 돌파)
            is_dead_cross = (macd_daily['prev_macd'] >= macd_daily['prev_signal'] and 
                             macd_daily['macd'] < macd_daily['signal'])
            
            # 추가 조건: MACD 히스토그램이 음수로 전환
            is_histogram_negative = (macd_daily['prev_histogram'] >= 0 and macd_daily['histogram'] < 0)
            
            # 매도 시그널
            if is_dead_cross or is_histogram_negative:
                logger.info(f"{code} MACD 매도 시그널 발생: 데드크로스={is_dead_cross}, 히스토그램전환={is_histogram_negative}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"{code} MACD 매도 시그널 확인 중 오류 발생: {e}", exc_info=True)
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
            # 골든 크로스 확인 (TechnicalAnalyzer 사용)
            is_golden = self.analyzer.check_ma_golden_cross(code, interval='D', short_period=self.short_period, long_period=self.long_period)
            
            if is_golden is None: # 데이터 부족 등으로 판단 불가
                logger.warning(f"{code} 이동평균선 골든크로스 확인 불가 (데이터 부족 등)")
                return False

            if is_golden:
                # 추가 조건: 현재 가격이 장기 이동평균선 위에 있는지 확인
                long_ma = self.analyzer.get_moving_average(code, interval='D', window=self.long_period)
                if long_ma is not None and current_price > long_ma:
                    logger.info(f"{code} 이동평균선 골든크로스 매수 시그널 발생 (현재가: {current_price}, {self.long_period}일선: {long_ma:.2f})")
                    return True
                else:
                    logger.info(f"{code} 골든크로스 발생했으나 현재가가 {self.long_period}일선 아래 ({current_price} <= {long_ma if long_ma else 'N/A'})")
            return False
        except Exception as e:
            logger.error(f"{code} 이동평균선 매수 시그널 확인 중 오류 발생: {e}", exc_info=True)
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
            profit_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            
            # 익절 조건 확인
            if profit_pct >= self.profit_target_pct:
                logger.info(f"{code} 익절 조건 도달: 수익률 {profit_pct:.2f}% >= 목표 {self.profit_target_pct:.2f}%")
                return True
                
            # 손절 조건 확인
            if profit_pct <= -self.stop_loss_pct:
                logger.info(f"{code} 손절 조건 도달: 수익률 {profit_pct:.2f}% <= 손절선 -{self.stop_loss_pct:.2f}%")
                return True
                
            # 데드 크로스 확인 (TechnicalAnalyzer 사용)
            is_dead = self.analyzer.check_ma_dead_cross(code, interval='D', short_period=self.short_period, long_period=self.long_period)

            if is_dead is None: # 데이터 부족 등으로 판단 불가
                logger.warning(f"{code} 이동평균선 데드크로스 확인 불가 (데이터 부족 등)")
                return False

            if is_dead:
                # 추가 조건: 현재 가격이 장기 이동평균선 아래에 있는지 확인
                long_ma = self.analyzer.get_moving_average(code, interval='D', window=self.long_period)
                if long_ma is not None and current_price < long_ma:
                    logger.info(f"{code} 이동평균선 데드크로스 매도 시그널 발생 (현재가: {current_price}, {self.long_period}일선: {long_ma:.2f})")
                    return True
                else:
                    logger.info(f"{code} 데드크로스 발생했으나 현재가가 {self.long_period}일선 위 ({current_price} >= {long_ma if long_ma else 'N/A'})")
            return False
        except Exception as e:
            logger.error(f"{code} 이동평균선 매도 시그널 확인 중 오류 발생: {e}", exc_info=True)
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
        RSI 매수 시그널 확인 (과매도 구간 진입 후 탈출 시)
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            
        Returns:
            bool: 매수 시그널 여부
        """
        try:
            # 일봉 기준 RSI 데이터 가져오기 (TechnicalAnalyzer 사용)
            rsi_data = self.analyzer.get_rsi_values(code, interval='D', window=self.rsi_period) 

            if not rsi_data or rsi_data.get('current_rsi') is None or rsi_data.get('prev_rsi') is None:
                logger.warning(f"{code} 일봉 RSI 데이터 부족으로 매수 시그널 확인 불가. Data: {rsi_data}")
                return False
            
            current_rsi = rsi_data['current_rsi']
            prev_rsi = rsi_data['prev_rsi']

            # 과매도 구간(예: 30) 진입 후, 해당 구간을 상향 돌파할 때 매수
            if prev_rsi < self.rsi_oversold and current_rsi >= self.rsi_oversold:
                logger.info(f"{code} RSI 과매도 탈출 매수 시그널: 현재 RSI ({current_rsi:.2f}) >= 과매도 기준 ({self.rsi_oversold}), 이전 RSI ({prev_rsi:.2f})")
                return True
            return False
        except Exception as e:
            logger.error(f"{code} RSI 매수 시그널 확인 중 오류 발생: {e}", exc_info=True)
            return False

    def should_sell(self, code: str, current_price: float, entry_price: float) -> bool:
        """
        RSI 매도 시그널 확인 (과매수 구간 진입 후 탈출 시)
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            entry_price: 매수 가격
            
        Returns:
            bool: 매도 시그널 여부
        """
        try:
            # 수익률 계산
            profit_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            
            # 익절 조건 확인
            if profit_pct >= self.profit_target_pct:
                logger.info(f"{code} 익절 조건 도달: 수익률 {profit_pct:.2f}% >= 목표 {self.profit_target_pct:.2f}%")
                return True
                
            # 손절 조건 확인
            if profit_pct <= -self.stop_loss_pct:
                logger.info(f"{code} 손절 조건 도달: 수익률 {profit_pct:.2f}% <= 손절선 -{self.stop_loss_pct:.2f}%")
                return True

            # 일봉 기준 RSI 데이터 가져오기 (TechnicalAnalyzer 사용)
            rsi_data = self.analyzer.get_rsi_values(code, interval='D', window=self.rsi_period)

            if not rsi_data or rsi_data.get('current_rsi') is None or rsi_data.get('prev_rsi') is None:
                logger.warning(f"{code} 일봉 RSI 데이터 부족으로 매도 시그널 확인 불가. Data: {rsi_data}")
                return False
            
            current_rsi = rsi_data['current_rsi']
            prev_rsi = rsi_data['prev_rsi']
            
            # 과매수 구간(예: 70) 진입 후, 해당 구간을 하향 돌파할 때 매도
            if prev_rsi > self.rsi_overbought and current_rsi <= self.rsi_overbought:
                logger.info(f"{code} RSI 과매수 탈출 매도 시그널: 현재 RSI ({current_rsi:.2f}) <= 과매수 기준 ({self.rsi_overbought}), 이전 RSI ({prev_rsi:.2f})")
                return True
            return False
        except Exception as e:
            logger.error(f"{code} RSI 매도 시그널 확인 중 오류 발생: {e}", exc_info=True)
            return False 