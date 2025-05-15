"""
한국 주식 자동매매 - 위험 관리 모듈
위험 관리 및 손익 제한 관련 클래스 및 함수 정의
"""

import logging
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from korea_stock_auto.config import TRADE_CONFIG
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api import KoreaInvestmentApiClient

logger = logging.getLogger("stock_auto")

class RiskManager:
    """위험 관리 클래스"""
    
    def __init__(self, api_client: KoreaInvestmentApiClient):
        """
        위험 관리자 초기화
        
        Args:
            api_client: API 클라이언트 인스턴스
        """
        self.api = api_client
        
        # 위험 관리 설정 로드
        self.daily_loss_limit = TRADE_CONFIG.get("daily_loss_limit", 0)  # 일일 손실 제한 (원)
        self.daily_loss_limit_pct = TRADE_CONFIG.get("daily_loss_limit_pct", 0)  # 일일 손실 제한 (%)
        self.daily_profit_limit_pct = TRADE_CONFIG.get("daily_profit_limit_pct", 0)  # 일일 수익 제한 (%)
        self.position_loss_limit = TRADE_CONFIG.get("position_loss_limit", 0)  # 포지션당 손실 제한
        self.max_position_size = TRADE_CONFIG.get("max_position_size", 0)  # 최대 포지션 크기
        self.trailing_stop_pct = TRADE_CONFIG.get("trailing_stop_pct", 0)  # 트레일링 스탑 비율
        self.max_exposure_ratio = TRADE_CONFIG.get("max_exposure_ratio", 0.2)  # 최대 노출 비율 (자산의 20%)
        
        # 상태 변수 초기화
        self.daily_pl = 0  # 일일 손익
        self.initial_total_assets = 0  # 초기 총 자산 가치
        self.position_max_prices = {}  # 종목별 최고가 추적
        self.risk_level = "NORMAL"  # 위험 수준 (NORMAL, CAUTION, HIGH)
        self.daily_trade_count = 0  # 일일 거래 횟수
        
        # 초기 자산 가치 설정
        self._initialize_assets()
    
    def _initialize_assets(self):
        """초기 자산 가치 계산"""
        self.initial_total_assets = self.calculate_total_assets()
        # 최초 초기화 시에만 로그 출력 
        logger.info(f"초기 총 자산 가치: {self.initial_total_assets:,}원")
    
    def calculate_total_assets(self) -> float:
        """
        총 자산 가치 계산
        
        Returns:
            float: 총 자산 가치
        """
        try:
            # 현금 잔고 조회
            cash_balance_info = self.api.get_balance()
            cash_balance = cash_balance_info.get("cash", 0) if cash_balance_info else 0
            
            # 주식 잔고 조회
            stock_balance_info = self.api.get_stock_balance()
            
            # 주식 평가 금액 계산
            stock_value = 0
            if stock_balance_info and "stocks" in stock_balance_info:
                for stock in stock_balance_info.get("stocks", []):
                    units = stock.get("units", 0)
                    current_price = stock.get("current_price", 0)
                    stock_value += units * current_price
                    
            # 총 자산 가치
            total_assets = cash_balance + stock_value
            
            return total_assets
            
        except Exception as e:
            logger.error(f"총 자산 가치 계산 중 오류 발생: {e}")
            return 0
    
    def update_daily_pl(self):
        """일일 손익 업데이트"""
        current_assets = self.calculate_total_assets()
        self.daily_pl = current_assets - self.initial_total_assets
        daily_pl_pct = (self.daily_pl / self.initial_total_assets) * 100 if self.initial_total_assets > 0 else 0
        
        logger.info(f"일일 손익 업데이트: {self.daily_pl:,}원 ({daily_pl_pct:.2f}%)")
        
        # 위험 수준 업데이트
        self.update_risk_level()
    
    def update_risk_level(self):
        """위험 수준 업데이트"""
        if self.initial_total_assets <= 0:
            self.risk_level = "NORMAL"
            return
            
        # 일일 손익률 계산
        daily_pl_pct = (self.daily_pl / self.initial_total_assets) * 100
        
        # 위험 수준 결정
        if daily_pl_pct <= -self.daily_loss_limit_pct * 0.7:  # 손실 제한의 70% 도달
            self.risk_level = "CAUTION"
            logger.warning(f"위험 수준 상향: CAUTION (손익률: {daily_pl_pct:.2f}%)")
            send_message(f"[주의] 일일 손실이 제한의 70%에 도달했습니다: {daily_pl_pct:.2f}%")
            
        elif daily_pl_pct <= -self.daily_loss_limit_pct:  # 손실 제한 도달
            self.risk_level = "HIGH"
            logger.warning(f"위험 수준 상향: HIGH (손익률: {daily_pl_pct:.2f}%)")
            send_message(f"[경고] 일일 손실 제한에 도달했습니다: {daily_pl_pct:.2f}%")
            
        elif daily_pl_pct >= self.daily_profit_limit_pct:  # 수익 제한 도달
            self.risk_level = "TAKE_PROFIT"
            logger.info(f"수익 실현 수준 도달: TAKE_PROFIT (손익률: {daily_pl_pct:.2f}%)")
            send_message(f"[알림] 일일 수익 목표에 도달했습니다: {daily_pl_pct:.2f}%")
            
        else:
            self.risk_level = "NORMAL"
    
    def should_stop_trading(self) -> bool:
        """
        거래 중단 여부 확인
        
        Returns:
            bool: 거래 중단 필요 여부
        """
        if self.risk_level == "HIGH":
            logger.warning("위험 수준 HIGH로 거래 중단")
            send_message("[거래 중단] 위험 수준이 높아 거래를 중단합니다.")
            return True
            
        if self.risk_level == "TAKE_PROFIT":
            logger.info("수익 목표 달성으로 거래 중단")
            send_message("[거래 중단] 일일 수익 목표 달성으로 거래를 중단합니다.")
            return True
            
        return False
    
    def check_position_loss_limit(self, code: str, current_price: float, entry_price: float) -> bool:
        """
        포지션별 손실 제한 확인
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            entry_price: 매수 가격
            
        Returns:
            bool: 손실 제한 도달 여부
        """
        if entry_price <= 0:
            return False
            
        # 손실률 계산
        loss_pct = ((current_price - entry_price) / entry_price) * 100
        
        # 손실 제한 확인
        if loss_pct <= -self.position_loss_limit:
            logger.warning(f"{code} 포지션 손실 제한 도달: {loss_pct:.2f}%")
            send_message(f"[손절] {code}: 손실률 {loss_pct:.2f}%로 손절 기준 도달")
            return True
            
        return False
    
    def check_trailing_stop(self, code: str, current_price: float) -> bool:
        """
        트레일링 스탑 확인
        
        Args:
            code: 종목 코드
            current_price: 현재 가격
            
        Returns:
            bool: 트레일링 스탑 도달 여부
        """
        # 트레일링 스탑이 설정되지 않은 경우
        if self.trailing_stop_pct <= 0:
            return False
            
        # 최고가 기록이 없는 경우 현재가로 초기화
        if code not in self.position_max_prices:
            self.position_max_prices[code] = current_price
            return False
            
        # 최고가 업데이트
        max_price = self.position_max_prices[code]
        if current_price > max_price:
            self.position_max_prices[code] = current_price
            return False
            
        # 트레일링 스탑 확인
        drawdown_pct = ((current_price - max_price) / max_price) * 100
        if drawdown_pct <= -self.trailing_stop_pct:
            logger.info(f"{code} 트레일링 스탑 도달: 최고가 {max_price}원 대비 {drawdown_pct:.2f}% 하락")
            send_message(f"[매도 신호] {code}: 최고가 대비 {drawdown_pct:.2f}% 하락으로 트레일링 스탑 발동")
            return True
            
        return False
    
    def check_exposure_limit(self, code: str, buy_qty: int, current_price: float) -> bool:
        """
        노출 한도 확인
        
        Args:
            code: 종목 코드
            buy_qty: 매수 수량
            current_price: 현재 가격
            
        Returns:
            bool: 노출 한도 초과 여부 (True: 초과, False: 안전)
        """
        # 총 자산 가치
        total_assets = self.calculate_total_assets()
        if total_assets <= 0:
            return True  # 자산 정보 오류 시 안전하게 거래 제한
            
        # 현재 보유 주식 가치
        stock_balance = self.api.get_stock_balance()
        current_stock_value = 0
        for stock_code, stock_info in stock_balance.items():
            qty = stock_info.get("qty", 0)
            price = stock_info.get("price", 0)
            current_stock_value += qty * price
            
        # 매수 예정 금액
        buy_amount = buy_qty * current_price
        
        # 매수 후 총 주식 가치
        total_stock_value_after_buy = current_stock_value + buy_amount
        
        # 노출 비율 계산
        exposure_ratio = total_stock_value_after_buy / total_assets
        
        # 노출 한도 확인
        if exposure_ratio > self.max_exposure_ratio:
            logger.warning(f"{code} 매수 시 노출 비율 {exposure_ratio:.2f}로 한도 {self.max_exposure_ratio:.2f} 초과")
            return True
            
        return False
    
    def get_position_diversity(self) -> Dict[str, float]:
        """
        포지션 다양성 분석
        
        Returns:
            dict: 섹터별 비중 정보
        """
        # 보유 주식 정보
        stock_balance = self.api.get_stock_balance()
        if not stock_balance:
            return {}
            
        # 섹터별 금액 합계
        sector_values = {}
        total_value = 0
        
        for code, stock_info in stock_balance.items():
            qty = stock_info.get("qty", 0)
            price = stock_info.get("price", 0)
            value = qty * price
            total_value += value
            
            # 종목 섹터 정보 조회
            stock_detail = self.api.get_stock_info(code)
            if stock_detail:
                sector = stock_detail.get("sector", "기타")
                if sector not in sector_values:
                    sector_values[sector] = 0
                sector_values[sector] += value
        
        # 섹터별 비중 계산
        sector_weights = {}
        if total_value > 0:
            for sector, value in sector_values.items():
                sector_weights[sector] = (value / total_value) * 100
                
        return sector_weights
    
    def reset_daily_stats(self):
        """일일 통계 초기화"""
        self.daily_pl = 0
        self.initial_total_assets = self.calculate_total_assets()
        self.daily_trade_count = 0
        self.risk_level = "NORMAL"
        self.position_max_prices = {}
        
        logger.info(f"일일 통계 초기화 완료. 초기 자산: {self.initial_total_assets:,}원")
        send_message(f"[일일 초기화] 트레이딩 통계가 초기화되었습니다. 초기 자산: {self.initial_total_assets:,}원") 