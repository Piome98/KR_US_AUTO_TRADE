"""
한국 주식 자동매매 - 모니터링 서비스

모니터링 및 알림을 담당합니다:
- 포트폴리오 상태 모니터링
- 로깅 및 알림
- 성과 분석
- 시스템 상태 추적
"""

import logging
import time
from typing import Dict, List, Any, Optional

from korea_stock_auto.config import AppConfig
from korea_stock_auto.trading.risk_manager import RiskManager
from korea_stock_auto.utils.utils import send_message

logger = logging.getLogger(__name__)


class MonitoringService:
    """모니터링 및 알림 서비스"""
    
    def __init__(self, config: AppConfig):
        """
        모니터링 서비스 초기화
        
        Args:
            config: 애플리케이션 설정
        """
        self.config = config
        
        # 모니터링 상태
        self.last_status_log_time: float = 0
        self.status_log_interval: int = 300  # 5분마다 상태 로깅
        self.is_monitoring: bool = False
        
        logger.debug("MonitoringService 초기화 완료")
    
    def start_monitoring(self) -> None:
        """모니터링 시작"""
        self.is_monitoring = True
        self.log_info("모니터링 시작")
    
    def stop_monitoring(self) -> None:
        """모니터링 중지"""
        self.is_monitoring = False
        self.log_info("모니터링 중지")
    
    def log_info(self, message: str) -> None:
        """정보 레벨 로깅"""
        logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """경고 레벨 로깅"""
        logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """오류 레벨 로깅"""
        logger.error(message)
    
    def log_portfolio_status(self, portfolio_status: Dict[str, Any]) -> None:
        """
        포트폴리오 상태 로깅 (단순화된 버전)
        
        Args:
            portfolio_status: 포트폴리오 상태 정보
        """
        try:
            logger.info("===== 포트폴리오 현황 =====")
            
            cash = portfolio_status.get("cash", 0)
            positions = portfolio_status.get("positions", {})
            positions_count = portfolio_status.get("positions_count", 0)
            target_count = portfolio_status.get("target_count", 0)
            
            logger.info(f"총 현금: {cash:,.0f}원")
            logger.info(f"보유 종목 수: {positions_count} / {target_count}")
            
            if not positions:
                logger.info("보유 종목 없음")
            else:
                total_stock_value = 0
                for code, position_info in positions.items():
                    name = position_info.get("name", code)
                    quantity = position_info.get("quantity", 0)
                    entry_price = position_info.get("entry_price", 0)
                    current_value = position_info.get("current_value", 0)
                    
                    total_stock_value += current_value
                    
                    if entry_price > 0:
                        profit_loss = current_value - (entry_price * quantity)
                        profit_loss_rate = (profit_loss / (entry_price * quantity)) * 100 if quantity > 0 else 0
                        logger.info(f"  - {name} ({code}): {quantity}주, 평가액 {current_value:,.0f}원, 수익률 {profit_loss_rate:.2f}%")
                    else:
                        logger.info(f"  - {name} ({code}): {quantity}주, 평가액 {current_value:,.0f}원")
                
                total_assets = cash + total_stock_value
                logger.info(f"총 주식 평가액: {total_stock_value:,.0f}원")
                logger.info(f"총 자산 평가액: {total_assets:,.0f}원")
            
            logger.info("===========================")
            
        except Exception as e:
            logger.error(f"포트폴리오 상태 로깅 중 오류: {e}")
    
    def log_portfolio_status_detailed(self, 
                           portfolio_info: Dict[str, Any], 
                           current_prices: Dict[str, float],
                           positions: Dict[str, Dict[str, Any]]) -> None:
        """
        포트폴리오 상태 상세 로깅 (기존 메서드)
        
        Args:
            portfolio_info: 포트폴리오 정보
            current_prices: 현재가 정보
            positions: 포지션 정보
        """
        try:
            logger.info("===== 포트폴리오 현황 =====")
            
            cash = portfolio_info.get("cash", 0)
            stock_value = portfolio_info.get("stock_value", 0)
            total_assets = portfolio_info.get("total_assets", 0)
            
            logger.info(f"총 현금: {cash:,.0f}원")
            logger.info(f"총 주식 평가액: {stock_value:,.0f}원")
            logger.info(f"총 자산 평가액: {total_assets:,.0f}원")
            
            holdings_count = len(positions)
            target_count = self.config.trading.target_buy_count
            logger.info(f"보유 종목 수: {holdings_count} / {target_count}")
            
            if not positions:
                logger.info("보유 종목 없음")
            else:
                for code, position_info in positions.items():
                    self._log_position_detail(code, position_info, current_prices.get(code, 0))
            
            logger.info("===========================")
            
        except Exception as e:
            logger.error(f"포트폴리오 상태 로깅 중 오류: {e}")
    
    def _log_position_detail(self, code: str, position_info: Dict[str, Any], current_price: float) -> None:
        """
        개별 포지션 상세 로깅
        
        Args:
            code: 종목 코드
            position_info: 포지션 정보
            current_price: 현재가
        """
        name = position_info.get("name", code)
        quantity = position_info.get("quantity", 0)
        entry_price = position_info.get("entry_price", 0)
        
        if current_price > 0 and quantity > 0:
            eval_value = current_price * quantity
            profit_loss = (current_price - entry_price) * quantity if entry_price > 0 else 0
            profit_loss_rate = (profit_loss / (entry_price * quantity)) * 100 if entry_price > 0 and quantity > 0 else 0
            
            logger.info(
                f"  - {name} ({code}): {quantity}주, "
                f"현재가 {current_price:,}, 평가액 {eval_value:,.0f}, "
                f"수익률 {profit_loss_rate:.2f}% (매수가: {entry_price:,})"
            )
        else:
            logger.info(f"  - {name} ({code}): {quantity}주 (현재가 정보 없음)")
    
    def send_trading_notification(self, message: str, level: str = "info") -> None:
        """
        매매 관련 알림 발송
        
        Args:
            message: 알림 메시지
            level: 로그 레벨 (info, warning, error)
        """
        try:
            # 로그 기록
            if level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)
            else:
                logger.info(message)
            
            # 디스코드 알림 (설정이 활성화된 경우)
            if self.config.notification.enable_discord:
                send_message(message, config.notification.discord_webhook_url)
                
        except Exception as e:
            logger.error(f"알림 발송 중 오류: {e}")
    
    def log_trading_start(self, api_type: str, trade_mode: str, strategy: str) -> None:
        """
        매매 시작 로깅
        
        Args:
            api_type: API 타입 (실전투자/모의투자)
            trade_mode: 거래 모드 (실제매매/시뮬레이션)
            strategy: 매매 전략
        """
        start_message = "[시스템 시작] 한국 주식 자동매매 시스템이 시작되었습니다."
        settings_message = f"[시스템 설정] API: {api_type}, 거래모드: {trade_mode}, 전략: {strategy}"
        
        logger.info("한국 주식 자동매매 시스템 시작")
        logger.info(f"API 설정: {api_type}")
        logger.info(f"거래 모드: {trade_mode}")
        logger.info(f"설정된 전략: {strategy}")
        
        self.send_trading_notification(start_message)
        self.send_trading_notification(settings_message) 