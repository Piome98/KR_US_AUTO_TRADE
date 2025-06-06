"""
한국 주식 자동매매 - 설정 모델 정의

모든 설정 데이터클래스를 정의합니다.
각 설정 클래스는 검증 로직을 포함합니다.
"""

import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 로깅 설정
logger = logging.getLogger(__name__)


class ConfigValidator(ABC):
    """설정 검증을 위한 추상 클래스"""
    
    @abstractmethod
    def validate(self) -> bool:
        """설정 유효성 검증"""
        pass
    
    @abstractmethod
    def get_validation_errors(self) -> list[str]:
        """검증 오류 목록 반환"""
        pass


@dataclass
class APIConfig(ConfigValidator):
    """API 설정 클래스"""
    app_key: str
    app_secret: str
    base_url: str
    account_number: str
    account_product_code: str = "01"
    
    def validate(self) -> bool:
        """API 설정 유효성 검증"""
        errors = self.get_validation_errors()
        if errors:
            logger.error(f"API 설정 검증 실패: {', '.join(errors)}")
            return False
        return True
    
    def get_validation_errors(self) -> list[str]:
        """API 설정 검증 오류 목록"""
        errors = []
        
        if not self.app_key or len(self.app_key) < 10:
            errors.append("APP_KEY가 유효하지 않습니다")
        
        if not self.app_secret or len(self.app_secret) < 50:
            errors.append("APP_SECRET이 유효하지 않습니다")
        
        if not self.base_url or not self.base_url.startswith('https://'):
            errors.append("BASE_URL이 유효하지 않습니다")
        
        if not self.account_number or len(self.account_number) != 8:
            errors.append("계좌번호가 유효하지 않습니다 (8자리 필요)")
        
        return errors


@dataclass 
class TradingConfig(ConfigValidator):
    """매매 설정 클래스"""
    real_trade: bool = False
    target_buy_count: int = 4
    buy_percentage: float = 0.25
    strategy: str = "macd"
    
    # 기술적 지표 설정
    macd_short_period: int = 5
    macd_long_period: int = 60
    ma_period: int = 20
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    
    # 주문 관련 설정
    min_order_amount: int = 10000
    max_buy_attempts: int = 3
    max_sell_attempts: int = 5
    order_wait_time: float = 1.0
    slippage_tolerance: float = 0.01
    
    def validate(self) -> bool:
        """매매 설정 유효성 검증"""
        errors = self.get_validation_errors()
        if errors:
            logger.error(f"매매 설정 검증 실패: {', '.join(errors)}")
            return False
        return True
    
    def get_validation_errors(self) -> list[str]:
        """매매 설정 검증 오류 목록"""
        errors = []
        
        if self.target_buy_count <= 0 or self.target_buy_count > 10:
            errors.append("매수 종목 수는 1-10 사이여야 합니다")
        
        if self.buy_percentage <= 0 or self.buy_percentage > 1:
            errors.append("매수 비율은 0-1 사이여야 합니다")
        
        if self.strategy not in ["macd", "ma", "rsi"]:
            errors.append("지원하지 않는 전략입니다")
        
        if self.macd_short_period >= self.macd_long_period:
            errors.append("MACD 단기 기간은 장기 기간보다 작아야 합니다")
        
        if self.rsi_oversold >= self.rsi_overbought:
            errors.append("RSI 과매도 값은 과매수 값보다 작아야 합니다")
        
        if self.min_order_amount <= 0:
            errors.append("최소 주문 금액은 양수여야 합니다")
        
        return errors


@dataclass
class RiskManagementConfig(ConfigValidator):
    """리스크 관리 설정 클래스"""
    daily_loss_limit: float = 500000
    daily_loss_limit_pct: float = 5.0
    daily_profit_limit_pct: float = 5.0
    position_loss_limit: float = 50000
    max_position_size: int = 1000
    exposure_limit_pct: float = 70.0
    trailing_stop_pct: float = 3.0
    max_exposure_ratio: float = 0.2
    
    def validate(self) -> bool:
        """리스크 관리 설정 유효성 검증"""
        errors = self.get_validation_errors()
        if errors:
            logger.error(f"리스크 관리 설정 검증 실패: {', '.join(errors)}")
            return False
        return True
    
    def get_validation_errors(self) -> list[str]:
        """리스크 관리 설정 검증 오류 목록"""
        errors = []
        
        if self.daily_loss_limit_pct <= 0 or self.daily_loss_limit_pct > 50:
            errors.append("일일 손실 제한은 0-50% 사이여야 합니다")
        
        if self.daily_profit_limit_pct <= 0 or self.daily_profit_limit_pct > 50:
            errors.append("일일 수익 제한은 0-50% 사이여야 합니다")
        
        if self.exposure_limit_pct <= 0 or self.exposure_limit_pct > 100:
            errors.append("노출도 제한은 0-100% 사이여야 합니다")
        
        if self.trailing_stop_pct <= 0 or self.trailing_stop_pct > 20:
            errors.append("트레일링 스탑은 0-20% 사이여야 합니다")
        
        if self.max_exposure_ratio <= 0 or self.max_exposure_ratio > 1:
            errors.append("최대 노출 비율은 0-1 사이여야 합니다")
        
        return errors


@dataclass
class StockFilterConfig(ConfigValidator):
    """종목 필터링 설정 클래스"""
    trade_volume_threshold: int = 1000000
    price_threshold: int = 3000
    market_cap_threshold: int = 400000000000
    monthly_volatility_threshold: float = 10.0
    trade_volume_increase_ratio: float = 4.0
    close_price_increase_ratio: float = 0.05
    score_threshold: int = 2
    exclude_etf: bool = True
    
    def validate(self) -> bool:
        """종목 필터링 설정 유효성 검증"""
        errors = self.get_validation_errors()
        if errors:
            logger.error(f"종목 필터링 설정 검증 실패: {', '.join(errors)}")
            return False
        return True
    
    def get_validation_errors(self) -> list[str]:
        """종목 필터링 설정 검증 오류 목록"""
        errors = []
        
        if self.trade_volume_threshold <= 0:
            errors.append("거래량 임계값은 양수여야 합니다")
        
        if self.price_threshold <= 0:
            errors.append("가격 임계값은 양수여야 합니다")
        
        if self.score_threshold < 0:
            errors.append("점수 임계값은 0 이상이어야 합니다")
        
        return errors


@dataclass
class NotificationConfig(ConfigValidator):
    """알림 설정 클래스"""
    discord_webhook_url: str = ""
    enable_discord: bool = True
    enable_log_file: bool = True
    
    def validate(self) -> bool:
        """알림 설정 유효성 검증"""
        errors = self.get_validation_errors()
        if errors:
            logger.warning(f"알림 설정 검증 실패: {', '.join(errors)}")
            # 알림 설정은 경고만 하고 계속 진행
        return True
    
    def get_validation_errors(self) -> list[str]:
        """알림 설정 검증 오류 목록"""
        errors = []
        
        if self.enable_discord and not self.discord_webhook_url:
            errors.append("디스코드 알림이 활성화되었지만 웹훅 URL이 설정되지 않았습니다")
        
        if self.discord_webhook_url and not self.discord_webhook_url.startswith('https://discord.com/'):
            errors.append("유효하지 않은 디스코드 웹훅 URL입니다")
        
        return errors


@dataclass
class SystemConfig(ConfigValidator):
    """시스템 설정 클래스"""
    log_level: str = "INFO"
    data_update_interval: int = 300
    strategy_run_interval: int = 60
    account_update_interval: int = 60
    
    def validate(self) -> bool:
        """시스템 설정 유효성 검증"""
        errors = self.get_validation_errors()
        if errors:
            logger.error(f"시스템 설정 검증 실패: {', '.join(errors)}")
            return False
        return True
    
    def get_validation_errors(self) -> list[str]:
        """시스템 설정 검증 오류 목록"""
        errors = []
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            errors.append(f"로그 레벨은 {valid_log_levels} 중 하나여야 합니다")
        
        if self.data_update_interval < 60:
            errors.append("데이터 업데이트 간격은 60초 이상이어야 합니다")
        
        if self.strategy_run_interval < 30:
            errors.append("전략 실행 간격은 30초 이상이어야 합니다")
        
        return errors


@dataclass
class AppConfig:
    """전체 애플리케이션 설정"""
    real_api: APIConfig
    vts_api: APIConfig
    use_realtime_api: bool
    trading: TradingConfig
    risk_management: RiskManagementConfig
    stock_filter: StockFilterConfig
    notification: NotificationConfig
    system: SystemConfig
    
    @property
    def current_api(self) -> APIConfig:
        """현재 사용 중인 API 설정 반환"""
        return self.real_api if self.use_realtime_api else self.vts_api
    
    def validate_all(self) -> bool:
        """모든 설정 검증"""
        configs = [
            self.real_api, self.vts_api, self.trading,
            self.risk_management, self.stock_filter,
            self.notification, self.system
        ]
        
        all_valid = True
        for config in configs:
            if not config.validate():
                all_valid = False
        
        return all_valid 