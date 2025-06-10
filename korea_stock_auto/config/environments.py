"""
한국 주식 자동매매 - 환경별 설정 관리

개발, 프로덕션, 테스트 환경별로 설정을 분리하여 관리합니다.
환경별 특화된 검증 로직과 기본값을 제공합니다.

v1.0 기능:
- 환경별 설정 클래스
- 자동 환경 감지
- 설정 오버라이드 지원
- 환경별 검증 규칙
"""

import os
import logging
from typing import Optional, Dict, Any, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

from .models import (
    AppConfig, APIConfig, TradingConfig, RiskManagementConfig,
    StockFilterConfig, NotificationConfig, SystemConfig
)

logger = logging.getLogger(__name__)


class Environment(Enum):
    """실행 환경 열거형"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"
    STAGING = "staging"


class EnvironmentDetector:
    """환경 자동 감지"""
    
    @staticmethod
    def detect_environment() -> Environment:
        """현재 환경 자동 감지"""
        env_name = os.getenv('TRADING_ENV', 'development').lower()
        
        if env_name in ['dev', 'development']:
            return Environment.DEVELOPMENT
        elif env_name in ['prod', 'production']:
            return Environment.PRODUCTION
        elif env_name in ['test', 'testing']:
            return Environment.TEST
        elif env_name in ['stage', 'staging']:
            return Environment.STAGING
        else:
            logger.warning(f"알 수 없는 환경: {env_name}, development로 설정")
            return Environment.DEVELOPMENT
    
    @staticmethod
    def is_production() -> bool:
        """프로덕션 환경 여부"""
        return EnvironmentDetector.detect_environment() == Environment.PRODUCTION
    
    @staticmethod
    def is_development() -> bool:
        """개발 환경 여부"""
        return EnvironmentDetector.detect_environment() == Environment.DEVELOPMENT
    
    @staticmethod
    def is_test() -> bool:
        """테스트 환경 여부"""
        return EnvironmentDetector.detect_environment() == Environment.TEST


@dataclass
class EnvironmentConfig(ABC):
    """환경별 설정 추상 클래스"""
    environment: Environment
    
    @abstractmethod
    def get_trading_config(self) -> TradingConfig:
        """매매 설정 반환"""
        pass
    
    @abstractmethod
    def get_risk_config(self) -> RiskManagementConfig:
        """리스크 관리 설정 반환"""
        pass
    
    @abstractmethod
    def get_system_config(self) -> SystemConfig:
        """시스템 설정 반환"""
        pass
    
    @abstractmethod
    def get_notification_config(self) -> NotificationConfig:
        """알림 설정 반환"""
        pass
    
    def validate_environment_specific(self) -> bool:
        """환경별 특화 검증"""
        return True


@dataclass
class DevelopmentConfig(EnvironmentConfig):
    """개발 환경 설정"""
    environment: Environment = field(default=Environment.DEVELOPMENT, init=False)
    
    def get_trading_config(self) -> TradingConfig:
        """개발 환경 매매 설정 (안전한 기본값)"""
        return TradingConfig(
            real_trade=False,  # 개발 환경에서는 실거래 금지
            target_buy_count=2,  # 적은 종목 수
            buy_percentage=0.1,  # 낮은 매수 비율
            strategy="macd",
            min_order_amount=10000,
            max_buy_attempts=2,
            max_sell_attempts=3,
            order_wait_time=2.0,  # 느린 주문 처리
            slippage_tolerance=0.02
        )
    
    def get_risk_config(self) -> RiskManagementConfig:
        """개발 환경 리스크 설정 (매우 보수적)"""
        return RiskManagementConfig(
            daily_loss_limit=100000,  # 낮은 손실 한도
            daily_loss_limit_pct=2.0,  # 매우 보수적
            daily_profit_limit_pct=3.0,
            position_loss_limit=20000,
            max_position_size=100,  # 작은 포지션
            exposure_limit_pct=30.0,  # 낮은 노출도
            trailing_stop_pct=2.0,
            max_exposure_ratio=0.1
        )
    
    def get_system_config(self) -> SystemConfig:
        """개발 환경 시스템 설정"""
        return SystemConfig(
            log_level="DEBUG",  # 상세 로깅
            data_update_interval=600,  # 느린 업데이트
            strategy_run_interval=120,  # 느린 전략 실행
            account_update_interval=120
        )
    
    def get_notification_config(self) -> NotificationConfig:
        """개발 환경 알림 설정"""
        return NotificationConfig(
            discord_webhook_url="",  # 개발환경은 알림 비활성화
            enable_discord=False,
            enable_log_file=True
        )
    
    def validate_environment_specific(self) -> bool:
        """개발 환경 특화 검증"""
        trading_config = self.get_trading_config()
        if trading_config.real_trade:
            logger.error("개발 환경에서는 실거래가 금지됩니다")
            return False
        
        risk_config = self.get_risk_config()
        if risk_config.exposure_limit_pct > 50.0:
            logger.error("개발 환경에서는 50% 이상 노출도가 금지됩니다")
            return False
        
        return True


@dataclass
class ProductionConfig(EnvironmentConfig):
    """프로덕션 환경 설정"""
    environment: Environment = field(default=Environment.PRODUCTION, init=False)
    
    def get_trading_config(self) -> TradingConfig:
        """프로덕션 환경 매매 설정 (최적화된 값)"""
        return TradingConfig(
            real_trade=True,  # 실거래 허용
            target_buy_count=4,
            buy_percentage=0.25,
            strategy="macd",
            min_order_amount=10000,
            max_buy_attempts=3,
            max_sell_attempts=5,
            order_wait_time=1.0,
            slippage_tolerance=0.01
        )
    
    def get_risk_config(self) -> RiskManagementConfig:
        """프로덕션 환경 리스크 설정 (균형잡힌 값)"""
        return RiskManagementConfig(
            daily_loss_limit=500000,
            daily_loss_limit_pct=5.0,
            daily_profit_limit_pct=5.0,
            position_loss_limit=50000,
            max_position_size=1000,
            exposure_limit_pct=70.0,
            trailing_stop_pct=3.0,
            max_exposure_ratio=0.2
        )
    
    def get_system_config(self) -> SystemConfig:
        """프로덕션 환경 시스템 설정"""
        return SystemConfig(
            log_level="INFO",  # 표준 로깅
            data_update_interval=300,
            strategy_run_interval=60,
            account_update_interval=60
        )
    
    def get_notification_config(self) -> NotificationConfig:
        """프로덕션 환경 알림 설정"""
        discord_url = os.getenv('DISCORD_WEBHOOK_URL', '')
        return NotificationConfig(
            discord_webhook_url=discord_url,
            enable_discord=bool(discord_url),
            enable_log_file=True
        )
    
    def validate_environment_specific(self) -> bool:
        """프로덕션 환경 특화 검증"""
        # API 키 검증
        if not os.getenv('REAL_APP_KEY') or not os.getenv('REAL_APP_SECRET'):
            logger.error("프로덕션 환경에서는 실제 API 키가 필요합니다")
            return False
        
        # 리스크 관리 검증
        risk_config = self.get_risk_config()
        if risk_config.daily_loss_limit_pct > 10.0:
            logger.error("프로덕션 환경에서는 일일 손실 한도가 10%를 넘을 수 없습니다")
            return False
        
        return True


@dataclass
class TestConfig(EnvironmentConfig):
    """테스트 환경 설정"""
    environment: Environment = field(default=Environment.TEST, init=False)
    
    def get_trading_config(self) -> TradingConfig:
        """테스트 환경 매매 설정 (빠른 테스트용)"""
        return TradingConfig(
            real_trade=False,  # 테스트에서는 실거래 금지
            target_buy_count=1,  # 최소 종목
            buy_percentage=0.05,  # 최소 비율
            strategy="macd",
            min_order_amount=1000,  # 낮은 최소 금액
            max_buy_attempts=1,
            max_sell_attempts=1,
            order_wait_time=0.1,  # 빠른 처리
            slippage_tolerance=0.05
        )
    
    def get_risk_config(self) -> RiskManagementConfig:
        """테스트 환경 리스크 설정 (제한적)"""
        return RiskManagementConfig(
            daily_loss_limit=10000,  # 매우 낮은 한도
            daily_loss_limit_pct=1.0,
            daily_profit_limit_pct=1.0,
            position_loss_limit=5000,
            max_position_size=10,  # 매우 작은 포지션
            exposure_limit_pct=10.0,
            trailing_stop_pct=1.0,
            max_exposure_ratio=0.05
        )
    
    def get_system_config(self) -> SystemConfig:
        """테스트 환경 시스템 설정"""
        return SystemConfig(
            log_level="DEBUG",  # 상세 로깅
            data_update_interval=60,  # 빠른 업데이트
            strategy_run_interval=30,
            account_update_interval=30
        )
    
    def get_notification_config(self) -> NotificationConfig:
        """테스트 환경 알림 설정"""
        return NotificationConfig(
            discord_webhook_url="",  # 테스트는 알림 비활성화
            enable_discord=False,
            enable_log_file=False  # 테스트는 파일 로그도 비활성화
        )
    
    def validate_environment_specific(self) -> bool:
        """테스트 환경 특화 검증"""
        # 테스트 환경에서는 모든 거래가 Mock이어야 함
        return True


class EnvironmentConfigFactory:
    """환경별 설정 팩토리"""
    
    _config_classes: Dict[Environment, Type[EnvironmentConfig]] = {
        Environment.DEVELOPMENT: DevelopmentConfig,
        Environment.PRODUCTION: ProductionConfig,
        Environment.TEST: TestConfig,
        Environment.STAGING: ProductionConfig  # 스테이징은 프로덕션과 동일
    }
    
    @classmethod
    def create_config(cls, environment: Optional[Environment] = None) -> EnvironmentConfig:
        """환경별 설정 생성"""
        if environment is None:
            environment = EnvironmentDetector.detect_environment()
        
        config_class = cls._config_classes.get(environment, DevelopmentConfig)
        config = config_class()
        
        logger.info(f"환경별 설정 생성: {environment.value}")
        
        # 환경별 검증 수행
        if not config.validate_environment_specific():
            raise ValueError(f"환경별 설정 검증 실패: {environment.value}")
        
        return config
    
    @classmethod
    def get_app_config(cls, environment: Optional[Environment] = None) -> AppConfig:
        """완전한 AppConfig 생성"""
        env_config = cls.create_config(environment)
        
        # API 설정은 환경변수에서 가져오기
        real_api = APIConfig(
            app_key=os.getenv('REAL_APP_KEY', ''),
            app_secret=os.getenv('REAL_APP_SECRET', ''),
            base_url=os.getenv('REAL_BASE_URL', 'https://openapi.koreainvestment.com:9443'),
            account_number=os.getenv('REAL_ACCOUNT_NUMBER', ''),
            account_product_code=os.getenv('REAL_ACCOUNT_PRODUCT_CODE', '01')
        )
        
        vts_api = APIConfig(
            app_key=os.getenv('VTS_APP_KEY', ''),
            app_secret=os.getenv('VTS_APP_SECRET', ''),
            base_url=os.getenv('VTS_BASE_URL', 'https://openapivts.koreainvestment.com:29443'),
            account_number=os.getenv('VTS_ACCOUNT_NUMBER', ''),
            account_product_code=os.getenv('VTS_ACCOUNT_PRODUCT_CODE', '01')
        )
        
        # StockFilterConfig는 공통 설정 사용
        stock_filter = StockFilterConfig()
        
        # 실시간 API 사용 여부
        use_realtime = env_config.environment == Environment.PRODUCTION
        
        app_config = AppConfig(
            real_api=real_api,
            vts_api=vts_api,
            use_realtime_api=use_realtime,
            trading=env_config.get_trading_config(),
            risk_management=env_config.get_risk_config(),
            stock_filter=stock_filter,
            notification=env_config.get_notification_config(),
            system=env_config.get_system_config()
        )
        
        # 전체 설정 검증
        if not app_config.validate_all():
            raise ValueError("애플리케이션 설정 검증 실패")
        
        return app_config


# 편의 함수들
def get_environment_config(environment: Optional[Environment] = None) -> EnvironmentConfig:
    """환경별 설정 가져오기"""
    return EnvironmentConfigFactory.create_config(environment)


def get_current_environment() -> Environment:
    """현재 환경 가져오기"""
    return EnvironmentDetector.detect_environment()


def is_production_environment() -> bool:
    """프로덕션 환경 여부"""
    return EnvironmentDetector.is_production()


def is_safe_environment() -> bool:
    """안전한 환경 여부 (개발/테스트)"""
    env = get_current_environment()
    return env in [Environment.DEVELOPMENT, Environment.TEST] 