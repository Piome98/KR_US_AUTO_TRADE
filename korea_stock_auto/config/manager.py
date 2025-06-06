"""
한국 주식 자동매매 - 설정 관리자

설정 파일 로딩, 파싱, 검증을 담당합니다.
YAML 파일과 AppConfig 객체 간의 변환을 처리합니다.
"""

import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .models import (
    AppConfig, APIConfig, TradingConfig, RiskManagementConfig,
    StockFilterConfig, NotificationConfig, SystemConfig
)

# 로깅 설정
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """설정 관련 오류"""
    pass


class ConfigManager:
    """설정 관리자 클래스"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        설정 관리자 초기화
        
        Args:
            config_path: 설정 파일 경로 (None인 경우 기본 경로 사용)
        """
        if config_path is None:
            # 프로젝트 루트에서 config.yaml 찾기
            current_dir = Path(__file__).parent.parent.parent
            config_path = current_dir / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Optional[AppConfig] = None
        
        logger.debug(f"ConfigManager 초기화: {self.config_path}")
    
    def load_config(self) -> AppConfig:
        """설정 파일 로드 및 검증"""
        try:
            if not self.config_path.exists():
                raise ConfigurationError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            if not raw_config:
                raise ConfigurationError("설정 파일이 비어있거나 올바르지 않습니다")
            
            config = self._parse_config(raw_config)
            
            if not config.validate_all():
                raise ConfigurationError("설정 검증에 실패했습니다")
            
            self._config = config
            logger.info(f"설정 파일 로드 완료: {self.config_path}")
            
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML 파싱 오류: {e}")
        except Exception as e:
            logger.error(f"설정 로드 중 오류 발생: {e}")
            raise ConfigurationError(f"설정 로드 실패: {e}")
    
    def _parse_config(self, raw_config: Dict[str, Any]) -> AppConfig:
        """원본 설정을 AppConfig 객체로 변환"""
        api_config = raw_config.get('API', {})
        
        # API 설정 파싱
        real_api = APIConfig(
            app_key=api_config.get('REAL', {}).get('APP_KEY', ''),
            app_secret=api_config.get('REAL', {}).get('APP_SECRET', ''),
            base_url=api_config.get('REAL', {}).get('BASE_URL', ''),
            account_number=api_config.get('REAL', {}).get('ACCOUNT_NUMBER', ''),
            account_product_code=api_config.get('ACCOUNT_PRODUCT_CODE', '01')
        )
        
        vts_api = APIConfig(
            app_key=api_config.get('VTS', {}).get('APP_KEY', ''),
            app_secret=api_config.get('VTS', {}).get('APP_SECRET', ''),
            base_url=api_config.get('VTS', {}).get('BASE_URL', ''),
            account_number=api_config.get('VTS', {}).get('ACCOUNT_NUMBER', ''),
            account_product_code=api_config.get('ACCOUNT_PRODUCT_CODE', '01')
        )
        
        # 매매 설정 파싱
        trading_config = raw_config.get('TRADING', {})
        trading = TradingConfig(
            real_trade=trading_config.get('REAL_TRADE', False),
            target_buy_count=trading_config.get('TARGET_BUY_COUNT', 4),
            buy_percentage=trading_config.get('BUY_PERCENTAGE', 0.25),
            strategy=trading_config.get('STRATEGY', 'macd'),
            macd_short_period=trading_config.get('MACD', {}).get('SHORT_PERIOD', 5),
            macd_long_period=trading_config.get('MACD', {}).get('LONG_PERIOD', 60),
            ma_period=trading_config.get('MOVING_AVERAGE', {}).get('PERIOD', 20),
            rsi_period=trading_config.get('RSI', {}).get('PERIOD', 14),
            rsi_oversold=trading_config.get('RSI', {}).get('OVERSOLD', 30),
            rsi_overbought=trading_config.get('RSI', {}).get('OVERBOUGHT', 70),
            min_order_amount=trading_config.get('MIN_ORDER_AMOUNT', 10000),
            max_buy_attempts=trading_config.get('MAX_BUY_ATTEMPTS', 3),
            max_sell_attempts=trading_config.get('MAX_SELL_ATTEMPTS', 5),
            order_wait_time=trading_config.get('ORDER_WAIT_TIME', 1.0),
            slippage_tolerance=trading_config.get('SLIPPAGE_TOLERANCE', 0.01)
        )
        
        # 리스크 관리 설정 파싱
        risk_config = raw_config.get('RISK_MANAGEMENT', {})
        risk_management = RiskManagementConfig(
            daily_loss_limit=risk_config.get('DAILY_LOSS_LIMIT', 500000),
            daily_loss_limit_pct=risk_config.get('DAILY_LOSS_LIMIT_PCT', 5.0),
            daily_profit_limit_pct=risk_config.get('DAILY_PROFIT_LIMIT_PCT', 5.0),
            position_loss_limit=risk_config.get('POSITION_LOSS_LIMIT', 50000),
            max_position_size=risk_config.get('MAX_POSITION_SIZE', 1000),
            exposure_limit_pct=risk_config.get('EXPOSURE_LIMIT_PCT', 70.0),
            trailing_stop_pct=risk_config.get('TRAILING_STOP_PCT', 3.0),
            max_exposure_ratio=risk_config.get('MAX_EXPOSURE_RATIO', 0.2)
        )
        
        # 종목 필터링 설정 파싱
        filter_config = raw_config.get('STOCK_FILTER', {})
        stock_filter = StockFilterConfig(
            trade_volume_threshold=filter_config.get('TRADE_VOLUME_THRESHOLD', 1000000),
            price_threshold=filter_config.get('PRICE_THRESHOLD', 3000),
            market_cap_threshold=filter_config.get('MARKET_CAP_THRESHOLD', 400000000000),
            monthly_volatility_threshold=filter_config.get('MONTHLY_VOLATILITY_THRESHOLD', 10.0),
            trade_volume_increase_ratio=filter_config.get('TRADE_VOLUME_INCREASE_RATIO', 4.0),
            close_price_increase_ratio=filter_config.get('CLOSE_PRICE_INCREASE_RATIO', 0.05),
            score_threshold=filter_config.get('SCORE_THRESHOLD', 2),
            exclude_etf=filter_config.get('EXCLUDE_ETF', True)
        )
        
        # 알림 설정 파싱
        notification_config = raw_config.get('NOTIFICATION', {})
        notification = NotificationConfig(
            discord_webhook_url=notification_config.get('DISCORD_WEBHOOK_URL', ''),
            enable_discord=notification_config.get('ENABLE_DISCORD', True),
            enable_log_file=notification_config.get('ENABLE_LOG_FILE', True)
        )
        
        # 시스템 설정 파싱
        system_config = raw_config.get('SYSTEM', {})
        system = SystemConfig(
            log_level=system_config.get('LOG_LEVEL', 'INFO'),
            data_update_interval=system_config.get('DATA_UPDATE_INTERVAL', 300),
            strategy_run_interval=system_config.get('STRATEGY_RUN_INTERVAL', 60),
            account_update_interval=system_config.get('ACCOUNT_UPDATE_INTERVAL', 60)
        )
        
        return AppConfig(
            real_api=real_api,
            vts_api=vts_api,
            use_realtime_api=api_config.get('USE_REALTIME_API', True),
            trading=trading,
            risk_management=risk_management,
            stock_filter=stock_filter,
            notification=notification,
            system=system
        )
    
    @property
    def config(self) -> AppConfig:
        """현재 로드된 설정 반환"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self) -> AppConfig:
        """설정 다시 로드"""
        self._config = None
        return self.load_config()


# 전역 설정 관리자 인스턴스
_config_manager = ConfigManager()

def get_config() -> AppConfig:
    """전역 설정 객체 반환"""
    return _config_manager.config 