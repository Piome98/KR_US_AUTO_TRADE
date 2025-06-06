"""
한국 주식 자동매매 - Config 패키지

설정 관리를 위한 통합 인터페이스를 제공합니다.

주요 구성 요소:
- models: 설정 데이터클래스들 (AppConfig, TradingConfig 등)
- manager: 설정 파일 로딩 및 관리 (ConfigManager, get_config)
- legacy: 하위 호환성 지원 (전역 변수들, 점진적 제거 예정)

권장 사용법:
    from korea_stock_auto.config import get_config
    
    config = get_config()
    strategy = config.trading.strategy
    api_key = config.current_api.app_key
"""

# 주요 공개 인터페이스
from .manager import get_config, ConfigManager, ConfigurationError
from .models import (
    AppConfig, 
    APIConfig,
    TradingConfig,
    RiskManagementConfig,
    StockFilterConfig,
    NotificationConfig,
    SystemConfig
)

# 레거시 지원 제거됨 - DI 컨테이너를 통한 AppConfig 사용 권장

# 공개 API
__all__ = [
    # 권장 인터페이스
    'get_config',
    'ConfigManager',
    'ConfigurationError',
    'AppConfig',
    'APIConfig',
    'TradingConfig',
    'RiskManagementConfig',
    'StockFilterConfig',
    'NotificationConfig',
    'SystemConfig'
]

# 버전 정보
__version__ = "3.0.0"  # Legacy config 완전 제거 버전 