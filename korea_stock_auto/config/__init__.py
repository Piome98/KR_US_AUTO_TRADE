"""
한국 주식 자동매매 - Config 패키지

설정 관리를 위한 통합 인터페이스를 제공합니다.

주요 구성 요소:
- models: 설정 데이터클래스들 (AppConfig, TradingConfig 등)
- manager: 설정 파일 로딩 및 관리 (ConfigManager, get_config)
- environments: 환경별 설정 관리 (개발/프로덕션/테스트)

v3.1 개선사항:
- 환경별 설정 자동 감지 및 적용
- DI 컨테이너 최적화 지원
- 설정 검증 강화
- 타입 안전성 향상

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

# 환경별 설정 관리
from .environments import (
    Environment,
    EnvironmentDetector,
    EnvironmentConfig,
    EnvironmentConfigFactory,
    DevelopmentConfig,
    ProductionConfig,
    TestConfig,
    get_environment_config,
    get_current_environment,
    is_production_environment,
    is_safe_environment
)

# 레거시 지원 제거됨 - DI 컨테이너를 통한 AppConfig 사용 권장

# 공개 API
__all__ = [
    # 권장 인터페이스
    'get_config',
    'ConfigManager',
    'ConfigurationError',
    
    # 설정 모델들
    'AppConfig',
    'APIConfig',
    'TradingConfig',
    'RiskManagementConfig',
    'StockFilterConfig',
    'NotificationConfig',
    'SystemConfig',
    
    # 환경별 설정
    'Environment',
    'EnvironmentDetector',
    'EnvironmentConfig',
    'EnvironmentConfigFactory',
    'DevelopmentConfig',
    'ProductionConfig',
    'TestConfig',
    'get_environment_config',
    'get_current_environment',
    'is_production_environment',
    'is_safe_environment'
]

# 버전 정보
__version__ = "3.1.0"  # 환경별 설정 지원 버전

# 편의 함수들
def get_config_for_environment(environment: Environment = None) -> AppConfig:
    """특정 환경에 맞는 설정 가져오기"""
    return EnvironmentConfigFactory.get_app_config(environment)


def validate_current_config() -> bool:
    """현재 설정 검증"""
    try:
        import logging
        logger = logging.getLogger(__name__)
        config = get_config()
        return config.validate_all()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"설정 검증 실패: {e}")
        return False


def get_safe_config() -> AppConfig:
    """안전한 설정 가져오기 (개발 환경 강제)"""
    return EnvironmentConfigFactory.get_app_config(Environment.DEVELOPMENT) 