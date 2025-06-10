"""
매퍼 설정 관리 모듈

Step 5.4: 의존성 주입 및 설정 최적화
환경별 매퍼 캐시 설정, 성능 임계값, 로깅 레벨 등을 중앙 관리
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)


class Environment(Enum):
    """실행 환경"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class CacheConfig:
    """매퍼 캐시 설정"""
    enable_cache: bool = True
    cache_ttl_seconds: int = 60
    max_cache_size: int = 1000
    cache_cleanup_interval: int = 300  # 5분마다 정리


@dataclass
class PerformanceConfig:
    """매퍼 성능 설정"""
    max_response_time_ms: int = 1000  # 최대 응답 시간 (밀리초)
    warning_threshold_ms: int = 500   # 경고 임계값 (밀리초)
    enable_performance_monitoring: bool = True
    log_slow_operations: bool = True


@dataclass
class MapperConfig:
    """개별 매퍼 설정"""
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging_level: int = logging.INFO
    error_retry_count: int = 3
    error_retry_delay_seconds: float = 1.0


@dataclass
class MapperSettings:
    """전체 매퍼 설정"""
    environment: Environment = Environment.DEVELOPMENT
    
    # 매퍼별 개별 설정
    stock_mapper: MapperConfig = field(default_factory=MapperConfig)
    portfolio_mapper: MapperConfig = field(default_factory=MapperConfig)
    account_mapper: MapperConfig = field(default_factory=MapperConfig)
    order_mapper: MapperConfig = field(default_factory=MapperConfig)
    
    # 전역 설정
    global_logging_level: int = logging.INFO
    enable_metrics: bool = True
    metrics_export_interval: int = 60  # 메트릭 내보내기 간격 (초)


def get_default_stock_mapper_config(env: Environment) -> MapperConfig:
    """종목 매퍼 기본 설정"""
    if env == Environment.PRODUCTION:
        # 프로덕션: 짧은 캐시, 성능 최적화
        cache_config = CacheConfig(
            enable_cache=True,
            cache_ttl_seconds=30,
            max_cache_size=2000,
            cache_cleanup_interval=120
        )
        performance_config = PerformanceConfig(
            max_response_time_ms=500,
            warning_threshold_ms=200,
            enable_performance_monitoring=True,
            log_slow_operations=True
        )
        return MapperConfig(
            cache_config=cache_config,
            performance_config=performance_config,
            logging_level=logging.WARNING,
            error_retry_count=2,
            error_retry_delay_seconds=0.5
        )
    
    elif env == Environment.STAGING:
        # 스테이징: 중간 수준 설정
        cache_config = CacheConfig(
            enable_cache=True,
            cache_ttl_seconds=45,
            max_cache_size=1500,
            cache_cleanup_interval=180
        )
        performance_config = PerformanceConfig(
            max_response_time_ms=800,
            warning_threshold_ms=400,
            enable_performance_monitoring=True,
            log_slow_operations=True
        )
        return MapperConfig(
            cache_config=cache_config,
            performance_config=performance_config,
            logging_level=logging.INFO,
            error_retry_count=3,
            error_retry_delay_seconds=1.0
        )
    
    else:  # DEVELOPMENT, TEST
        # 개발/테스트: 디버깅 친화적 설정
        cache_config = CacheConfig(
            enable_cache=True,
            cache_ttl_seconds=60,
            max_cache_size=500,
            cache_cleanup_interval=300
        )
        performance_config = PerformanceConfig(
            max_response_time_ms=2000,
            warning_threshold_ms=1000,
            enable_performance_monitoring=True,
            log_slow_operations=True
        )
        return MapperConfig(
            cache_config=cache_config,
            performance_config=performance_config,
            logging_level=logging.DEBUG,
            error_retry_count=5,
            error_retry_delay_seconds=2.0
        )


def get_default_portfolio_mapper_config(env: Environment) -> MapperConfig:
    """포트폴리오 매퍼 기본 설정"""
    if env == Environment.PRODUCTION:
        cache_config = CacheConfig(
            enable_cache=True,
            cache_ttl_seconds=120,  # 포트폴리오는 더 긴 캐시
            max_cache_size=1000,
            cache_cleanup_interval=300
        )
        performance_config = PerformanceConfig(
            max_response_time_ms=1000,
            warning_threshold_ms=500,
            enable_performance_monitoring=True,
            log_slow_operations=True
        )
        return MapperConfig(
            cache_config=cache_config,
            performance_config=performance_config,
            logging_level=logging.WARNING,
            error_retry_count=2,
            error_retry_delay_seconds=0.5
        )
    
    elif env == Environment.STAGING:
        cache_config = CacheConfig(
            enable_cache=True,
            cache_ttl_seconds=90,
            max_cache_size=800,
            cache_cleanup_interval=240
        )
        performance_config = PerformanceConfig(
            max_response_time_ms=1500,
            warning_threshold_ms=800,
            enable_performance_monitoring=True,
            log_slow_operations=True
        )
        return MapperConfig(
            cache_config=cache_config,
            performance_config=performance_config,
            logging_level=logging.INFO,
            error_retry_count=3,
            error_retry_delay_seconds=1.0
        )
    
    else:  # DEVELOPMENT, TEST
        cache_config = CacheConfig(
            enable_cache=True,
            cache_ttl_seconds=120,
            max_cache_size=500,
            cache_cleanup_interval=300
        )
        performance_config = PerformanceConfig(
            max_response_time_ms=3000,
            warning_threshold_ms=1500,
            enable_performance_monitoring=True,
            log_slow_operations=True
        )
        return MapperConfig(
            cache_config=cache_config,
            performance_config=performance_config,
            logging_level=logging.DEBUG,
            error_retry_count=5,
            error_retry_delay_seconds=2.0
        )


def get_default_account_mapper_config(env: Environment) -> MapperConfig:
    """계좌 매퍼 기본 설정"""
    if env == Environment.PRODUCTION:
        cache_config = CacheConfig(
            enable_cache=True,
            cache_ttl_seconds=60,
            max_cache_size=500,
            cache_cleanup_interval=180
        )
        performance_config = PerformanceConfig(
            max_response_time_ms=800,
            warning_threshold_ms=400,
            enable_performance_monitoring=True,
            log_slow_operations=True
        )
        return MapperConfig(
            cache_config=cache_config,
            performance_config=performance_config,
            logging_level=logging.WARNING,
            error_retry_count=2,
            error_retry_delay_seconds=0.5
        )
    
    elif env == Environment.STAGING:
        cache_config = CacheConfig(
            enable_cache=True,
            cache_ttl_seconds=75,
            max_cache_size=400,
            cache_cleanup_interval=210
        )
        performance_config = PerformanceConfig(
            max_response_time_ms=1200,
            warning_threshold_ms=600,
            enable_performance_monitoring=True,
            log_slow_operations=True
        )
        return MapperConfig(
            cache_config=cache_config,
            performance_config=performance_config,
            logging_level=logging.INFO,
            error_retry_count=3,
            error_retry_delay_seconds=1.0
        )
    
    else:  # DEVELOPMENT, TEST
        cache_config = CacheConfig(
            enable_cache=True,
            cache_ttl_seconds=90,
            max_cache_size=300,
            cache_cleanup_interval=300
        )
        performance_config = PerformanceConfig(
            max_response_time_ms=2500,
            warning_threshold_ms=1200,
            enable_performance_monitoring=True,
            log_slow_operations=True
        )
        return MapperConfig(
            cache_config=cache_config,
            performance_config=performance_config,
            logging_level=logging.DEBUG,
            error_retry_count=5,
            error_retry_delay_seconds=2.0
        )


def get_default_order_mapper_config(env: Environment) -> MapperConfig:
    """주문 매퍼 기본 설정 (실시간성 중요)"""
    if env == Environment.PRODUCTION:
        cache_config = CacheConfig(
            enable_cache=False,  # 주문은 실시간성이 중요
            cache_ttl_seconds=10,
            max_cache_size=100,
            cache_cleanup_interval=60
        )
        performance_config = PerformanceConfig(
            max_response_time_ms=300,  # 주문은 더 빠른 응답 필요
            warning_threshold_ms=150,
            enable_performance_monitoring=True,
            log_slow_operations=True
        )
        return MapperConfig(
            cache_config=cache_config,
            performance_config=performance_config,
            logging_level=logging.INFO,  # 주문은 로깅 중요
            error_retry_count=1,  # 주문은 재시도 최소화
            error_retry_delay_seconds=0.1
        )
    
    elif env == Environment.STAGING:
        cache_config = CacheConfig(
            enable_cache=False,
            cache_ttl_seconds=15,
            max_cache_size=150,
            cache_cleanup_interval=120
        )
        performance_config = PerformanceConfig(
            max_response_time_ms=500,
            warning_threshold_ms=250,
            enable_performance_monitoring=True,
            log_slow_operations=True
        )
        return MapperConfig(
            cache_config=cache_config,
            performance_config=performance_config,
            logging_level=logging.INFO,
            error_retry_count=2,
            error_retry_delay_seconds=0.2
        )
    
    else:  # DEVELOPMENT, TEST
        cache_config = CacheConfig(
            enable_cache=True,  # 개발환경에서는 캐시 허용
            cache_ttl_seconds=30,
            max_cache_size=200,
            cache_cleanup_interval=180
        )
        performance_config = PerformanceConfig(
            max_response_time_ms=1000,
            warning_threshold_ms=500,
            enable_performance_monitoring=True,
            log_slow_operations=True
        )
        return MapperConfig(
            cache_config=cache_config,
            performance_config=performance_config,
            logging_level=logging.DEBUG,
            error_retry_count=3,
            error_retry_delay_seconds=1.0
        )


def create_mapper_settings(env: Optional[Environment] = None) -> MapperSettings:
    """환경에 맞는 매퍼 설정 생성"""
    
    # 환경 변수에서 환경 결정
    if env is None:
        env_str = os.getenv('KOREA_STOCK_ENV', 'development').lower()
        try:
            env = Environment(env_str)
        except ValueError:
            logger.warning(f"알 수 없는 환경: {env_str}, development로 설정")
            env = Environment.DEVELOPMENT
    
    logger.info(f"매퍼 설정 환경: {env.value}")
    
    # 환경별 전역 설정
    if env == Environment.PRODUCTION:
        global_logging = logging.WARNING
        enable_metrics = True
        metrics_interval = 30
    elif env == Environment.STAGING:
        global_logging = logging.INFO
        enable_metrics = True
        metrics_interval = 60
    else:
        global_logging = logging.DEBUG
        enable_metrics = True
        metrics_interval = 120
    
    return MapperSettings(
        environment=env,
        stock_mapper=get_default_stock_mapper_config(env),
        portfolio_mapper=get_default_portfolio_mapper_config(env),
        account_mapper=get_default_account_mapper_config(env),
        order_mapper=get_default_order_mapper_config(env),
        global_logging_level=global_logging,
        enable_metrics=enable_metrics,
        metrics_export_interval=metrics_interval
    )


def get_mapper_cache_config(mapper_name: str, settings: MapperSettings) -> Dict[str, Any]:
    """매퍼별 캐시 설정을 딕셔너리로 반환 (매퍼 클래스가 받을 수 있는 파라미터만)"""
    
    mapper_configs = {
        'stock': settings.stock_mapper,
        'portfolio': settings.portfolio_mapper,
        'account': settings.account_mapper,
        'order': settings.order_mapper
    }
    
    # 매퍼 이름 정규화
    normalized_name = mapper_name.lower().replace('mapper', '')
    
    if normalized_name not in mapper_configs:
        logger.warning(f"알 수 없는 매퍼: {mapper_name}, 기본 설정 사용")
        normalized_name = 'stock'  # 기본값
    
    config = mapper_configs[normalized_name]
    
    # 매퍼 클래스가 실제로 받을 수 있는 파라미터만 반환
    # BaseMapper는 enable_cache와 cache_ttl_seconds만 생성자에서 받음
    return {
        'enable_cache': config.cache_config.enable_cache,
        'cache_ttl_seconds': config.cache_config.cache_ttl_seconds
    }


def get_mapper_extended_config(mapper_name: str, settings: MapperSettings) -> Dict[str, Any]:
    """매퍼별 확장 설정을 딕셔너리로 반환 (모든 설정 포함)"""
    
    mapper_configs = {
        'stock': settings.stock_mapper,
        'portfolio': settings.portfolio_mapper,
        'account': settings.account_mapper,
        'order': settings.order_mapper
    }
    
    # 매퍼 이름 정규화
    normalized_name = mapper_name.lower().replace('mapper', '')
    
    if normalized_name not in mapper_configs:
        logger.warning(f"알 수 없는 매퍼: {mapper_name}, 기본 설정 사용")
        normalized_name = 'stock'  # 기본값
    
    config = mapper_configs[normalized_name]
    
    # 모든 설정 반환 (성능 모니터링 등에 사용)
    return {
        'enable_cache': config.cache_config.enable_cache,
        'cache_ttl_seconds': config.cache_config.cache_ttl_seconds,
        'max_cache_size': config.cache_config.max_cache_size,
        'cache_cleanup_interval': config.cache_config.cache_cleanup_interval
    }


def get_mapper_performance_config(mapper_name: str, settings: MapperSettings) -> Dict[str, Any]:
    """매퍼별 성능 설정을 딕셔너리로 반환"""
    
    mapper_configs = {
        'stock': settings.stock_mapper,
        'portfolio': settings.portfolio_mapper,
        'account': settings.account_mapper,
        'order': settings.order_mapper
    }
    
    # 매퍼 이름 정규화
    normalized_name = mapper_name.lower().replace('mapper', '')
    
    if normalized_name not in mapper_configs:
        logger.warning(f"알 수 없는 매퍼: {mapper_name}, 기본 설정 사용")
        normalized_name = 'stock'  # 기본값
    
    config = mapper_configs[normalized_name]
    
    return {
        'max_response_time_ms': config.performance_config.max_response_time_ms,
        'warning_threshold_ms': config.performance_config.warning_threshold_ms,
        'enable_performance_monitoring': config.performance_config.enable_performance_monitoring,
        'log_slow_operations': config.performance_config.log_slow_operations,
        'error_retry_count': config.error_retry_count,
        'error_retry_delay_seconds': config.error_retry_delay_seconds
    }


# 전역 설정 인스턴스
_mapper_settings: Optional[MapperSettings] = None


def get_mapper_settings() -> MapperSettings:
    """전역 매퍼 설정 인스턴스 가져오기"""
    global _mapper_settings
    if _mapper_settings is None:
        _mapper_settings = create_mapper_settings()
    return _mapper_settings


def reset_mapper_settings() -> None:
    """전역 매퍼 설정 초기화 (주로 테스트용)"""
    global _mapper_settings
    _mapper_settings = None


def update_mapper_settings(new_settings: MapperSettings) -> None:
    """전역 매퍼 설정 업데이트"""
    global _mapper_settings
    _mapper_settings = new_settings
    logger.info(f"매퍼 설정 업데이트 완료: {new_settings.environment.value}") 