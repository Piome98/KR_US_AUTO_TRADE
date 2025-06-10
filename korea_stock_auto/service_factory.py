"""
한국 주식 자동매매 - 서비스 팩토리

DI 컨테이너에 등록할 모든 서비스의 팩토리 메서드를 정의합니다.
의존성 관계를 명시적으로 정의하여 테스트와 유지보수를 용이하게 합니다.

v2.0 개선사항:
- 서비스 라이프사이클 명확한 정의
- 의존성 자동 추적
- Lazy Loading 최적화
- 스코프 기반 서비스 관리
"""

from .di_container import get_container, ServiceLifecycle
from .config import get_config, AppConfig
from .api import KoreaInvestmentApiClient
from typing import Type, Set, TYPE_CHECKING, TypeVar, Any
from contextlib import AbstractContextManager

# 도메인 서비스 import
from .domain import (
    OrderDomainService, PortfolioDomainService, RiskDomainService
)

# 타입 힌팅을 위한 import (런타임에는 실행되지 않음)
if TYPE_CHECKING:
    from .trading.technical_analyzer import TechnicalAnalyzer
    from .trading.stock_selector import StockSelector
    from .trading.trade_executor import TradeExecutor
    from .trading.risk_manager import RiskManager
    from .trading.trading_strategy import TradingStrategy, MACDStrategy, MovingAverageStrategy, RSIStrategy
    from .services import TradingService, PortfolioService, MonitoringService, MarketDataService

# Trading 모듈들을 런타임에 동적으로 import하여 순환 참조 방지

T = TypeVar('T')


def configure_services() -> None:
    """모든 서비스를 DI 컨테이너에 등록 (API 매퍼 통합)"""
    container = get_container()
    
    # Config 등록 (싱글톤, 즉시 로딩)
    container.register_singleton(
        AppConfig,
        lambda: get_config(),
        lazy=False,  # 설정은 즉시 로딩
        dependencies=set()
    )
    
    # API 매퍼들 등록 (최우선 - 다른 서비스들이 의존)
    _register_api_mappers()
    
    # API 클라이언트 등록 (싱글톤, Lazy Loading) 
    container.register_singleton(
        KoreaInvestmentApiClient,
        lambda: KoreaInvestmentApiClient(container.get(AppConfig)),
        lazy=True,
        dependencies={AppConfig}
    )
    
    # 도메인 서비스들 등록
    _register_domain_services()
    
    # Trading 모듈들을 지연 임포트로 등록 (순환 참조 방지)
    try:
        _register_trading_services()
    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Trading 서비스 등록 건너뜀 (import 실패): {e}")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Trading 서비스 등록 건너뜀 (오류): {e}")
    
    # 새로운 서비스 계층 등록 (도메인 엔터티 통합)
    try:
        _register_service_layer()
    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"서비스 계층 등록 건너뜀 (import 실패): {e}")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"서비스 계층 등록 건너뜀 (오류): {e}")


def configure_mappers_only() -> None:
    """매퍼만 등록하는 간단한 설정 함수 (테스트용)"""
    container = get_container()
    
    # Config 등록
    container.register_singleton(
        AppConfig,
        lambda: get_config(),
        lazy=False,
        dependencies=set()
    )
    
    # API 매퍼들만 등록
    _register_api_mappers()
    
    # 도메인 서비스들 등록 (매퍼 테스트에 필요할 수 있음)
    _register_domain_services()


def _register_api_mappers() -> None:
    """
    API 매퍼들을 DI 컨테이너에 등록
    
    Step 5.4: 의존성 주입 및 설정 최적화
    환경별 설정을 사용하여 매퍼를 등록하고, 성능 모니터링 지원
    """
    container = get_container()
    
    try:
        from .api.mappers import (
            StockMapper, PortfolioMapper, AccountMapper, OrderMapper
        )
        from .config.mapper_config import get_mapper_settings, get_mapper_cache_config
        
        # 환경별 매퍼 설정 가져오기
        mapper_settings = get_mapper_settings()
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"매퍼 등록 환경: {mapper_settings.environment.value}")
        
        # StockMapper 등록 (환경별 설정 적용)
        stock_config = get_mapper_cache_config('stock', mapper_settings)
        container.register_singleton(
            StockMapper,
            lambda: StockMapper(**stock_config),
            lazy=True,
            dependencies=set()
        )
        logger.debug(f"StockMapper 등록 완료: {stock_config}")
        
        # PortfolioMapper 등록 (환경별 설정 적용)
        portfolio_config = get_mapper_cache_config('portfolio', mapper_settings)
        container.register_singleton(
            PortfolioMapper,
            lambda: PortfolioMapper(**portfolio_config),
            lazy=True,
            dependencies=set()
        )
        logger.debug(f"PortfolioMapper 등록 완료: {portfolio_config}")
        
        # AccountMapper 등록 (환경별 설정 적용)
        account_config = get_mapper_cache_config('account', mapper_settings)
        container.register_singleton(
            AccountMapper,
            lambda: AccountMapper(**account_config),
            lazy=True,
            dependencies=set()
        )
        logger.debug(f"AccountMapper 등록 완료: {account_config}")
        
        # OrderMapper 등록 (환경별 설정 적용)
        order_config = get_mapper_cache_config('order', mapper_settings)
        container.register_singleton(
            OrderMapper,
            lambda: OrderMapper(**order_config),
            lazy=True,
            dependencies=set()
        )
        logger.debug(f"OrderMapper 등록 완료: {order_config}")
        
        logger.info("API 매퍼들 환경별 설정으로 DI 컨테이너 등록 완료")
        
    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"API 매퍼 import 실패, 등록 건너뜀: {e}")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"API 매퍼 등록 중 오류: {e}")


def _register_domain_services() -> None:
    """도메인 서비스들 등록 (라이프사이클 최적화)"""
    container = get_container()
    
    # 도메인 서비스들은 상태를 가지지 않으므로 싱글톤으로 등록
    # 하지만 자주 사용되므로 즉시 로딩
    container.register_singleton(
        OrderDomainService,
        lambda: OrderDomainService(),
        lazy=False,  # 주문 관련 로직은 즉시 로딩
        dependencies=set()
    )
    
    container.register_singleton(
        PortfolioDomainService,
        lambda: PortfolioDomainService(),
        lazy=False,  # 포트폴리오 계산은 즉시 로딩
        dependencies=set()
    )
    
    container.register_singleton(
        RiskDomainService,
        lambda: RiskDomainService(),
        lazy=True,  # 리스크 관리는 필요시 로딩
        dependencies=set()
    )


def _register_trading_services() -> None:
    """Trading 서비스들 등록 (라이프사이클 최적화)"""
    container = get_container()
    
    # Lazy import로 순환 참조 방지
    from .trading.technical_analyzer import TechnicalAnalyzer
    from .trading.stock_selector import StockSelector
    from .trading.trade_executor import TradeExecutor
    from .trading.risk_manager import RiskManager
    from .trading.trading_strategy import TradingStrategy, MACDStrategy, MovingAverageStrategy, RSIStrategy
    
    # TechnicalAnalyzer 등록 (싱글톤, Lazy Loading)
    container.register_singleton(
        TechnicalAnalyzer,
        lambda: TechnicalAnalyzer(container.get(KoreaInvestmentApiClient)),
        lazy=True,
        dependencies={KoreaInvestmentApiClient}
    )
    
    # StockSelector 등록 (싱글톤, Lazy Loading)
    container.register_singleton(
        StockSelector,
        lambda: StockSelector(
            container.get(KoreaInvestmentApiClient), 
            container.get(AppConfig)
        ),
        lazy=True,
        dependencies={KoreaInvestmentApiClient, AppConfig}
    )
    
    # TradeExecutor 등록 (싱글톤, 즉시 로딩 - 매매 핵심)
    container.register_singleton(
        TradeExecutor,
        lambda: TradeExecutor(
            container.get(KoreaInvestmentApiClient), 
            container.get(AppConfig)
        ),
        lazy=False,  # 매매 실행은 즉시 로딩
        dependencies={KoreaInvestmentApiClient, AppConfig}
    )
    
    # RiskManager 등록 (싱글톤, 즉시 로딩 - 리스크 관리 핵심)
    container.register_singleton(
        RiskManager,
        lambda: RiskManager(
            container.get(KoreaInvestmentApiClient), 
            container.get(AppConfig)
        ),
        lazy=False,  # 리스크 관리는 즉시 로딩
        dependencies={KoreaInvestmentApiClient, AppConfig}
    )
    
    # 전략들 등록 (팩토리 메서드로, Transient)
    def create_strategy(strategy_type: str = None) -> 'TradingStrategy':
        """전략 팩토리 함수"""
        analyzer = container.get(TechnicalAnalyzer)
        api_client = container.get(KoreaInvestmentApiClient)
        config = container.get(AppConfig)
        
        if strategy_type is None:
            strategy_type = config.trading.strategy
            
        if strategy_type == "ma":
            return MovingAverageStrategy(api_client, analyzer, config)
        elif strategy_type == "rsi":
            return RSIStrategy(api_client, analyzer, config)
        else:
            return MACDStrategy(api_client, analyzer, config)
    
    # 전략은 설정에 따라 다르므로 Transient로 등록
    container.register_transient(
        TradingStrategy,
        lambda: create_strategy(),
        dependencies={TechnicalAnalyzer, KoreaInvestmentApiClient, AppConfig}
    )


def _register_service_layer() -> None:
    """애플리케이션 서비스 계층 등록 (스코프 기반 최적화)"""
    container = get_container()
    
    # Lazy import
    from .services import (
        TradingService, PortfolioService, 
        MonitoringService, MarketDataService
    )
    
    # Trading 모듈들도 import (의존성 때문에 필요)
    from .trading.trade_executor import TradeExecutor
    from .trading.risk_manager import RiskManager
    from .trading.trading_strategy import TradingStrategy
    from .trading.technical_analyzer import TechnicalAnalyzer
    from .trading.stock_selector import StockSelector
    
    # TradingService 등록 (Scoped - 매매 세션별)
    container.register_scoped(
        TradingService,
        lambda: TradingService(
            api=container.get(KoreaInvestmentApiClient),
            config=container.get(AppConfig),
            executor=container.get(TradeExecutor),
            risk_manager=container.get(RiskManager),
            strategy=container.get(TradingStrategy)
        ),
        dependencies={KoreaInvestmentApiClient, AppConfig, TradeExecutor, RiskManager, TradingStrategy}
    )
    
    # PortfolioService 등록 (Scoped - 매매 세션별)
    container.register_scoped(
        PortfolioService,
        lambda: PortfolioService(
            api=container.get(KoreaInvestmentApiClient),
            config=container.get(AppConfig)
        ),
        dependencies={KoreaInvestmentApiClient, AppConfig}
    )
    
    # MonitoringService 등록 (싱글톤, Lazy Loading)
    container.register_singleton(
        MonitoringService,
        lambda: MonitoringService(
            config=container.get(AppConfig)
        ),
        lazy=True,
        dependencies={AppConfig}
    )
    
    # MarketDataService 등록 (싱글톤, Lazy Loading)
    container.register_singleton(
        MarketDataService,
        lambda: MarketDataService(
            api=container.get(KoreaInvestmentApiClient),
            config=container.get(AppConfig),
            analyzer=container.get(TechnicalAnalyzer),
            selector=container.get(StockSelector)
        ),
        lazy=True,
        dependencies={KoreaInvestmentApiClient, AppConfig, TechnicalAnalyzer, StockSelector}
    )
    
    # 새로운 Trader 등록 (Scoped - 매매 세션별)
    def register_trader_v2():
        try:
            from .trading.trader_v2 import TraderV2
            container.register_scoped(
                TraderV2,
                lambda: TraderV2(
                    config=container.get(AppConfig),
                    trading_service=container.get(TradingService),
                    portfolio_service=container.get(PortfolioService),
                    monitoring_service=container.get(MonitoringService),
                    market_data_service=container.get(MarketDataService),
                    selector=container.get(StockSelector)
                ),
                dependencies={AppConfig, TradingService, PortfolioService, MonitoringService, MarketDataService, StockSelector}
            )
        except ImportError:
            # TraderV2가 없으면 무시
            pass
    
    register_trader_v2()


def get_service(service_type: Type[T]) -> T:
    """서비스 인스턴스 가져오기 (편의 함수)"""
    return get_container().get(service_type)


def get_scoped_service(service_type: Type[T], scope_name: str = "trading_session") -> T:
    """스코프 서비스 가져오기 (편의 함수)"""
    container = get_container()
    with container.scope(scope_name):
        return container.get(service_type)


def create_trading_session_scope() -> AbstractContextManager[Any]:
    """매매 세션 스코프 생성 (편의 함수)"""
    container = get_container()
    return container.scope("trading_session")


def get_service_dependencies(service_type: Type) -> Set[Type]:
    """서비스 의존성 정보 가져오기"""
    container = get_container()
    service_info = container.get_service_info(service_type)
    if service_info:
        return set(service_info.get('dependencies', []))
    return set()


def validate_service_configuration() -> bool:
    """서비스 설정 검증"""
    container = get_container()
    stats = container.get_container_stats()
    
    print(f"=== DI 컨테이너 검증 결과 ===")
    print(f"총 서비스 수: {stats['total_services']}")
    print(f"싱글톤 서비스: {stats['singleton_services']}")
    print(f"트랜지언트 서비스: {stats['transient_services']}")
    print(f"스코프 서비스: {stats['scoped_services']}")
    print(f"활성 인스턴스: {stats['active_instances']}")
    
    # 핵심 서비스들이 등록되었는지 확인
    required_services = [
        AppConfig,
        KoreaInvestmentApiClient,
        OrderDomainService,
        PortfolioDomainService
    ]
    
    missing_services = []
    for service_type in required_services:
        if not container.is_registered(service_type):
            missing_services.append(service_type.__name__)
    
    if missing_services:
        print(f"❌ 누락된 핵심 서비스: {missing_services}")
        return False
    
    print("✅ 모든 핵심 서비스가 등록됨")
    return True 