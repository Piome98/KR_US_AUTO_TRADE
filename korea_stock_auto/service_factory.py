"""
한국 주식 자동매매 - 서비스 팩토리

DI 컨테이너에 등록할 모든 서비스의 팩토리 메서드를 정의합니다.
의존성 관계를 명시적으로 정의하여 테스트와 유지보수를 용이하게 합니다.
"""

from korea_stock_auto.di_container import get_container
from korea_stock_auto.config import get_config, AppConfig
from korea_stock_auto.api import KoreaInvestmentApiClient

# 도메인 서비스 import
from korea_stock_auto.domain import (
    OrderDomainService, PortfolioDomainService, RiskDomainService
)

# Trading 모듈들을 런타임에 동적으로 import하여 순환 참조 방지


def configure_services() -> None:
    """모든 서비스를 DI 컨테이너에 등록 (도메인 서비스 포함)"""
    container = get_container()
    
    # Config 등록 (싱글톤)
    container.register_singleton(
        AppConfig,
        lambda: get_config()
    )
    
    # API 클라이언트 등록 (싱글톤) 
    container.register_singleton(
        KoreaInvestmentApiClient,
        lambda: KoreaInvestmentApiClient(container.get(AppConfig))
    )
    
    # 도메인 서비스들 등록 (싱글톤)
    def register_domain_services():
        """도메인 서비스들 등록"""
        
        # 도메인 서비스들은 상태를 가지지 않으므로 싱글톤으로 등록
        container.register_singleton(
            OrderDomainService,
            lambda: OrderDomainService()
        )
        
        container.register_singleton(
            PortfolioDomainService,
            lambda: PortfolioDomainService()
        )
        
        container.register_singleton(
            RiskDomainService,
            lambda: RiskDomainService()
        )
    
    # Trading 모듈들을 지연 임포트로 등록 (순환 참조 방지)
    def register_trading_services():
        from korea_stock_auto.trading.technical_analyzer import TechnicalAnalyzer
        from korea_stock_auto.trading.stock_selector import StockSelector
        from korea_stock_auto.trading.trade_executor import TradeExecutor
        from korea_stock_auto.trading.risk_manager import RiskManager
        from korea_stock_auto.trading.trading_strategy import TradingStrategy, MACDStrategy, MovingAverageStrategy, RSIStrategy
        
        # API 클라이언트 의존성 가져오기
        api_client = container.get(KoreaInvestmentApiClient)
        config = container.get(AppConfig)
        
        # TechnicalAnalyzer 등록
        container.register_singleton(
            TechnicalAnalyzer,
            lambda: TechnicalAnalyzer(api_client)
        )
        
        # StockSelector 등록
        container.register_singleton(
            StockSelector,
            lambda: StockSelector(api_client, config)
        )
        
        # TradeExecutor 등록  
        container.register_singleton(
            TradeExecutor,
            lambda: TradeExecutor(api_client, config)
        )
        
        # RiskManager 등록
        container.register_singleton(
            RiskManager,
            lambda: RiskManager(api_client, config)
        )
        
        # 전략들 등록 (팩토리 메서드로)
        def create_strategy(strategy_type: str) -> TradingStrategy:
            analyzer = container.get(TechnicalAnalyzer)
            if strategy_type == "ma":
                return MovingAverageStrategy(api_client, analyzer, config)
            elif strategy_type == "rsi":
                return RSIStrategy(api_client, analyzer, config)
            else:
                return MACDStrategy(api_client, analyzer, config)
        
        # 전략은 설정에 따라 다르므로 팩토리 등록
        container.register_singleton(
            TradingStrategy,
            lambda: create_strategy(container.get(AppConfig).trading.strategy)
        )
    
    # 새로운 서비스 계층 등록 (도메인 엔터티 통합)
    def register_service_layer():
        from korea_stock_auto.services import (
            TradingService, PortfolioService, 
            MonitoringService, MarketDataService
        )
        
        # 의존성 가져오기 (동적 import로 클래스 참조)
        from korea_stock_auto.trading.technical_analyzer import TechnicalAnalyzer
        from korea_stock_auto.trading.stock_selector import StockSelector
        from korea_stock_auto.trading.trade_executor import TradeExecutor
        from korea_stock_auto.trading.risk_manager import RiskManager
        from korea_stock_auto.trading.trading_strategy import TradingStrategy
        
        api_client = container.get(KoreaInvestmentApiClient)
        config = container.get(AppConfig)
        executor = container.get(TradeExecutor)
        risk_manager = container.get(RiskManager)
        strategy = container.get(TradingStrategy)
        analyzer = container.get(TechnicalAnalyzer)
        selector = container.get(StockSelector)
        
        # TradingService 등록 (도메인 엔터티 통합 버전)
        container.register_singleton(
            TradingService,
            lambda: TradingService(
                api=api_client,
                config=config,
                executor=executor,
                risk_manager=risk_manager,
                strategy=strategy
            )
        )
        
        # PortfolioService 등록 (도메인 엔터티 통합 버전)
        container.register_singleton(
            PortfolioService,
            lambda: PortfolioService(
                api=api_client,
                config=config
            )
        )
        
        # MonitoringService 등록
        container.register_singleton(
            MonitoringService,
            lambda: MonitoringService(
                config=config
            )
        )
        
        # MarketDataService 등록
        container.register_singleton(
            MarketDataService,
            lambda: MarketDataService(
                api=api_client,
                config=config,
                analyzer=analyzer,
                selector=selector
            )
        )
        
        # 새로운 Trader 등록 (서비스 계층 + 도메인 엔터티 사용)
        from korea_stock_auto.trading.trader_v2 import TraderV2
        container.register_singleton(
            TraderV2,
            lambda: TraderV2(
                config=config,
                trading_service=container.get(TradingService),
                portfolio_service=container.get(PortfolioService),
                monitoring_service=container.get(MonitoringService),
                market_data_service=container.get(MarketDataService),
                selector=selector
            )
        )
    
    # 서비스들 순차적으로 등록
    register_domain_services()      # 도메인 서비스들 먼저 등록
    register_trading_services()     # Trading 서비스들 등록
    register_service_layer()        # 애플리케이션 서비스 계층 등록


def get_service(service_type):
    """서비스 인스턴스 가져오기 (편의 함수)"""
    return get_container().get(service_type) 