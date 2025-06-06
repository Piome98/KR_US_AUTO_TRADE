"""
한국 주식 자동매매 - 서비스 계층

비즈니스 로직을 담당하는 서비스들을 정의합니다.
Trader 클래스의 God Object 문제를 해결하기 위해 기능별로 분리했습니다.

서비스 구성:
- PortfolioService: 포트폴리오 상태 관리
- TradingService: 매매 실행 로직  
- MonitoringService: 모니터링 및 알림
- MarketDataService: 시장 데이터 관리
"""

from .portfolio_service import PortfolioService
from .trading_service import TradingService
from .monitoring_service import MonitoringService
from .market_data_service import MarketDataService

__all__ = [
    'PortfolioService',
    'TradingService', 
    'MonitoringService',
    'MarketDataService'
]

__version__ = "3.0.0"  # 3단계 서비스 계층 분리 