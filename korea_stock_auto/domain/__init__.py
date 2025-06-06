"""
한국 주식 자동매매 - 도메인 엔터티 패키지

4단계: 도메인 엔터티 도입
비즈니스 로직을 중앙화하고 데이터 무결성을 보장합니다.

주요 엔터티:
- Stock: 주식 정보 및 가격 관련 로직
- Order: 주문 정보 및 실행 로직
- Portfolio: 포트폴리오 상태 및 관리 로직
- Position: 포지션 정보 및 손익 계산
- TradingSignal: 매매 신호 및 전략 로직

도메인 서비스:
- OrderDomainService: 주문 관련 복잡한 비즈니스 로직
- PortfolioDomainService: 포트폴리오 관련 복잡한 비즈니스 로직
- RiskDomainService: 리스크 관리 관련 도메인 로직
"""

from .entities import (
    Stock,
    Order,
    Portfolio,
    Position,
    TradingSignal,
    OrderType,
    OrderStatus,
    SignalType,
    SignalStrength
)

from .services import (
    OrderDomainService,
    PortfolioDomainService,
    RiskDomainService
)

from .value_objects import (
    Money,
    Price,
    Quantity,
    Percentage,
    DomainValidationError
)

__all__ = [
    # 엔터티
    'Stock',
    'Order', 
    'Portfolio',
    'Position',
    'TradingSignal',
    
    # Enum 타입들
    'OrderType',
    'OrderStatus', 
    'SignalType',
    'SignalStrength',
    
    # 도메인 서비스
    'OrderDomainService',
    'PortfolioDomainService',
    'RiskDomainService',
    
    # 값 객체
    'Money',
    'Price',
    'Quantity',
    'Percentage',
    'DomainValidationError'
]

__version__ = "4.0.0"  # 4단계 완료 버전 