"""
한국 주식 자동매매 - 도메인 엔터티

비즈니스 로직을 중앙화하고 데이터 무결성을 보장하는 도메인 엔터티들입니다.
각 엔터티는 식별자를 가지며, 상태 변경과 비즈니스 규칙을 관리합니다.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import uuid4, UUID
import logging

from .value_objects import Money, Price, Quantity, Percentage, DomainValidationError

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """주문 유형"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "PENDING"        # 대기 중
    SUBMITTED = "SUBMITTED"    # 제출됨
    FILLED = "FILLED"         # 체결됨
    CANCELLED = "CANCELLED"   # 취소됨
    REJECTED = "REJECTED"     # 거부됨
    PARTIAL_FILLED = "PARTIAL_FILLED"  # 부분 체결


class SignalType(Enum):
    """매매 신호 유형"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(Enum):
    """신호 강도"""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


@dataclass
class Stock:
    """
    주식 엔터티
    
    책임:
    - 주식 기본 정보 관리
    - 현재 가격 추적
    - 가격 변동률 계산
    - 매매 가능 여부 판단
    """
    
    code: str
    name: str
    current_price: Price
    previous_close: Price
    market_cap: Money = field(default_factory=Money.zero)
    volume: Quantity = field(default_factory=Quantity.zero)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """엔터티 생성 후 검증"""
        if not self.code or len(self.code) != 6:
            raise DomainValidationError(f"주식 코드는 6자리여야 합니다: {self.code}")
        
        if not self.name:
            raise DomainValidationError("주식명은 필수입니다")
        
        if not isinstance(self.current_price, Price):
            raise DomainValidationError("현재 가격은 Price 객체여야 합니다")
        
        if not isinstance(self.previous_close, Price):
            raise DomainValidationError("전일종가는 Price 객체여야 합니다")
    
    def change_percentage(self) -> Percentage:
        """전일 대비 변동률 계산"""
        return self.current_price.change_percentage(self.previous_close)
    
    def update_price(self, new_price: Price) -> None:
        """가격 업데이트"""
        if not isinstance(new_price, Price):
            raise DomainValidationError("새 가격은 Price 객체여야 합니다")
        
        self.current_price = new_price
        self.updated_at = datetime.now()
        
        logger.debug(f"{self.code} 가격 업데이트: {new_price}")
    
    def is_price_up(self) -> bool:
        """상승 여부 확인"""
        return self.current_price.value > self.previous_close.value
    
    def is_price_down(self) -> bool:
        """하락 여부 확인"""
        return self.current_price.value < self.previous_close.value
    
    def is_tradeable(self) -> bool:
        """매매 가능 여부 확인"""
        # 기본적인 매매 가능 조건들
        return (
            self.current_price.value > Money.zero() and
            not self.volume.is_zero() and
            datetime.now().timestamp() - self.updated_at.timestamp() < 300  # 5분 이내 업데이트
        )
    
    def calculate_order_amount(self, quantity: Quantity) -> Money:
        """주문 금액 계산"""
        return Money.won(self.current_price.value.amount * quantity.value)
    
    def __str__(self) -> str:
        change_pct = self.change_percentage()
        direction = "↑" if self.is_price_up() else "↓" if self.is_price_down() else "→"
        return f"{self.name}({self.code}) {self.current_price} {direction}{change_pct}"
    
    def __repr__(self) -> str:
        return f"Stock(code='{self.code}', name='{self.name}', price={self.current_price})"


@dataclass
class Order:
    """
    주문 엔터티
    
    책임:
    - 주문 정보 관리
    - 주문 상태 추적
    - 체결 처리
    - 주문 검증
    """
    
    id: UUID
    stock: Stock
    order_type: OrderType
    quantity: Quantity
    target_price: Price
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Quantity = field(default_factory=Quantity.zero)
    filled_price: Optional[Price] = None
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """엔터티 생성 후 검증"""
        if not isinstance(self.stock, Stock):
            raise DomainValidationError("stock은 Stock 엔터티여야 합니다")
        
        if not isinstance(self.quantity, Quantity) or self.quantity.is_zero():
            raise DomainValidationError("수량은 0보다 커야 합니다")
        
        if not isinstance(self.target_price, Price):
            raise DomainValidationError("목표 가격은 Price 객체여야 합니다")
    
    @classmethod
    def create_buy_order(cls, stock: Stock, quantity: Quantity, target_price: Optional[Price] = None) -> Order:
        """매수 주문 생성"""
        if target_price is None:
            target_price = stock.current_price
        
        return cls(
            id=uuid4(),
            stock=stock,
            order_type=OrderType.BUY,
            quantity=quantity,
            target_price=target_price
        )
    
    @classmethod
    def create_sell_order(cls, stock: Stock, quantity: Quantity, target_price: Optional[Price] = None) -> Order:
        """매도 주문 생성"""
        if target_price is None:
            target_price = stock.current_price
        
        return cls(
            id=uuid4(),
            stock=stock,
            order_type=OrderType.SELL,
            quantity=quantity,
            target_price=target_price
        )
    
    def submit(self) -> None:
        """주문 제출"""
        if self.status != OrderStatus.PENDING:
            raise DomainValidationError(f"PENDING 상태의 주문만 제출할 수 있습니다: {self.status}")
        
        self.status = OrderStatus.SUBMITTED
        self.submitted_at = datetime.now()
        
        logger.info(f"주문 제출: {self}")
    
    def fill(self, filled_quantity: Quantity, filled_price: Price) -> None:
        """주문 체결 처리"""
        if self.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]:
            raise DomainValidationError(f"제출된 주문만 체결할 수 있습니다: {self.status}")
        
        if filled_quantity > self.quantity:
            raise DomainValidationError("체결 수량이 주문 수량을 초과할 수 없습니다")
        
        if filled_quantity > (self.quantity - self.filled_quantity):
            raise DomainValidationError("체결 수량이 남은 수량을 초과할 수 없습니다")
        
        self.filled_quantity = self.filled_quantity + filled_quantity
        self.filled_price = filled_price
        self.filled_at = datetime.now()
        
        # 체결 상태 결정
        if self.filled_quantity == self.quantity:
            self.status = OrderStatus.FILLED
            logger.info(f"주문 완전 체결: {self}")
        else:
            self.status = OrderStatus.PARTIAL_FILLED
            logger.info(f"주문 부분 체결: {self}, 체결량: {filled_quantity}")
    
    def cancel(self, reason: str = "사용자 취소") -> None:
        """주문 취소"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise DomainValidationError(f"취소할 수 없는 주문 상태: {self.status}")
        
        self.status = OrderStatus.CANCELLED
        self.error_message = reason
        
        logger.info(f"주문 취소: {self}, 사유: {reason}")
    
    def reject(self, reason: str) -> None:
        """주문 거부"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            raise DomainValidationError(f"거부할 수 없는 주문 상태: {self.status}")
        
        self.status = OrderStatus.REJECTED
        self.error_message = reason
        
        logger.warning(f"주문 거부: {self}, 사유: {reason}")
    
    def remaining_quantity(self) -> Quantity:
        """남은 수량 계산"""
        return self.quantity - self.filled_quantity
    
    def total_amount(self) -> Money:
        """총 주문 금액"""
        return Money.won(self.target_price.value.amount * self.quantity.value)
    
    def filled_amount(self) -> Money:
        """체결 금액"""
        if self.filled_price is None or self.filled_quantity.is_zero():
            return Money.zero()
        return Money.won(self.filled_price.value.amount * self.filled_quantity.value)
    
    def is_buy_order(self) -> bool:
        """매수 주문 여부"""
        return self.order_type == OrderType.BUY
    
    def is_sell_order(self) -> bool:
        """매도 주문 여부"""
        return self.order_type == OrderType.SELL
    
    def is_active(self) -> bool:
        """활성 주문 여부 (취소/거부/완료되지 않은 주문)"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]
    
    def __str__(self) -> str:
        type_str = "매수" if self.is_buy_order() else "매도"
        return (f"{type_str} {self.stock.code} {self.quantity} @ {self.target_price} "
                f"[{self.status.value}]")
    
    def __repr__(self) -> str:
        return f"Order(id={self.id}, {self.order_type.value}, {self.stock.code}, {self.quantity}, {self.status.value})"


@dataclass
class Position:
    """
    포지션 엔터티
    
    책임:
    - 보유 포지션 정보 관리
    - 손익 계산
    - 평균 단가 관리
    - 매도 가능 수량 관리
    """
    
    stock: Stock
    quantity: Quantity
    average_price: Price
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """엔터티 생성 후 검증"""
        if not isinstance(self.stock, Stock):
            raise DomainValidationError("stock은 Stock 엔터티여야 합니다")
        
        if not isinstance(self.quantity, Quantity):
            raise DomainValidationError("quantity는 Quantity 객체여야 합니다")
        
        if self.quantity.value < 0:
            raise DomainValidationError("포지션 수량은 0 이상이어야 합니다")
        
        if not isinstance(self.average_price, Price):
            raise DomainValidationError("average_price는 Price 객체여야 합니다")
    
    def total_cost(self) -> Money:
        """총 매수 금액"""
        return Money.won(self.average_price.value.amount * self.quantity.value)
    
    def current_value(self) -> Money:
        """현재 평가액"""
        return Money.won(self.stock.current_price.value.amount * self.quantity.value)
    
    def unrealized_pnl(self) -> Money:
        """미실현 손익 (절댓값)"""
        current_val = self.current_value()
        total_cost = self.total_cost()
        
        # 손익 계산 (절댓값으로 반환)
        pnl_amount = abs(current_val.amount - total_cost.amount)
        return Money.won(pnl_amount)
    
    def unrealized_pnl_percentage(self) -> Percentage:
        """미실현 손익률"""
        total_cost = self.total_cost()
        if total_cost.is_zero():
            return Percentage.zero()
        
        current_val = self.current_value()
        pnl_ratio = (current_val.amount - total_cost.amount) / total_cost.amount * 100
        return Percentage(pnl_ratio)
    
    def is_profitable(self) -> bool:
        """수익 포지션 여부"""
        return self.current_value() > self.total_cost()
    
    def is_losing(self) -> bool:
        """손실 포지션 여부"""
        return self.current_value() < self.total_cost()
    
    def add_shares(self, quantity: Quantity, price: Price) -> None:
        """주식 추가 매수 (평균 단가 재계산)"""
        if quantity.is_zero():
            return
        
        # 기존 총액
        existing_total = self.total_cost()
        
        # 추가 매수 총액
        additional_total = Money.won(price.value.amount * quantity.value)
        
        # 새로운 총 수량 및 총액
        new_quantity = self.quantity + quantity
        new_total = existing_total + additional_total
        
        # 새로운 평균 단가 계산
        new_average_price = Price(Money.won(new_total.amount / new_quantity.value))
        
        # 업데이트
        self.quantity = new_quantity
        self.average_price = new_average_price
        self.last_updated = datetime.now()
        
        logger.info(f"{self.stock.code} 포지션 추가: +{quantity}, 새 평균가: {new_average_price}")
    
    def reduce_shares(self, quantity: Quantity) -> Money:
        """주식 매도 (실현 손익 계산)"""
        if quantity > self.quantity:
            raise DomainValidationError("매도 수량이 보유 수량을 초과할 수 없습니다")
        
        if quantity.is_zero():
            return Money.zero()
        
        # 실현 손익 계산 (절댓값)
        sell_cost = Money.won(self.average_price.value.amount * quantity.value)
        sell_value = Money.won(self.stock.current_price.value.amount * quantity.value)
        
        # 수량 차감
        self.quantity = self.quantity - quantity
        self.last_updated = datetime.now()
        
        # 실현 손익 반환 (절댓값)
        realized_pnl_amount = abs(sell_value.amount - sell_cost.amount)
        
        logger.info(f"{self.stock.code} 포지션 매도: -{quantity}, 실현손익: {realized_pnl_amount:,}원")
        
        return Money.won(realized_pnl_amount)
    
    def is_empty(self) -> bool:
        """빈 포지션 여부 (수량 0)"""
        return self.quantity.is_zero()
    
    def can_sell(self, quantity: Quantity) -> bool:
        """매도 가능 여부 확인"""
        return quantity <= self.quantity and not quantity.is_zero()
    
    def __str__(self) -> str:
        pnl_pct = self.unrealized_pnl_percentage()
        pnl_sign = "+" if self.is_profitable() else "-" if self.is_losing() else ""
        return (f"{self.stock.name}({self.stock.code}) {self.quantity} "
                f"@ {self.average_price} → {self.stock.current_price} "
                f"({pnl_sign}{pnl_pct})")
    
    def __repr__(self) -> str:
        return f"Position({self.stock.code}, {self.quantity}, {self.average_price})"


@dataclass
class Portfolio:
    """
    포트폴리오 엔터티
    
    책임:
    - 전체 포트폴리오 상태 관리
    - 현금 및 포지션 관리
    - 총 자산 계산
    - 리스크 메트릭 계산
    """
    
    cash: Money
    positions: Dict[str, Position] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """엔터티 생성 후 검증"""
        if not isinstance(self.cash, Money):
            raise DomainValidationError("cash는 Money 객체여야 합니다")
    
    def total_value(self) -> Money:
        """총 자산 가치"""
        positions_value = Money.zero()
        for position in self.positions.values():
            if not position.is_empty():
                positions_value = positions_value + position.current_value()
        
        return self.cash + positions_value
    
    def total_cost(self) -> Money:
        """총 매수 비용"""
        positions_cost = Money.zero()
        for position in self.positions.values():
            if not position.is_empty():
                positions_cost = positions_cost + position.total_cost()
        
        return positions_cost
    
    def get_position(self, stock_code: str) -> Optional[Position]:
        """포지션 조회"""
        return self.positions.get(stock_code)
    
    def has_position(self, stock_code: str) -> bool:
        """포지션 보유 여부"""
        position = self.get_position(stock_code)
        return position is not None and not position.is_empty()
    
    def add_cash(self, amount: Money) -> None:
        """현금 추가"""
        self.cash = self.cash + amount
        self.last_updated = datetime.now()
    
    def spend_cash(self, amount: Money) -> None:
        """현금 차감"""
        if amount > self.cash:
            raise DomainValidationError("현금이 부족합니다")
        
        self.cash = self.cash - amount
        self.last_updated = datetime.now()
    
    def add_position(self, stock: Stock, quantity: Quantity, price: Price) -> None:
        """포지션 추가/업데이트"""
        if stock.code in self.positions:
            # 기존 포지션에 추가
            self.positions[stock.code].add_shares(quantity, price)
        else:
            # 새 포지션 생성
            self.positions[stock.code] = Position(
                stock=stock,
                quantity=quantity,
                average_price=price
            )
        
        self.last_updated = datetime.now()
    
    def reduce_position(self, stock_code: str, quantity: Quantity) -> Money:
        """포지션 매도"""
        if stock_code not in self.positions:
            raise DomainValidationError(f"보유하지 않은 종목입니다: {stock_code}")
        
        position = self.positions[stock_code]
        realized_pnl = position.reduce_shares(quantity)
        
        # 포지션이 비었으면 제거
        if position.is_empty():
            del self.positions[stock_code]
        
        self.last_updated = datetime.now()
        return realized_pnl
    
    def get_position_count(self) -> int:
        """보유 종목 수"""
        return len([p for p in self.positions.values() if not p.is_empty()])
    
    def get_exposure_ratio(self) -> Percentage:
        """투자 노출도 (투자금액/총자산)"""
        total_val = self.total_value()
        if total_val.is_zero():
            return Percentage.zero()
        
        invested_amount = total_val - self.cash
        exposure_ratio = invested_amount.amount / total_val.amount * 100
        return Percentage(exposure_ratio)
    
    def can_buy(self, amount: Money) -> bool:
        """매수 가능 여부 (현금 충분성)"""
        return amount <= self.cash
    
    def __str__(self) -> str:
        total_val = self.total_value()
        position_count = self.get_position_count()
        exposure = self.get_exposure_ratio()
        
        return (f"Portfolio: 총자산 {total_val}, 현금 {self.cash}, "
                f"보유종목 {position_count}개, 노출도 {exposure}")
    
    def __repr__(self) -> str:
        return f"Portfolio(cash={self.cash}, positions={len(self.positions)})"


@dataclass
class TradingSignal:
    """
    매매 신호 엔터티
    
    책임:
    - 매매 신호 정보 관리
    - 신호 강도 및 근거 추적
    - 신호 유효성 검증
    """
    
    id: UUID
    stock: Stock
    signal_type: SignalType
    strength: SignalStrength
    target_price: Price
    confidence: Percentage
    reason: str
    indicators: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        """엔터티 생성 후 검증"""
        if not isinstance(self.stock, Stock):
            raise DomainValidationError("stock은 Stock 엔터티여야 합니다")
        
        if not isinstance(self.confidence, Percentage):
            raise DomainValidationError("confidence는 Percentage 객체여야 합니다")
        
        if not self.reason:
            raise DomainValidationError("신호 근거는 필수입니다")
    
    @classmethod
    def create_buy_signal(cls, stock: Stock, strength: SignalStrength, target_price: Price,
                         confidence: Percentage, reason: str, 
                         indicators: Optional[Dict[str, float]] = None) -> TradingSignal:
        """매수 신호 생성"""
        return cls(
            id=uuid4(),
            stock=stock,
            signal_type=SignalType.BUY,
            strength=strength,
            target_price=target_price,
            confidence=confidence,
            reason=reason,
            indicators=indicators or {}
        )
    
    @classmethod
    def create_sell_signal(cls, stock: Stock, strength: SignalStrength, target_price: Price,
                          confidence: Percentage, reason: str,
                          indicators: Optional[Dict[str, float]] = None) -> TradingSignal:
        """매도 신호 생성"""
        return cls(
            id=uuid4(),
            stock=stock,
            signal_type=SignalType.SELL,
            strength=strength,
            target_price=target_price,
            confidence=confidence,
            reason=reason,
            indicators=indicators or {}
        )
    
    def is_buy_signal(self) -> bool:
        """매수 신호 여부"""
        return self.signal_type == SignalType.BUY
    
    def is_sell_signal(self) -> bool:
        """매도 신호 여부"""
        return self.signal_type == SignalType.SELL
    
    def is_hold_signal(self) -> bool:
        """보유 신호 여부"""
        return self.signal_type == SignalType.HOLD
    
    def is_strong(self) -> bool:
        """강한 신호 여부"""
        return self.strength == SignalStrength.STRONG
    
    def is_expired(self) -> bool:
        """신호 만료 여부"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """신호 유효성 확인"""
        return not self.is_expired() and self.confidence.value > 50
    
    def price_target_percentage(self) -> Percentage:
        """목표가 대비 현재가 차이율"""
        return self.target_price.change_percentage(self.stock.current_price)
    
    def __str__(self) -> str:
        action = "매수" if self.is_buy_signal() else "매도" if self.is_sell_signal() else "보유"
        return (f"{action} 신호 - {self.stock.code} @ {self.target_price} "
                f"[{self.strength.value}, 신뢰도: {self.confidence}] {self.reason}")
    
    def __repr__(self) -> str:
        return f"TradingSignal({self.signal_type.value}, {self.stock.code}, {self.strength.value})" 