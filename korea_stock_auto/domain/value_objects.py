"""
한국 주식 자동매매 - 값 객체 (Value Objects)

도메인에서 사용하는 불변 값 객체들을 정의합니다.
값 객체는 불변성을 보장하고 비즈니스 규칙을 검증합니다.
"""

from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Union, Any
import logging

logger = logging.getLogger(__name__)


class DomainValidationError(Exception):
    """도메인 검증 오류"""
    pass


@dataclass(frozen=True)
class Money:
    """
    돈을 나타내는 값 객체
    
    특징:
    - 불변성 보장 (frozen=True)
    - 정확한 소수점 계산 (Decimal 사용)
    - 음수 방지 (비즈니스 규칙)
    - 연산 오버로딩 지원
    """
    
    amount: Decimal
    currency: str = "KRW"
    
    def __post_init__(self):
        """객체 생성 후 검증"""
        if not isinstance(self.amount, Decimal):
            # Decimal로 변환
            object.__setattr__(self, 'amount', Decimal(str(self.amount)))
        
        if self.amount < 0:
            raise DomainValidationError(f"돈은 음수일 수 없습니다: {self.amount}")
        
        if self.currency != "KRW":
            raise DomainValidationError(f"지원하지 않는 통화입니다: {self.currency}")
    
    @classmethod
    def won(cls, amount: Union[int, float, str, Decimal]) -> Money:
        """원화 Money 객체 생성"""
        return cls(Decimal(str(amount)), "KRW")
    
    @classmethod
    def zero(cls) -> Money:
        """0원 Money 객체 생성"""
        return cls.won(0)
    
    def __add__(self, other: Money) -> Money:
        """덧셈 연산"""
        if not isinstance(other, Money):
            raise TypeError("Money 객체끼리만 연산 가능합니다")
        if self.currency != other.currency:
            raise DomainValidationError("다른 통화끼리는 연산할 수 없습니다")
        return Money(self.amount + other.amount, self.currency)
    
    def __sub__(self, other: Money) -> Money:
        """뺄셈 연산"""
        if not isinstance(other, Money):
            raise TypeError("Money 객체끼리만 연산 가능합니다")
        if self.currency != other.currency:
            raise DomainValidationError("다른 통화끼리는 연산할 수 없습니다")
        result_amount = self.amount - other.amount
        if result_amount < 0:
            raise DomainValidationError("결과가 음수가 될 수 없습니다")
        return Money(result_amount, self.currency)
    
    def __mul__(self, multiplier: Union[int, float, Decimal]) -> Money:
        """곱셈 연산"""
        if not isinstance(multiplier, (int, float, Decimal)):
            raise TypeError("숫자와만 곱셈 가능합니다")
        result_amount = self.amount * Decimal(str(multiplier))
        return Money(result_amount, self.currency)
    
    def __truediv__(self, divisor: Union[int, float, Decimal]) -> Money:
        """나눗셈 연산"""
        if not isinstance(divisor, (int, float, Decimal)):
            raise TypeError("숫자로만 나눗셈 가능합니다")
        if divisor == 0:
            raise DomainValidationError("0으로 나눌 수 없습니다")
        result_amount = self.amount / Decimal(str(divisor))
        return Money(result_amount, self.currency)
    
    def __lt__(self, other: Money) -> bool:
        """작다 비교"""
        if not isinstance(other, Money):
            raise TypeError("Money 객체끼리만 비교 가능합니다")
        return self.amount < other.amount
    
    def __le__(self, other: Money) -> bool:
        """작거나 같다 비교"""
        if not isinstance(other, Money):
            raise TypeError("Money 객체끼리만 비교 가능합니다")
        return self.amount <= other.amount
    
    def __gt__(self, other: Money) -> bool:
        """크다 비교"""
        if not isinstance(other, Money):
            raise TypeError("Money 객체끼리만 비교 가능합니다")
        return self.amount > other.amount
    
    def __ge__(self, other: Money) -> bool:
        """크거나 같다 비교"""
        if not isinstance(other, Money):
            raise TypeError("Money 객체끼리만 비교 가능합니다")
        return self.amount >= other.amount
    
    def is_zero(self) -> bool:
        """0원인지 확인"""
        return self.amount == 0
    
    def round(self, decimal_places: int = 0) -> Money:
        """반올림"""
        rounded_amount = self.amount.quantize(
            Decimal('0' if decimal_places == 0 else f"0.{'0' * decimal_places}"),
            rounding=ROUND_HALF_UP
        )
        return Money(rounded_amount, self.currency)
    
    def to_int(self) -> int:
        """정수로 변환 (원 단위)"""
        return int(self.amount)
    
    def to_float(self) -> float:
        """실수로 변환"""
        return float(self.amount)
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"{self.amount:,}원"
    
    def __repr__(self) -> str:
        """개발자용 문자열 표현"""
        return f"Money(amount={self.amount}, currency='{self.currency}')"


@dataclass(frozen=True)
class Price:
    """
    주식 가격을 나타내는 값 객체
    
    특징:
    - Money를 래핑하여 주식 가격 특화 기능 제공
    - 가격 변동률 계산
    - 가격 레벨 비교
    """
    
    value: Money
    
    def __post_init__(self):
        """객체 생성 후 검증"""
        if not isinstance(self.value, Money):
            object.__setattr__(self, 'value', Money.won(self.value))
        
        if self.value.is_zero():
            raise DomainValidationError("주식 가격은 0일 수 없습니다")
    
    @classmethod
    def won(cls, amount: Union[int, float, str, Decimal]) -> Price:
        """원화 가격 생성"""
        return cls(Money.won(amount))
    
    def change_percentage(self, other_price: Price) -> Percentage:
        """다른 가격 대비 변동률 계산"""
        if other_price.value.is_zero():
            raise DomainValidationError("기준 가격이 0일 수 없습니다")
        
        change_ratio = (self.value.amount - other_price.value.amount) / other_price.value.amount * 100
        return Percentage(change_ratio)
    
    def apply_percentage(self, percentage: Percentage) -> Price:
        """퍼센트 적용한 가격 계산"""
        multiplier = 1 + (percentage.value / 100)
        new_value = self.value * multiplier
        return Price(new_value)
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __repr__(self) -> str:
        return f"Price({self.value})"


@dataclass(frozen=True)
class Quantity:
    """
    수량을 나타내는 값 객체
    
    특징:
    - 정수 수량만 허용 (주식은 1주 단위)
    - 음수 방지
    """
    
    value: int
    
    def __post_init__(self):
        """객체 생성 후 검증"""
        if not isinstance(self.value, int):
            try:
                object.__setattr__(self, 'value', int(self.value))
            except (ValueError, TypeError):
                raise DomainValidationError(f"수량은 정수여야 합니다: {self.value}")
        
        if self.value < 0:
            raise DomainValidationError(f"수량은 음수일 수 없습니다: {self.value}")
    
    @classmethod
    def zero(cls) -> Quantity:
        """0 수량 생성"""
        return cls(0)
    
    def __add__(self, other: Quantity) -> Quantity:
        """덧셈 연산"""
        if not isinstance(other, Quantity):
            raise TypeError("Quantity 객체끼리만 연산 가능합니다")
        return Quantity(self.value + other.value)
    
    def __sub__(self, other: Quantity) -> Quantity:
        """뺄셈 연산"""
        if not isinstance(other, Quantity):
            raise TypeError("Quantity 객체끼리만 연산 가능합니다")
        result = self.value - other.value
        if result < 0:
            raise DomainValidationError("결과가 음수가 될 수 없습니다")
        return Quantity(result)
    
    def __mul__(self, multiplier: Union[int, float]) -> Quantity:
        """곱셈 연산"""
        if not isinstance(multiplier, (int, float)):
            raise TypeError("숫자와만 곱셈 가능합니다")
        result = int(self.value * multiplier)
        return Quantity(result)
    
    def __lt__(self, other: Quantity) -> bool:
        """작다 비교"""
        if not isinstance(other, Quantity):
            raise TypeError("Quantity 객체끼리만 비교 가능합니다")
        return self.value < other.value
    
    def __le__(self, other: Quantity) -> bool:
        """작거나 같다 비교"""
        if not isinstance(other, Quantity):
            raise TypeError("Quantity 객체끼리만 비교 가능합니다")
        return self.value <= other.value
    
    def __gt__(self, other: Quantity) -> bool:
        """크다 비교"""
        if not isinstance(other, Quantity):
            raise TypeError("Quantity 객체끼리만 비교 가능합니다")
        return self.value > other.value
    
    def __ge__(self, other: Quantity) -> bool:
        """크거나 같다 비교"""
        if not isinstance(other, Quantity):
            raise TypeError("Quantity 객체끼리만 비교 가능합니다")
        return self.value >= other.value
    
    def __eq__(self, other: Any) -> bool:
        """같다 비교"""
        if not isinstance(other, Quantity):
            return False
        return self.value == other.value
    
    def is_zero(self) -> bool:
        """0인지 확인"""
        return self.value == 0
    
    def __str__(self) -> str:
        return f"{self.value:,}주"
    
    def __repr__(self) -> str:
        return f"Quantity({self.value})"


@dataclass(frozen=True)
class Percentage:
    """
    퍼센트를 나타내는 값 객체
    
    특징:
    - Decimal 사용으로 정확한 계산
    - 범위 검증 (필요시)
    """
    
    value: Decimal
    
    def __post_init__(self):
        """객체 생성 후 검증"""
        if not isinstance(self.value, Decimal):
            object.__setattr__(self, 'value', Decimal(str(self.value)))
    
    @classmethod
    def from_ratio(cls, ratio: Union[float, Decimal]) -> Percentage:
        """비율(0.1 = 10%)에서 퍼센트 생성"""
        return cls(Decimal(str(ratio)) * 100)
    
    @classmethod
    def zero(cls) -> Percentage:
        """0% 생성"""
        return cls(Decimal('0'))
    
    def to_ratio(self) -> Decimal:
        """비율로 변환 (10% = 0.1)"""
        return self.value / 100
    
    def is_positive(self) -> bool:
        """양수인지 확인"""
        return self.value > 0
    
    def is_negative(self) -> bool:
        """음수인지 확인"""
        return self.value < 0
    
    def is_zero(self) -> bool:
        """0인지 확인"""
        return self.value == 0
    
    def abs(self) -> Percentage:
        """절댓값"""
        return Percentage(abs(self.value))
    
    def __add__(self, other: Percentage) -> Percentage:
        """덧셈 연산"""
        if not isinstance(other, Percentage):
            raise TypeError("Percentage 객체끼리만 연산 가능합니다")
        return Percentage(self.value + other.value)
    
    def __sub__(self, other: Percentage) -> Percentage:
        """뺄셈 연산"""
        if not isinstance(other, Percentage):
            raise TypeError("Percentage 객체끼리만 연산 가능합니다")
        return Percentage(self.value - other.value)
    
    def __lt__(self, other: Percentage) -> bool:
        """작다 비교"""
        if not isinstance(other, Percentage):
            raise TypeError("Percentage 객체끼리만 비교 가능합니다")
        return self.value < other.value
    
    def __le__(self, other: Percentage) -> bool:
        """작거나 같다 비교"""
        if not isinstance(other, Percentage):
            raise TypeError("Percentage 객체끼리만 비교 가능합니다")
        return self.value <= other.value
    
    def __gt__(self, other: Percentage) -> bool:
        """크다 비교"""
        if not isinstance(other, Percentage):
            raise TypeError("Percentage 객체끼리만 비교 가능합니다")
        return self.value > other.value
    
    def __ge__(self, other: Percentage) -> bool:
        """크거나 같다 비교"""
        if not isinstance(other, Percentage):
            raise TypeError("Percentage 객체끼리만 비교 가능합니다")
        return self.value >= other.value
    
    def __str__(self) -> str:
        return f"{self.value:.2f}%"
    
    def __repr__(self) -> str:
        return f"Percentage({self.value})" 