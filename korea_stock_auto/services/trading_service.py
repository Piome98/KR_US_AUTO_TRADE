"""
한국 주식 자동매매 - 매매 서비스

매매 실행 로직을 담당합니다:
- 매수 로직 실행
- 매도 로직 실행
- 매매 신호 판단
- 리스크 관리 통합
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

from korea_stock_auto.config import AppConfig
from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.trading.trade_executor import TradeExecutor
from korea_stock_auto.trading.risk_manager import RiskManager
from korea_stock_auto.trading.trading_strategy import TradingStrategy
from korea_stock_auto.utils.utils import send_message

# 도메인 엔터티 및 서비스 import
from korea_stock_auto.domain import (
    Stock, Order, Portfolio, Position, TradingSignal,
    OrderType, SignalType, SignalStrength
)
from korea_stock_auto.domain import Money, Price, Quantity, Percentage
from korea_stock_auto.domain import OrderDomainService, RiskDomainService

logger = logging.getLogger(__name__)


class TradingService:
    """매매 실행 서비스 (도메인 엔터티 통합)"""
    
    def __init__(self, 
                 api: KoreaInvestmentApiClient,
                 config: AppConfig,
                 executor: TradeExecutor,
                 risk_manager: RiskManager,
                 strategy: TradingStrategy):
        """
        매매 서비스 초기화
        
        Args:
            api: 한국투자증권 API 클라이언트
            config: 애플리케이션 설정
            executor: 거래 실행기
            risk_manager: 리스크 관리자
            strategy: 매매 전략
        """
        self.api = api
        self.config = config
        self.executor = executor
        self.risk_manager = risk_manager
        self.strategy = strategy
        
        logger.debug("TradingService 초기화 완료 (도메인 엔터티 통합)")
    
    def execute_buy_order(self, 
                         stock: Stock,
                         portfolio: Portfolio) -> Tuple[bool, Dict[str, Any]]:
        """
        매수 주문 실행 (도메인 엔터티 사용)
        
        Args:
            stock: 주식 엔터티
            portfolio: 포트폴리오 엔터티
            
        Returns:
            Tuple[bool, Dict]: (성공 여부, 주문 결과)
        """
        try:
            # 매수 신호 확인
            if not self.strategy.should_buy(stock.code, stock.current_price.value.to_float()):
                return False, {"reason": "매수 신호 없음"}
            
            # 매수 수량 계산
            buy_percentage = self.config.trading.buy_percentage
            available_cash = portfolio.cash
            quantity = OrderDomainService.calculate_buy_quantity(
                available_cash, stock.current_price, buy_percentage
            )
            
            if quantity.is_zero():
                return False, {"reason": "매수 가능 수량이 0입니다"}
            
            # 주문 도메인 검증
            if not OrderDomainService.validate_buy_order(
                portfolio, stock, quantity, stock.current_price
            ):
                return False, {"reason": "주문 검증 실패"}
            
            # 리스크 관리 확인
            order_amount = stock.calculate_order_amount(quantity)
            if not self.risk_manager.can_buy(
                stock.code, 
                order_amount.to_float(), 
                available_cash.to_float(), 
                portfolio.get_position_count()
            ):
                return False, {"reason": "리스크 관리 조건에 따라 매수 보류"}
            
            # 도메인 Order 엔터티 생성
            order = Order.create_buy_order(stock, quantity, stock.current_price)
            
            # 매수 주문 실행
            success, order_details = self.executor.place_order(
                stock.code, "buy", stock.current_price.value.to_float(), 
                order_cash=order_amount.to_float()
            )
            
            if success:
                # 주문 체결 처리
                filled_quantity = Quantity(order_details.get('quantity', quantity.value))
                filled_price = Price.won(order_details.get('price', stock.current_price.value.to_float()))
                order.submit()
                order.fill(filled_quantity, filled_price)
                
                logger.info(f"{stock.code} 매수 주문 성공: {order}")
                send_message(f"[매수 성공] {stock.code} 가격: {filled_price} ({filled_quantity}주)", self.config.notification.discord_webhook_url)
                
                return True, {
                    "order_id": str(order.id),
                    "quantity": filled_quantity.value,
                    "price": filled_price.value.to_float(),
                    "amount": order.filled_amount().to_float(),
                    "order": order
                }
            else:
                order.reject(f"주문 실행 실패: {order_details}")
                logger.error(f"{stock.code} 매수 주문 실패: {order_details}")
                send_message(f"[매수 실패] {stock.code} 가격: {stock.current_price} ({order_details})", self.config.notification.discord_webhook_url)
                return False, {"reason": order_details, "order": order}
                
        except Exception as e:
            logger.error(f"{stock.code} 매수 처리 중 오류: {e}")
            return False, {"reason": f"매수 처리 오류: {e}"}
    
    def execute_sell_order(self, 
                          position: Position) -> Tuple[bool, Dict[str, Any]]:
        """
        매도 주문 실행 (도메인 엔터티 사용)
        
        Args:
            position: 포지션 엔터티
            
        Returns:
            Tuple[bool, Dict]: (성공 여부, 주문 결과)
        """
        try:
            stock = position.stock
            
            # 매도 신호 판단
            sell_signal = self._determine_sell_signal(position)
            if not sell_signal:
                return False, {"reason": "매도 신호 없음"}
            
            if position.quantity.is_zero():
                return False, {"reason": "매도 가능 수량이 없음"}
            
            # 도메인 Order 엔터티 생성
            order = Order.create_sell_order(stock, position.quantity, stock.current_price)
            
            # 매도 주문 실행
            success, order_details = self.executor.place_order(
                stock.code, "sell", stock.current_price.value.to_float(), 
                quantity=position.quantity.value
            )
            
            if success:
                # 주문 체결 처리
                filled_quantity = Quantity(order_details.get('quantity', position.quantity.value))
                filled_price = Price.won(order_details.get('price', stock.current_price.value.to_float()))
                order.submit()
                order.fill(filled_quantity, filled_price)
                
                logger.info(f"{stock.code} 매도 주문 성공: {order} (사유: {sell_signal.reason})")
                send_message(f"[매도 성공] {stock.code} 가격: {filled_price} ({filled_quantity}주) - {sell_signal.reason}", self.config.notification.discord_webhook_url)
                
                return True, {
                    "order_id": str(order.id),
                    "quantity": filled_quantity.value,
                    "price": filled_price.value.to_float(),
                    "amount": order.filled_amount().to_float(),
                    "reason": sell_signal.reason,
                    "order": order
                }
            else:
                order.reject(f"주문 실행 실패: {order_details}")
                logger.error(f"{stock.code} 매도 주문 실패: {order_details}")
                send_message(f"[매도 실패] {stock.code} 가격: {stock.current_price} ({order_details})", self.config.notification.discord_webhook_url)
                return False, {"reason": order_details, "order": order}
                
        except Exception as e:
            logger.error(f"{position.stock.code} 매도 처리 중 오류: {e}")
            return False, {"reason": f"매도 처리 오류: {e}"}
    
    def _determine_sell_signal(self, position: Position) -> Optional[TradingSignal]:
        """
        매도 신호 판단 (도메인 엔터티 사용)
        
        Args:
            position: 포지션 엔터티
            
        Returns:
            Optional[TradingSignal]: 매도 신호 (None이면 매도 안함)
        """
        stock = position.stock
        current_price = stock.current_price.value.to_float()
        entry_price = position.average_price.value.to_float()
        
        # 손절 조건 확인
        if self.risk_manager.check_stop_loss(stock.code, current_price, entry_price):
            return TradingSignal.create_sell_signal(
                stock=stock,
                strength=SignalStrength.STRONG,
                target_price=stock.current_price,
                confidence=Percentage(90),
                reason="손절",
                indicators={"loss_pct": position.unrealized_pnl_percentage().value}
            )
        
        # 익절 조건 확인
        if self.risk_manager.check_take_profit(stock.code, current_price, entry_price):
            return TradingSignal.create_sell_signal(
                stock=stock,
                strength=SignalStrength.MODERATE,
                target_price=stock.current_price,
                confidence=Percentage(80),
                reason="익절",
                indicators={"profit_pct": position.unrealized_pnl_percentage().value}
            )
        
        # 전략 기반 매도 신호 확인
        if self.strategy.should_sell(stock.code, current_price, entry_price):
            return TradingSignal.create_sell_signal(
                stock=stock,
                strength=SignalStrength.MODERATE,
                target_price=stock.current_price,
                confidence=Percentage(70),
                reason="전략 신호",
                indicators={"strategy": self.config.trading.strategy}
            )
        
        return None
    
    def execute_sell_all_positions(self, 
                                  portfolio: Portfolio) -> Tuple[int, int, List[str]]:
        """
        모든 포지션 매도 (도메인 엔터티 사용)
        
        Args:
            portfolio: 포트폴리오 엔터티
            
        Returns:
            Tuple[int, int, List[str]]: (성공 개수, 실패 개수, 실패 종목 리스트)
        """
        positions = portfolio.positions
        
        if not positions:
            logger.info("매도할 보유 종목이 없습니다.")
            return 0, 0, []
        
        logger.info(f"모든 보유 종목 매도 시작 ({len(positions)}개)")
        send_message(f"[전량 매도] {len(positions, self.config.notification.discord_webhook_url)}개 종목 매도 시작")
        
        success_count = 0
        fail_list = []
        
        for code, position in positions.items():
            try:
                if position.is_empty():
                    logger.warning(f"{code}({position.stock.name}) 보유 수량이 없음")
                    fail_list.append(f"{code}({position.stock.name}) - 수량 없음")
                    continue
                
                # 강제 매도 주문 생성 (시장가)
                order = Order.create_sell_order(
                    position.stock, 
                    position.quantity, 
                    position.stock.current_price
                )
                
                success, order_details = self.executor.place_order(
                    code, "sell", position.stock.current_price.value.to_float(),
                    quantity=position.quantity.value
                )
                
                if success:
                    success_count += 1
                    filled_quantity = Quantity(order_details.get('quantity', position.quantity.value))
                    filled_price = Price.won(order_details.get('price', position.stock.current_price.value.to_float()))
                    order.submit()
                    order.fill(filled_quantity, filled_price)
                    
                    logger.info(f"{code} 매도 성공: {filled_quantity}주 @ {filled_price}")
                else:
                    fail_list.append(f"{code}({position.stock.name}) - {order_details}")
                    logger.error(f"{code} 매도 실패: {order_details}")
                    
            except Exception as e:
                fail_list.append(f"{code} - 오류: {e}")
                logger.error(f"{code} 매도 중 오류: {e}")
        
        fail_count = len(fail_list)
        logger.info(f"전량 매도 완료: 성공 {success_count}개, 실패 {fail_count}개")
        send_message(f"[전량 매도 완료] 성공: {success_count}개, 실패: {fail_count}개")
        
        return success_count, fail_count, fail_list
    
    def _get_current_price_from_api(self, code: str) -> float:
        """
        API에서 현재가 조회 (백워드 호환성)
        
        Args:
            code: 종목 코드
            
        Returns:
            float: 현재가 (조회 실패 시 0)
        """
        try:
            price_data = self.api.get_current_price(code)
            if price_data and "price" in price_data:
                return float(price_data["price"])
            return 0.0
        except Exception as e:
            logger.error(f"{code} 현재가 조회 실패: {e}")
            return 0.0
    
    def can_execute_buy(self, 
                       stock: Stock,
                       portfolio: Portfolio) -> Tuple[bool, str]:
        """
        매수 실행 가능 여부 확인 (도메인 엔터티 사용)
        
        Args:
            stock: 주식 엔터티
            portfolio: 포트폴리오 엔터티
            
        Returns:
            Tuple[bool, str]: (실행 가능 여부, 사유)
        """
        try:
            # 주식 매매 가능 여부
            if not stock.is_tradeable():
                return False, f"{stock.code} 매매 불가능 상태"
            
            # 매수 수량 계산
            buy_percentage = self.config.trading.buy_percentage
            quantity = OrderDomainService.calculate_buy_quantity(
                portfolio.cash, stock.current_price, buy_percentage
            )
            
            if quantity.is_zero():
                return False, "매수 가능 수량이 0"
            
            # 도메인 주문 검증
            if not OrderDomainService.validate_buy_order(
                portfolio, stock, quantity, stock.current_price
            ):
                return False, "도메인 주문 검증 실패"
            
            # 포지션 수 제한 확인
            if not portfolio.can_buy(stock.calculate_order_amount(quantity)):
                return False, "현금 부족"
            
            return True, "매수 가능"
            
        except Exception as e:
            return False, f"매수 가능성 검증 중 오류: {e}"
    
    def can_execute_sell(self, 
                        position: Position) -> Tuple[bool, str]:
        """
        매도 실행 가능 여부 확인 (도메인 엔터티 사용)
        
        Args:
            position: 포지션 엔터티
            
        Returns:
            Tuple[bool, str]: (실행 가능 여부, 사유)
        """
        try:
            # 포지션 유효성 확인
            if position.is_empty():
                return False, "보유 수량이 없음"
            
            # 주식 매매 가능 여부
            if not position.stock.is_tradeable():
                return False, f"{position.stock.code} 매매 불가능 상태"
            
            # 매도 신호 확인
            sell_signal = self._determine_sell_signal(position)
            if not sell_signal:
                return False, "매도 신호 없음"
            
            return True, f"매도 가능 ({sell_signal.reason})"
            
        except Exception as e:
            return False, f"매도 가능성 검증 중 오류: {e}"
    
    def generate_trading_signals(self, 
                               stocks: List[Stock]) -> List[TradingSignal]:
        """
        주식 리스트에 대한 매매 신호 생성
        
        Args:
            stocks: 주식 엔터티 리스트
            
        Returns:
            List[TradingSignal]: 생성된 매매 신호들
        """
        signals = []
        
        for stock in stocks:
            try:
                current_price = stock.current_price.value.to_float()
                
                # 매수 신호 확인
                if self.strategy.should_buy(stock.code, current_price):
                    signal = TradingSignal.create_buy_signal(
                        stock=stock,
                        strength=SignalStrength.MODERATE,
                        target_price=stock.current_price,
                        confidence=Percentage(75),
                        reason=f"{self.config.trading.strategy} 전략 매수 신호",
                        indicators={"strategy": self.config.trading.strategy}
                    )
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"{stock.code} 신호 생성 중 오류: {e}")
                continue
        
        return signals
    
    # 백워드 호환성을 위한 메서드들
    def execute_buy_order_legacy(self, 
                         code: str, 
                         current_price: float, 
                         available_cash: float,
                         current_positions_count: int) -> Tuple[bool, Dict[str, Any]]:
        """레거시 매수 주문 실행 (백워드 호환성)"""
        try:
            # Stock 엔터티 생성
            stock = Stock(
                code=code,
                name=code,
                current_price=Price.won(current_price),
                previous_close=Price.won(current_price)
            )
            
            # Portfolio 엔터티 생성 (임시)
            portfolio = Portfolio(cash=Money.won(available_cash))
            
            # 새로운 방식으로 실행
            return self.execute_buy_order(stock, portfolio)
            
        except Exception as e:
            logger.error(f"레거시 매수 주문 처리 중 오류: {e}")
            return False, {"reason": f"처리 오류: {e}"}
    
    def execute_sell_order_legacy(self, 
                          code: str, 
                          current_price: float, 
                          entry_price: float,
                          quantity: int) -> Tuple[bool, Dict[str, Any]]:
        """레거시 매도 주문 실행 (백워드 호환성)"""
        try:
            # Stock 엔터티 생성
            stock = Stock(
                code=code,
                name=code,
                current_price=Price.won(current_price),
                previous_close=Price.won(current_price)
            )
            
            # Position 엔터티 생성
            position = Position(
                stock=stock,
                quantity=Quantity(quantity),
                average_price=Price.won(entry_price)
            )
            
            # 새로운 방식으로 실행
            return self.execute_sell_order(position)
            
        except Exception as e:
            logger.error(f"레거시 매도 주문 처리 중 오류: {e}")
            return False, {"reason": f"처리 오류: {e}"} 