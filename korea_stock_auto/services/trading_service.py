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

# API 매퍼 통합
from korea_stock_auto.api.mappers import (
    OrderMapper, AccountMapper, StockMapper, MappingError
)

logger = logging.getLogger(__name__)


class TradingService:
    """매매 실행 서비스 (도메인 엔터티 통합 + API 매퍼 통합)"""
    
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
        
        # API 매퍼 초기화
        self.order_mapper = OrderMapper(
            enable_cache=False,  # 주문은 실시간성이 중요하므로 캐시 비활성화
            cache_ttl_seconds=30
        )
        self.account_mapper = AccountMapper(
            enable_cache=True,
            cache_ttl_seconds=60
        )
        self.stock_mapper = StockMapper(
            enable_cache=True,
            cache_ttl_seconds=30
        )
        
        logger.debug("TradingService 초기화 완료 (도메인 엔터티 + API 매퍼 통합)")
    
    def execute_buy_order(self, 
                         stock: Stock,
                         portfolio: Portfolio) -> Tuple[bool, Dict[str, Any]]:
        """
        매수 주문 실행 (도메인 엔터티 사용 + API 매퍼 통합)
        
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
            
            # 매수 주문 실행 (API 호출)
            success, order_details = self.executor.place_order(
                stock.code, "buy", stock.current_price.value.to_float(), 
                order_cash=order_amount.to_float()
            )
            
            # OrderMapper를 통한 Order 엔터티 생성
            try:
                if success and order_details:
                    # 주문 성공 시 OrderMapper 사용
                    order = self.order_mapper.map_from_order_submit_response(
                        order_details, stock, OrderType.BUY, 
                        quantity.value, int(stock.current_price.value.to_float())
                    )
                    
                    logger.info(f"{stock.code} 매수 주문 성공 (매퍼): {order}")
                    send_message(f"[매수 성공] {stock.code} 가격: {order.target_price} ({quantity}주)", 
                               self.config.notification.discord_webhook_url)
                    
                    return True, {
                        "order_id": str(order.id),
                        "quantity": order.quantity.value,
                        "price": order.target_price.value.to_float(),
                        "amount": order.quantity.value * order.target_price.value.to_float(),
                        "order": order,
                        "api_order_id": order.error_message.replace("API_ORDER_ID:", "") if order.error_message else None
                    }
                else:
                    # 주문 실패 시 수동으로 Order 엔터티 생성
                    order = Order.create_buy_order(stock, quantity, stock.current_price)
                    order.reject(f"주문 실행 실패: {order_details}")
                    
                    logger.error(f"{stock.code} 매수 주문 실패: {order_details}")
                    send_message(f"[매수 실패] {stock.code} 가격: {stock.current_price} ({order_details})", 
                               self.config.notification.discord_webhook_url)
                    return False, {"reason": order_details, "order": order}
                    
            except MappingError as e:
                logger.warning(f"Order 매핑 실패, 수동 생성: {e}")
                # 백워드 호환성: 수동으로 Order 엔터티 생성
                return self._execute_buy_order_legacy(stock, portfolio, quantity, order_details)
                
        except Exception as e:
            logger.error(f"{stock.code} 매수 처리 중 오류: {e}")
            return False, {"reason": f"매수 처리 오류: {e}"}
    
    def _execute_buy_order_legacy(self, stock: Stock, portfolio: Portfolio, 
                                 quantity: Quantity, order_details: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """매수 주문 실행 (기존 방식, 백워드 호환성)"""
        try:
            order = Order.create_buy_order(stock, quantity, stock.current_price)
            
            if order_details and order_details.get('success', False):
                filled_quantity = Quantity(order_details.get('quantity', quantity.value))
                filled_price = Price.won(order_details.get('price', stock.current_price.value.to_float()))
                order.submit()
                order.fill(filled_quantity, filled_price)
                
                return True, {
                    "order_id": str(order.id),
                    "quantity": filled_quantity.value,
                    "price": filled_price.value.to_float(),
                    "amount": order.filled_amount().to_float(),
                    "order": order
                }
            else:
                order.reject(f"주문 실행 실패: {order_details}")
                return False, {"reason": order_details, "order": order}
                
        except Exception as e:
            logger.error(f"Legacy 매수 주문 처리 실패: {e}")
            return False, {"reason": f"Legacy 매수 처리 오류: {e}"}
    
    def execute_sell_order(self, 
                          position: Position) -> Tuple[bool, Dict[str, Any]]:
        """
        매도 주문 실행 (도메인 엔터티 사용 + API 매퍼 통합)
        
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
            
            # 매도 주문 실행 (API 호출)
            success, order_details = self.executor.place_order(
                stock.code, "sell", stock.current_price.value.to_float(), 
                quantity=position.quantity.value
            )
            
            # OrderMapper를 통한 Order 엔터티 생성
            try:
                if success and order_details:
                    # 주문 성공 시 OrderMapper 사용
                    order = self.order_mapper.map_from_order_submit_response(
                        order_details, stock, OrderType.SELL, 
                        position.quantity.value, int(stock.current_price.value.to_float())
                    )
                    
                    logger.info(f"{stock.code} 매도 주문 성공 (매퍼): {order} (사유: {sell_signal.reason})")
                    send_message(f"[매도 성공] {stock.code} 가격: {order.target_price} ({position.quantity}주) - {sell_signal.reason}", 
                               self.config.notification.discord_webhook_url)
                    
                    return True, {
                        "order_id": str(order.id),
                        "quantity": order.quantity.value,
                        "price": order.target_price.value.to_float(),
                        "amount": order.quantity.value * order.target_price.value.to_float(),
                        "reason": sell_signal.reason,
                        "order": order,
                        "api_order_id": order.error_message.replace("API_ORDER_ID:", "") if order.error_message else None
                    }
                else:
                    # 주문 실패 시 수동으로 Order 엔터티 생성
                    order = Order.create_sell_order(stock, position.quantity, stock.current_price)
                    order.reject(f"주문 실행 실패: {order_details}")
                    
                    logger.error(f"{stock.code} 매도 주문 실패: {order_details}")
                    send_message(f"[매도 실패] {stock.code} 가격: {stock.current_price} ({order_details})", 
                               self.config.notification.discord_webhook_url)
                    return False, {"reason": order_details, "order": order}
                    
            except MappingError as e:
                logger.warning(f"Order 매핑 실패, 수동 생성: {e}")
                # 백워드 호환성: 수동으로 Order 엔터티 생성
                return self._execute_sell_order_legacy(position, sell_signal, order_details)
                
        except Exception as e:
            logger.error(f"{position.stock.code} 매도 처리 중 오류: {e}")
            return False, {"reason": f"매도 처리 오류: {e}"}
    
    def _execute_sell_order_legacy(self, position: Position, sell_signal: TradingSignal, 
                                  order_details: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """매도 주문 실행 (기존 방식, 백워드 호환성)"""
        try:
            stock = position.stock
            order = Order.create_sell_order(stock, position.quantity, stock.current_price)
            
            if order_details and order_details.get('success', False):
                filled_quantity = Quantity(order_details.get('quantity', position.quantity.value))
                filled_price = Price.won(order_details.get('price', stock.current_price.value.to_float()))
                order.submit()
                order.fill(filled_quantity, filled_price)
                
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
                return False, {"reason": order_details, "order": order}
                
        except Exception as e:
            logger.error(f"Legacy 매도 주문 처리 실패: {e}")
            return False, {"reason": f"Legacy 매도 처리 오류: {e}"}
    
    def get_buyable_amount(self, stock_code: str, stock_price: int) -> Optional[Dict[str, Any]]:
        """
        매수 가능 금액 조회 (API 매퍼 통합)
        
        Args:
            stock_code: 종목 코드
            stock_price: 주식 가격
            
        Returns:
            Dict: 매수 가능 정보 또는 None
        """
        try:
            # API 호출
            buyable_response = self.api.get_buyable_amount(stock_code, stock_price)
            if not buyable_response:
                return None
            
            # AccountMapper를 통한 변환
            try:
                # API 응답에 요청 정보 추가 (매퍼에서 필요)
                buyable_response['stock_code'] = stock_code
                buyable_response['stock_price'] = stock_price
                
                buyable_amount = self.account_mapper.map_single(buyable_response)
                
                return {
                    'cash': buyable_amount.cash.to_float(),
                    'max_buy_amount': buyable_amount.max_buy_amount.to_float(),
                    'stock_code': buyable_amount.stock_code,
                    'stock_price': buyable_amount.stock_price,
                    'max_quantity': buyable_amount.max_quantity
                }
                
            except MappingError as e:
                logger.warning(f"BuyableAmount 매핑 실패, 기존 방식 사용: {e}")
                # 백워드 호환성: 기존 방식으로 반환
                return buyable_response
                
        except Exception as e:
            logger.error(f"매수 가능 금액 조회 실패: {e}")
            return None
    
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
    
    def execute_portfolio_rebalancing(self, 
                                   portfolio: Portfolio, 
                                   target_allocations: Dict[str, float]) -> Dict[str, Any]:
        """
        포트폴리오 리밸런싱 실행 (도메인 엔터티 사용)
        
        Args:
            portfolio: 포트폴리오 엔터티
            target_allocations: 목표 비중 (종목코드: 비중%)
            
        Returns:
            Dict: 리밸런싱 결과
        """
        try:
            logger.info("포트폴리오 리밸런싱 시작")
            
            total_value = portfolio.total_value()
            results = {
                "total_actions": 0,
                "buy_actions": 0,
                "sell_actions": 0,
                "errors": []
            }
            
            for code, target_weight in target_allocations.items():
                try:
                    target_amount = Money.won(total_value.amount * (target_weight / 100))
                    current_position = portfolio.get_position(code)
                    
                    if current_position and not current_position.is_empty():
                        current_value = current_position.current_value()
                        
                        # 목표 금액과 현재 금액 차이 계산
                        difference = target_amount.amount - current_value.amount
                        
                        # 임계값 이상의 차이가 있을 때만 리밸런싱
                        threshold = total_value.amount * 0.05  # 5% 임계값
                        
                        if abs(difference) > threshold:
                            if difference > 0:
                                # 매수 필요
                                logger.info(f"{code} 추가 매수 필요: {difference:,.0f}원")
                                results["buy_actions"] += 1
                            else:
                                # 매도 필요
                                sell_ratio = abs(difference) / current_value.amount
                                sell_quantity = Quantity(int(current_position.quantity.value * sell_ratio))
                                
                                if sell_quantity.value > 0:
                                    logger.info(f"{code} 부분 매도: {sell_quantity}주")
                                    results["sell_actions"] += 1
                    
                    results["total_actions"] += 1
                    
                except Exception as e:
                    error_msg = f"{code} 리밸런싱 중 오류: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            logger.info(f"리밸런싱 완료: 총 {results['total_actions']}개 종목 처리")
            return results
            
        except Exception as e:
            logger.error(f"포트폴리오 리밸런싱 중 오류: {e}")
            return {"error": str(e)}
    
    def calculate_optimal_portfolio(self, 
                                  available_cash: Money, 
                                  stocks: List[Stock]) -> Portfolio:
        """
        최적 포트폴리오 계산 (도메인 엔터티 사용)
        
        Args:
            available_cash: 사용 가능한 현금
            stocks: 고려할 주식 리스트
            
        Returns:
            Portfolio: 최적화된 포트폴리오
        """
        try:
            logger.info(f"최적 포트폴리오 계산 시작 (현금: {available_cash}, 종목: {len(stocks)}개)")
            
            # 새로운 포트폴리오 생성
            optimal_portfolio = Portfolio(cash=available_cash)
            
            # 매매 신호가 있는 종목들 필터링
            buy_signals = self.generate_trading_signals(stocks)
            
            if not buy_signals:
                logger.info("매수 신호가 있는 종목이 없음")
                return optimal_portfolio
            
            # 신호 강도에 따라 정렬
            sorted_signals = sorted(buy_signals, 
                                  key=lambda s: (s.strength.value, s.confidence.value), 
                                  reverse=True)
            
            # 목표 종목 수
            target_count = min(self.config.trading.target_buy_count, len(sorted_signals))
            
            # 종목당 할당 금액
            allocation_per_stock = available_cash.amount / target_count
            
            for signal in sorted_signals[:target_count]:
                try:
                    stock = signal.stock
                    max_amount = Money.won(allocation_per_stock)
                    
                    # 최적 수량 계산
                    quantity = OrderDomainService.calculate_buy_quantity(
                        max_amount, stock.current_price, 100  # 100% 할당
                    )
                    
                    if not quantity.is_zero():
                        optimal_portfolio.add_position(stock, quantity, stock.current_price)
                        logger.info(f"최적 포트폴리오에 추가: {stock.code} {quantity}주")
                
                except Exception as e:
                    logger.error(f"{signal.stock.code} 최적 포트폴리오 계산 중 오류: {e}")
                    continue
            
            logger.info(f"최적 포트폴리오 계산 완료: {optimal_portfolio.get_position_count()}개 종목")
            return optimal_portfolio
            
        except Exception as e:
            logger.error(f"최적 포트폴리오 계산 중 오류: {e}")
            return Portfolio(cash=available_cash)
    
    def clear_mappers_cache(self) -> None:
        """매퍼 캐시 전체 삭제"""
        self.order_mapper.clear_cache()
        self.account_mapper.clear_cache()
        self.stock_mapper.clear_cache()
        logger.debug("모든 매퍼 캐시 삭제 완료")
    
    def get_mappers_cache_stats(self) -> Dict[str, Any]:
        """매퍼 캐시 통계 조회"""
        return {
            'order_mapper': self.order_mapper.get_cache_stats(),
            'account_mapper': self.account_mapper.get_cache_stats(),
            'stock_mapper': self.stock_mapper.get_cache_stats()
        }
    
    def query_orders(self, date: Optional[str] = None) -> List[Order]:
        """
        주문 조회 (API 매퍼 통합)
        
        Args:
            date: 조회 날짜 (YYYYMMDD, None이면 당일)
            
        Returns:
            List[Order]: 주문 엔터티 리스트
        """
        try:
            # API 호출
            orders_response = self.api.get_orders(date)
            if not orders_response or 'output' not in orders_response:
                return []
            
            orders = []
            order_list = orders_response['output']
            
            if isinstance(order_list, list):
                for order_data in order_list:
                    try:
                        # OrderMapper를 통한 Order 엔터티 생성
                        order = self.order_mapper.map_from_order_inquiry_response({
                            'output': [order_data]
                        })
                        orders.append(order)
                        
                    except MappingError as e:
                        logger.warning(f"주문 매핑 실패: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"주문 데이터 처리 중 오류: {e}")
                        continue
            
            logger.debug(f"주문 조회 완료: {len(orders)}건")
            return orders
            
        except Exception as e:
            logger.error(f"주문 조회 실패: {e}")
            return []
    
    def create_stock_from_market_data(self, code: str, current_price: float) -> Stock:
        """
        시장 데이터로부터 Stock 엔터티 생성 (유틸리티 메서드)
        
        Args:
            code: 종목 코드
            current_price: 현재가
            
        Returns:
            Stock: Stock 엔터티
        """
        try:
            # StockMapper를 통한 Stock 엔터티 생성
            stock_data = {
                'stock_data': {
                    'code': code,
                    'name': code,  # 기본값, 추후 API에서 업데이트
                    'current_price': current_price,
                    'previous_close': current_price,
                    'volume': 0,
                    'market_cap': 0
                }
            }
            
            stock = self.stock_mapper.map_single(stock_data)
            logger.debug(f"Stock 엔터티 생성: {code} @ {current_price}")
            return stock
            
        except Exception as e:
            logger.error(f"Stock 엔터티 생성 실패: {e}")
            # 백워드 호환성: 직접 생성
            return Stock(
                code=code,
                name=code,
                current_price=Price.won(current_price),
                previous_close=Price.won(current_price)
            )
    
    # Legacy 메서드들이 제거되었습니다.
    # 모든 호출은 도메인 엔터티를 사용하는 execute_buy_order, execute_sell_order를 사용해주세요. 