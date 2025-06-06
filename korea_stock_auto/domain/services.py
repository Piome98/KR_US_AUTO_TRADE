"""
한국 주식 자동매매 - 도메인 서비스

복잡한 비즈니스 로직을 처리하는 도메인 서비스들입니다.
여러 엔터티에 걸친 비즈니스 규칙이나 복잡한 계산을 담당합니다.
"""

from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import logging

from .entities import Stock, Order, Portfolio, Position, TradingSignal, OrderType, SignalType, SignalStrength
from .value_objects import Money, Price, Quantity, Percentage, DomainValidationError

logger = logging.getLogger(__name__)


class OrderDomainService:
    """
    주문 관련 도메인 서비스
    
    책임:
    - 주문 검증 로직
    - 주문 크기 계산
    - 주문 실행 가능 여부 판단
    """
    
    @staticmethod
    def validate_buy_order(portfolio: Portfolio, stock: Stock, quantity: Quantity, target_price: Price) -> bool:
        """
        매수 주문 검증
        
        Args:
            portfolio: 포트폴리오
            stock: 주식
            quantity: 수량
            target_price: 목표 가격
        
        Returns:
            bool: 주문 가능 여부
        """
        try:
            # 1. 주식 매매 가능 여부 확인
            if not stock.is_tradeable():
                logger.warning(f"{stock.code} 매매 불가능한 상태")
                return False
            
            # 2. 주문 금액 계산
            order_amount = Money.won(target_price.value.amount * quantity.value)
            
            # 3. 현금 충분성 확인
            if not portfolio.can_buy(order_amount):
                logger.warning(f"현금 부족: 필요 {order_amount}, 보유 {portfolio.cash}")
                return False
            
            # 4. 최소 주문 금액 확인 (예: 1만원 이상)
            min_order_amount = Money.won(10000)
            if order_amount < min_order_amount:
                logger.warning(f"최소 주문 금액 미만: {order_amount}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"매수 주문 검증 중 오류: {e}")
            return False
    
    @staticmethod
    def validate_sell_order(portfolio: Portfolio, stock: Stock, quantity: Quantity) -> bool:
        """
        매도 주문 검증
        
        Args:
            portfolio: 포트폴리오
            stock: 주식
            quantity: 수량
        
        Returns:
            bool: 주문 가능 여부
        """
        try:
            # 1. 포지션 보유 확인
            if not portfolio.has_position(stock.code):
                logger.warning(f"{stock.code} 보유 포지션 없음")
                return False
            
            # 2. 매도 가능 수량 확인
            position = portfolio.get_position(stock.code)
            if not position.can_sell(quantity):
                logger.warning(f"매도 수량 초과: 요청 {quantity}, 보유 {position.quantity}")
                return False
            
            # 3. 주식 매매 가능 여부 확인
            if not stock.is_tradeable():
                logger.warning(f"{stock.code} 매매 불가능한 상태")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"매도 주문 검증 중 오류: {e}")
            return False
    
    @staticmethod
    def calculate_buy_quantity(available_cash: Money, target_price: Price, buy_percentage: float) -> Quantity:
        """
        매수 수량 계산
        
        Args:
            available_cash: 가용 현금
            target_price: 목표 가격
            buy_percentage: 매수 비율 (0.0 ~ 1.0)
        
        Returns:
            Quantity: 계산된 매수 수량
        """
        try:
            # 투자할 금액 계산
            investment_amount = available_cash * buy_percentage
            
            # 수량 계산 (소수점 버림)
            quantity_value = int(investment_amount.amount / target_price.value.amount)
            
            return Quantity(quantity_value)
            
        except Exception as e:
            logger.error(f"매수 수량 계산 중 오류: {e}")
            return Quantity.zero()
    
    @staticmethod
    def calculate_position_size_limit(portfolio: Portfolio, max_position_ratio: float) -> Money:
        """
        단일 종목 최대 투자 한도 계산
        
        Args:
            portfolio: 포트폴리오
            max_position_ratio: 최대 포지션 비율 (0.0 ~ 1.0)
        
        Returns:
            Money: 단일 종목 최대 투자 금액
        """
        try:
            total_value = portfolio.total_value()
            max_position_value = total_value * max_position_ratio
            
            return max_position_value
            
        except Exception as e:
            logger.error(f"포지션 크기 한도 계산 중 오류: {e}")
            return Money.zero()


class PortfolioDomainService:
    """
    포트폴리오 관련 도메인 서비스
    
    책임:
    - 포트폴리오 리밸런싱
    - 리스크 계산
    - 성과 분석
    """
    
    @staticmethod
    def calculate_portfolio_metrics(portfolio: Portfolio) -> Dict[str, float]:
        """
        포트폴리오 지표 계산
        
        Args:
            portfolio: 포트폴리오
        
        Returns:
            Dict[str, float]: 포트폴리오 지표들
        """
        try:
            total_value = portfolio.total_value()
            total_cost = portfolio.total_cost()
            
            metrics = {
                'total_value': total_value.to_float(),
                'total_cost': total_cost.to_float(),
                'cash_ratio': float(portfolio.cash.amount / total_value.amount * 100) if not total_value.is_zero() else 0,
                'position_count': portfolio.get_position_count(),
                'exposure_ratio': float(portfolio.get_exposure_ratio().value)
            }
            
            # 총 손익률 계산
            if not total_cost.is_zero():
                total_return_pct = float((total_value.amount - total_cost.amount) / total_cost.amount * 100)
                metrics['total_return_pct'] = total_return_pct
            else:
                metrics['total_return_pct'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"포트폴리오 지표 계산 중 오류: {e}")
            return {}
    
    @staticmethod
    def get_position_weights(portfolio: Portfolio) -> Dict[str, float]:
        """
        포지션별 비중 계산
        
        Args:
            portfolio: 포트폴리오
        
        Returns:
            Dict[str, float]: 종목별 비중 (%)
        """
        try:
            total_value = portfolio.total_value()
            if total_value.is_zero():
                return {}
            
            weights = {}
            for stock_code, position in portfolio.positions.items():
                if not position.is_empty():
                    position_value = position.current_value()
                    weight = position_value.amount / total_value.amount * 100
                    weights[stock_code] = float(weight)
            
            return weights
            
        except Exception as e:
            logger.error(f"포지션 비중 계산 중 오류: {e}")
            return {}
    
    @staticmethod
    def find_overweight_positions(portfolio: Portfolio, max_weight: float) -> List[str]:
        """
        과중 투자된 종목 찾기
        
        Args:
            portfolio: 포트폴리오
            max_weight: 최대 허용 비중 (%)
        
        Returns:
            List[str]: 과중 투자된 종목 코드들
        """
        try:
            weights = PortfolioDomainService.get_position_weights(portfolio)
            
            overweight_stocks = []
            for stock_code, weight in weights.items():
                if weight > max_weight:
                    overweight_stocks.append(stock_code)
                    logger.warning(f"{stock_code} 과중 투자: {weight:.2f}% (최대: {max_weight}%)")
            
            return overweight_stocks
            
        except Exception as e:
            logger.error(f"과중 포지션 검색 중 오류: {e}")
            return []
    
    @staticmethod
    def calculate_diversification_score(portfolio: Portfolio) -> float:
        """
        분산투자 점수 계산 (0.0 ~ 1.0)
        
        Args:
            portfolio: 포트폴리오
        
        Returns:
            float: 분산투자 점수 (높을수록 잘 분산됨)
        """
        try:
            position_count = portfolio.get_position_count()
            if position_count <= 1:
                return 0.0
            
            weights = PortfolioDomainService.get_position_weights(portfolio)
            
            # 허핀달 지수 계산 (농도 측정)
            herfindahl_index = sum(weight ** 2 for weight in weights.values())
            
            # 분산투자 점수 (0~1, 1이 가장 분산)
            max_herfindahl = 10000  # 100%^2
            diversification_score = 1 - (herfindahl_index / max_herfindahl)
            
            return max(0.0, min(1.0, diversification_score))
            
        except Exception as e:
            logger.error(f"분산투자 점수 계산 중 오류: {e}")
            return 0.0


class RiskDomainService:
    """
    리스크 관리 도메인 서비스
    
    책임:
    - 리스크 지표 계산
    - 손실 한도 체크
    - 리스크 알림
    """
    
    @staticmethod
    def check_daily_loss_limit(portfolio: Portfolio, initial_value: Money, daily_loss_limit_pct: float) -> bool:
        """
        일일 손실 한도 체크
        
        Args:
            portfolio: 현재 포트폴리오
            initial_value: 당일 시작 자산
            daily_loss_limit_pct: 일일 손실 한도 (%)
        
        Returns:
            bool: 손실 한도 초과 여부
        """
        try:
            current_value = portfolio.total_value()
            
            if initial_value.is_zero():
                return False
            
            # 손실률 계산
            if current_value < initial_value:
                loss_amount = initial_value - current_value
                loss_pct = (loss_amount.amount / initial_value.amount * 100)
                is_exceeded = float(loss_pct) > daily_loss_limit_pct
                
                if is_exceeded:
                    logger.warning(f"일일 손실 한도 초과: {loss_pct:.2f}% (한도: {daily_loss_limit_pct}%)")
                
                return is_exceeded
            
            return False
            
        except Exception as e:
            logger.error(f"일일 손실 한도 체크 중 오류: {e}")
            return False
    
    @staticmethod
    def check_position_loss_limit(position: Position, loss_limit_pct: float) -> bool:
        """
        개별 포지션 손실 한도 체크
        
        Args:
            position: 포지션
            loss_limit_pct: 손실 한도 (%)
        
        Returns:
            bool: 손실 한도 초과 여부
        """
        try:
            if position.is_profitable():
                return False
            
            if position.is_losing():
                loss_pct = position.unrealized_pnl_percentage()
                # 손실인 경우 음수이므로 절댓값으로 비교
                is_exceeded = abs(float(loss_pct.value)) > loss_limit_pct
                
                if is_exceeded:
                    logger.warning(f"{position.stock.code} 포지션 손실 한도 초과: {loss_pct}% (한도: {loss_limit_pct}%)")
                
                return is_exceeded
            
            return False
            
        except Exception as e:
            logger.error(f"포지션 손실 한도 체크 중 오류: {e}")
            return False
    
    @staticmethod
    def calculate_var_estimate(portfolio: Portfolio, confidence_level: float = 0.95) -> Money:
        """
        간단한 VaR (Value at Risk) 추정
        
        Args:
            portfolio: 포트폴리오
            confidence_level: 신뢰수준 (기본: 95%)
        
        Returns:
            Money: 추정 VaR
        """
        try:
            # 간단한 VaR 계산 (실제로는 더 복잡한 모델 필요)
            total_value = portfolio.total_value()
            
            # 포트폴리오 변동성을 대략 20%로 가정 (실제로는 과거 데이터 기반 계산)
            estimated_volatility = 0.20
            
            # 정규분포 가정하에 95% VaR ≈ 1.645 * σ * V
            z_score = 1.645 if confidence_level == 0.95 else 2.33  # 99%의 경우
            
            var_amount = total_value * estimated_volatility * z_score
            
            return var_amount
            
        except Exception as e:
            logger.error(f"VaR 계산 중 오류: {e}")
            return Money.zero()
    
    @staticmethod
    def assess_portfolio_risk_level(portfolio: Portfolio) -> str:
        """
        포트폴리오 리스크 수준 평가
        
        Args:
            portfolio: 포트폴리오
        
        Returns:
            str: 리스크 수준 ("LOW", "MEDIUM", "HIGH")
        """
        try:
            # 1. 노출도 확인
            exposure = portfolio.get_exposure_ratio()
            
            # 2. 분산투자 점수 확인
            diversification = PortfolioDomainService.calculate_diversification_score(portfolio)
            
            # 3. 포지션 수 확인
            position_count = portfolio.get_position_count()
            
            # 리스크 점수 계산 (0~100)
            risk_score = 0
            
            # 노출도 점수 (높을수록 위험)
            if float(exposure.value) > 80:
                risk_score += 40
            elif float(exposure.value) > 60:
                risk_score += 25
            else:
                risk_score += 10
            
            # 분산투자 점수 (낮을수록 위험)
            if diversification < 0.3:
                risk_score += 30
            elif diversification < 0.6:
                risk_score += 15
            else:
                risk_score += 5
            
            # 포지션 수 점수 (적을수록 위험)
            if position_count <= 2:
                risk_score += 30
            elif position_count <= 5:
                risk_score += 15
            else:
                risk_score += 5
            
            # 리스크 수준 결정
            if risk_score >= 70:
                return "HIGH"
            elif risk_score >= 40:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception as e:
            logger.error(f"포트폴리오 리스크 평가 중 오류: {e}")
            return "MEDIUM"


class TradingSignalDomainService:
    """
    매매 신호 관련 도메인 서비스
    
    책임:
    - 신호 검증
    - 신호 우선순위 결정
    - 신호 조합 로직
    """
    
    @staticmethod
    def validate_signal(signal: TradingSignal) -> bool:
        """
        매매 신호 검증
        
        Args:
            signal: 매매 신호
        
        Returns:
            bool: 신호 유효성
        """
        try:
            # 1. 기본 유효성 체크
            if not signal.is_valid():
                return False
            
            # 2. 주식 매매 가능 여부
            if not signal.stock.is_tradeable():
                logger.warning(f"{signal.stock.code} 매매 불가능한 상태")
                return False
            
            # 3. 목표가 합리성 체크
            current_price = signal.stock.current_price
            price_diff_pct = abs(float(signal.price_target_percentage().value))
            
            # 목표가와 현재가 차이가 너무 크면 비현실적
            if price_diff_pct > 20:  # 20% 이상 차이
                logger.warning(f"{signal.stock.code} 목표가 차이 과다: {price_diff_pct:.2f}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"신호 검증 중 오류: {e}")
            return False
    
    @staticmethod
    def prioritize_signals(signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        매매 신호 우선순위 정렬
        
        Args:
            signals: 매매 신호 리스트
        
        Returns:
            List[TradingSignal]: 우선순위순으로 정렬된 신호들
        """
        try:
            # 우선순위 점수 계산 함수
            def calculate_priority_score(signal: TradingSignal) -> float:
                score = 0.0
                
                # 1. 신호 강도 점수
                if signal.strength == SignalStrength.STRONG:
                    score += 3.0
                elif signal.strength == SignalStrength.MODERATE:
                    score += 2.0
                else:  # WEAK
                    score += 1.0
                
                # 2. 신뢰도 점수
                score += float(signal.confidence.value) / 100 * 2.0
                
                # 3. 신호 신선도 (최근일수록 높은 점수)
                signal_age_hours = (datetime.now() - signal.created_at).total_seconds() / 3600
                freshness_score = max(0, 1.0 - signal_age_hours / 24)  # 24시간 후 0점
                score += freshness_score
                
                return score
            
            # 점수 기준으로 내림차순 정렬
            return sorted(signals, key=calculate_priority_score, reverse=True)
            
        except Exception as e:
            logger.error(f"신호 우선순위 정렬 중 오류: {e}")
            return signals
    
    @staticmethod
    def filter_conflicting_signals(signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        상충하는 신호 필터링 (같은 종목에 대한 서로 다른 신호)
        
        Args:
            signals: 매매 신호 리스트
        
        Returns:
            List[TradingSignal]: 필터링된 신호들
        """
        try:
            # 종목별로 그룹화
            signals_by_stock: Dict[str, List[TradingSignal]] = {}
            for signal in signals:
                stock_code = signal.stock.code
                if stock_code not in signals_by_stock:
                    signals_by_stock[stock_code] = []
                signals_by_stock[stock_code].append(signal)
            
            filtered_signals = []
            
            # 각 종목별로 가장 강한 신호만 선택
            for stock_code, stock_signals in signals_by_stock.items():
                if len(stock_signals) == 1:
                    filtered_signals.extend(stock_signals)
                else:
                    # 우선순위 정렬 후 첫 번째만 선택
                    prioritized = TradingSignalDomainService.prioritize_signals(stock_signals)
                    filtered_signals.append(prioritized[0])
                    
                    logger.info(f"{stock_code} 신호 충돌 해결: {len(stock_signals)}개 중 1개 선택")
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"신호 충돌 해결 중 오류: {e}")
            return signals
    
    @staticmethod
    def combine_signal_confidence(primary_signal: TradingSignal, 
                                 supporting_signals: List[TradingSignal]) -> Percentage:
        """
        주 신호와 보조 신호들을 조합하여 종합 신뢰도 계산
        
        Args:
            primary_signal: 주 신호
            supporting_signals: 보조 신호들
        
        Returns:
            Percentage: 종합 신뢰도
        """
        try:
            combined_confidence = primary_signal.confidence.value
            
            # 같은 방향의 보조 신호들만 고려
            same_direction_signals = [
                s for s in supporting_signals 
                if s.signal_type == primary_signal.signal_type
            ]
            
            # 보조 신호들의 가중 평균으로 신뢰도 보정
            if same_direction_signals:
                supporting_confidence = sum(
                    float(s.confidence.value) for s in same_direction_signals
                ) / len(same_direction_signals)
                
                # 보조 신호의 10% 가중치로 반영
                boost = supporting_confidence * 0.1
                combined_confidence = min(100, combined_confidence + boost)
            
            return Percentage(combined_confidence)
            
        except Exception as e:
            logger.error(f"신호 신뢰도 조합 중 오류: {e}")
            return primary_signal.confidence 