"""
한국 주식 자동매매 - 포트폴리오 서비스

포트폴리오 상태 관리를 담당합니다:
- 계좌 정보 조회 및 업데이트
- 보유 종목 관리
- 포트폴리오 상태 추적
- 자산 평가
"""

import logging
import time
from typing import Dict, List, Any, Optional

from korea_stock_auto.config import AppConfig
from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.domain import Portfolio, Stock, Position
from korea_stock_auto.domain import Money, Price, Quantity
from korea_stock_auto.domain import PortfolioDomainService

# API 매퍼 통합
from korea_stock_auto.api.mappers import (
    PortfolioMapper, AccountMapper, StockMapper, MappingError
)

logger = logging.getLogger(__name__)


class PortfolioService:
    """포트폴리오 관리 서비스 (도메인 엔터티 통합 + API 매퍼 통합)"""
    
    def __init__(self, api: KoreaInvestmentApiClient, config: AppConfig):
        """
        포트폴리오 서비스 초기화
        
        Args:
            api: 한국투자증권 API 클라이언트
            config: 애플리케이션 설정
        """
        self.api = api
        self.config = config
        
        # API 매퍼 초기화
        self.portfolio_mapper = PortfolioMapper(
            enable_cache=True, 
            cache_ttl_seconds=120  # 포트폴리오 데이터는 2분 캐시
        )
        self.account_mapper = AccountMapper(
            enable_cache=True,
            cache_ttl_seconds=60   # 계좌 데이터는 1분 캐시
        )
        self.stock_mapper = StockMapper(
            enable_cache=True,
            cache_ttl_seconds=30   # 주식 데이터는 30초 캐시
        )
        
        # 도메인 엔터티
        self.portfolio: Portfolio = Portfolio(cash=Money.zero())
        
        # 백워드 호환성을 위한 기존 인터페이스 유지
        self.total_cash: float = 0
        self.bought_list: List[str] = []
        self.stock_dict: Dict[str, Dict[str, Any]] = {}
        self.entry_prices: Dict[str, float] = {}
        
        # 업데이트 관리
        self.last_account_update_time: float = 0
        self.account_update_interval: int = config.system.account_update_interval
        
        logger.debug("PortfolioService 초기화 완료 (도메인 엔터티 + API 매퍼 통합)")
    
    def initialize(self) -> bool:
        """
        포트폴리오 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.update_account_info(force=True)
            logger.info(f"포트폴리오 초기화 완료: 현금 {self.portfolio.cash}, 보유 종목 {self.portfolio.get_position_count()}개")
            return True
        except Exception as e:
            logger.error(f"포트폴리오 초기화 실패: {e}")
            return False
    
    def update_account_info(self, force: bool = False) -> bool:
        """
        계좌 정보 및 보유 종목 정보 업데이트 (API 매퍼 통합)
        
        Args:
            force: 강제 업데이트 여부 (interval 무시)
            
        Returns:
            bool: 업데이트 성공 여부
        """
        current_time = time.time()
        
        if not force and current_time - self.last_account_update_time < self.account_update_interval:
            return True
        
        try:
            # 계좌 잔고 조회 (API 호출)
            balance_info = self.api.get_account_balance()
            if not balance_info:
                logger.warning("계좌 잔고 정보 조회 실패")
                return False
            
            # 매퍼를 통한 Portfolio 엔터티 생성
            try:
                self.portfolio = self.portfolio_mapper.map_from_balance_response(balance_info)
                
                # 백워드 호환성을 위한 데이터 동기화
                self.total_cash = self.portfolio.cash.to_float()
                self._sync_legacy_data_from_portfolio()
                
                logger.debug(f"Portfolio 엔터티 업데이트 완료: 현금 {self.portfolio.cash}, 보유종목 {self.portfolio.get_position_count()}개")
                
            except MappingError as e:
                logger.error(f"Portfolio 매핑 실패: {e}")
                # 백워드 호환성: 기존 방식으로 처리
                return self._update_account_info_legacy(balance_info)
            
            self.last_account_update_time = current_time
            return True
            
        except Exception as e:
            logger.error(f"계좌 정보 업데이트 실패: {e}")
            return False
    
    def _sync_legacy_data_from_portfolio(self) -> None:
        """Portfolio 엔터티에서 백워드 호환성 데이터 동기화"""
        self.bought_list.clear()
        self.stock_dict.clear()
        self.entry_prices.clear()
        
        for code, position in self.portfolio.positions.items():
            # 매수 종목 리스트 업데이트
            self.bought_list.append(code)
            
            # 종목 정보 딕셔너리 업데이트 (기존 API 형식으로 변환)
            self.stock_dict[code] = {
                'code': position.stock.code,
                'prdt_name': position.stock.name,
                'hldg_qty': str(position.quantity.value),
                'pchs_avg_pric': str(int(position.average_price.value.to_float())),
                'prpr': str(int(position.stock.current_price.value.to_float())),
                'evlu_amt': str(int(position.current_value().to_float())),
                'pchs_amt': str(int(position.purchase_amount().to_float())),
                'evlu_pfls_amt': str(int(position.unrealized_profit_loss().to_float())),
                'evlu_pfls_rt': f"{position.unrealized_profit_loss_percentage():.2f}",
                'sll_able_qty': str(position.quantity.value)
            }
            
            # 진입가 업데이트
            self.entry_prices[code] = position.average_price.value.to_float()
    
    def _update_account_info_legacy(self, balance_info: Dict[str, Any]) -> bool:
        """기존 방식의 계좌 정보 업데이트 (백워드 호환성)"""
        try:
            # 현금 정보 업데이트
            cash_amount = balance_info.get("cash", 0)
            self.portfolio.cash = Money.won(cash_amount)
            self.total_cash = cash_amount
            logger.debug(f"계좌 잔고 업데이트 (legacy): {self.portfolio.cash}")
            
            # 보유 종목 조회
            stock_balance = self.api.get_stock_balance()
            if stock_balance and "stocks" in stock_balance:
                self._update_stock_holdings_legacy(stock_balance["stocks"])
                logger.debug(f"보유 종목 업데이트 (legacy): {self.portfolio.get_position_count()}개")
                return True
            else:
                logger.warning("보유 종목 정보 조회 실패")
                return False
                
        except Exception as e:
            logger.error(f"Legacy 계좌 정보 업데이트 실패: {e}")
            return False
    
    def _update_stock_holdings(self, stocks: List[Dict[str, Any]]) -> None:
        """
        보유 종목 정보 업데이트 (API 매퍼 통합 버전)
        
        Args:
            stocks: API에서 조회한 보유 종목 리스트
        """
        # 기존 포지션 백업
        previous_positions = dict(self.portfolio.positions)
        
        # 새로운 포트폴리오 데이터로 업데이트
        new_positions = {}
        updated_stock_dict = {}
        updated_bought_list = []
        updated_entry_prices = {}
        
        for stock_data in stocks:
            try:
                # StockMapper를 통한 Stock 엔터티 생성
                stock = self.stock_mapper.map_from_balance_response(stock_data)
                
                # Position 데이터 추출
                quantity_value = int(stock_data.get("hldg_qty", 0))
                if quantity_value <= 0:
                    continue
                
                avg_price_value = float(stock_data.get("pchs_avg_pric", 0))
                
                # Position 엔터티 생성
                position = Position(
                    stock=stock,
                    quantity=Quantity(quantity_value),
                    average_price=Price.won(avg_price_value)
                )
                
                new_positions[stock.code] = position
                
                # 백워드 호환성을 위한 데이터 유지
                updated_stock_dict[stock.code] = stock_data
                updated_bought_list.append(stock.code)
                updated_entry_prices[stock.code] = avg_price_value
                
                # 새로운 포지션 로깅
                if stock.code not in previous_positions:
                    logger.info(f"새로운 보유 종목 감지: {stock.code}({stock.name}), 진입가: {avg_price_value:,}")
                    
            except MappingError as e:
                logger.warning(f"종목 {stock_data.get('code', 'Unknown')} 매핑 실패: {e}")
                continue
            except Exception as e:
                logger.error(f"종목 {stock_data.get('code', 'Unknown')} 데이터 처리 중 오류: {e}")
                continue
        
        # 포트폴리오 업데이트
        self.portfolio.positions = new_positions
        
        # 백워드 호환성 데이터 업데이트
        self.stock_dict = updated_stock_dict
        self.bought_list = updated_bought_list
        self.entry_prices = updated_entry_prices
        
        # 매도된 종목 로깅
        sold_stocks = set(previous_positions.keys()) - set(new_positions.keys())
        for code in sold_stocks:
            logger.info(f"매도 완료된 종목 정리: {code}")
    
    def _update_stock_holdings_legacy(self, stocks: List[Dict[str, Any]]) -> None:
        """
        보유 종목 정보 업데이트 (기존 방식, 백워드 호환성)
        
        Args:
            stocks: API에서 조회한 보유 종목 리스트
        """
        # 기존 포지션 백업
        previous_positions = dict(self.portfolio.positions)
        
        # 새로운 포트폴리오 데이터로 업데이트
        new_positions = {}
        updated_stock_dict = {}
        updated_bought_list = []
        updated_entry_prices = {}
        
        for stock_data in stocks:
            code = stock_data.get("code")
            if not code:
                continue
                
            quantity = int(stock_data.get("hldg_qty", 0))
            if quantity <= 0:
                continue
                
            name = stock_data.get("prdt_name", code)
            avg_price = float(stock_data.get("pchs_avg_pric", 0))
            current_price = float(stock_data.get("prpr", avg_price))
            
            try:
                # Stock 엔터티 생성
                stock = Stock(
                    code=code,
                    name=name,
                    current_price=Price.won(current_price),
                    previous_close=Price.won(current_price)  # 정확한 전일종가는 별도 API 필요
                )
                
                # Position 엔터티 생성
                position = Position(
                    stock=stock,
                    quantity=Quantity(quantity),
                    average_price=Price.won(avg_price)
                )
                
                new_positions[code] = position
                
                # 백워드 호환성을 위한 데이터 유지
                updated_stock_dict[code] = stock_data
                updated_bought_list.append(code)
                updated_entry_prices[code] = avg_price
                
                # 새로운 포지션 로깅
                if code not in previous_positions:
                    logger.info(f"새로운 보유 종목 감지: {code}, 진입가: {avg_price:,}")
                    
            except Exception as e:
                logger.error(f"종목 {code} 데이터 처리 중 오류: {e}")
                continue
        
        # 포트폴리오 업데이트
        self.portfolio.positions = new_positions
        
        # 백워드 호환성 데이터 업데이트
        self.stock_dict = updated_stock_dict
        self.bought_list = updated_bought_list
        self.entry_prices = updated_entry_prices
        
        # 매도된 종목 로깅
        sold_stocks = set(previous_positions.keys()) - set(new_positions.keys())
        for code in sold_stocks:
            logger.info(f"매도 완료된 종목 정리: {code}")
    
    def add_position(self, position: Position) -> None:
        """
        새로운 포지션 추가 (도메인 엔터티 사용)
        
        Args:
            position: 포지션 엔터티
        """
        try:
            code = position.stock.code
            
            # 포트폴리오에 포지션 추가/업데이트
            self.portfolio.add_position(
                position.stock, 
                position.quantity, 
                position.average_price
            )
            
            # 백워드 호환성 데이터 동기화
            self._sync_legacy_data_from_portfolio()
            
            logger.info(f"포지션 추가 완료: {position}")
            
        except Exception as e:
            logger.error(f"포지션 추가 실패: {e}")
            raise
    
    def get_account_balance(self) -> Optional[Dict[str, Any]]:
        """
        계좌 잔고 정보 조회 (API 매퍼 통합)
        
        Returns:
            Dict: 계좌 잔고 정보 또는 None
        """
        try:
            # API에서 잔고 정보 조회
            balance_info = self.api.get_account_balance()
            if not balance_info:
                return None
            
            # AccountMapper를 통한 변환
            try:
                account_balance = self.account_mapper.map_from_balance_response(balance_info)
                
                return {
                    'cash': account_balance.cash.to_float(),
                    'stock_value': account_balance.stock_value.to_float(),
                    'total_assets': account_balance.total_assets.to_float(),
                    'total_profit_loss': account_balance.total_profit_loss.to_float(),
                    'profit_loss_rate': account_balance.profit_loss_rate,
                    'updated_at': account_balance.updated_at
                }
                
            except MappingError as e:
                logger.warning(f"AccountBalance 매핑 실패, 기존 방식 사용: {e}")
                # 백워드 호환성: 기존 방식으로 반환
                return balance_info
                
        except Exception as e:
            logger.error(f"계좌 잔고 조회 실패: {e}")
            return None
    
    def clear_mappers_cache(self) -> None:
        """매퍼 캐시 전체 삭제"""
        self.portfolio_mapper.clear_cache()
        self.account_mapper.clear_cache()
        self.stock_mapper.clear_cache()
        logger.debug("모든 매퍼 캐시 삭제 완료")
    
    def get_mappers_cache_stats(self) -> Dict[str, Any]:
        """매퍼 캐시 통계 조회"""
        return {
            'portfolio_mapper': self.portfolio_mapper.get_cache_stats(),
            'account_mapper': self.account_mapper.get_cache_stats(),
            'stock_mapper': self.stock_mapper.get_cache_stats()
        }
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        현재 가격을 반영한 포트폴리오 가치 계산 (도메인 엔터티 사용)
        
        Args:
            current_prices: 종목별 현재 가격
            
        Returns:
            Dict: 포트폴리오 가치 정보
        """
        try:
            # 주식 가격 업데이트
            for code, price in current_prices.items():
                position = self.portfolio.get_position(code)
                if position:
                    position.stock.update_price(Price.won(price))
            
            # 포트폴리오 지표 계산
            metrics = PortfolioDomainService.calculate_portfolio_metrics(self.portfolio)
            
            return {
                "total_value": metrics.get("total_value", 0),
                "cash": self.portfolio.cash.to_float(),
                "invested_value": metrics.get("total_value", 0) - self.portfolio.cash.to_float(),
                "total_return_pct": metrics.get("total_return_pct", 0),
                "position_count": metrics.get("position_count", 0),
                "exposure_ratio": metrics.get("exposure_ratio", 0)
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 가치 계산 중 오류: {e}")
            return {
                "total_value": self.total_cash,
                "cash": self.total_cash,
                "invested_value": 0,
                "total_return_pct": 0,
                "position_count": 0,
                "exposure_ratio": 0
            }
    
    def get_available_cash_for_buy(self) -> float:
        """
        매수 가능한 현금 반환
        
        Returns:
            float: 매수 가능 현금
        """
        return self.portfolio.cash.to_float()
    
    def can_add_position(self) -> bool:
        """
        새로운 포지션 추가 가능 여부 확인
        
        Returns:
            bool: 추가 가능 여부
        """
        current_count = self.portfolio.get_position_count()
        max_count = self.config.trading.target_buy_count
        return current_count < max_count
    
    def has_position(self, code: str) -> bool:
        """
        특정 종목 보유 여부 확인
        
        Args:
            code: 종목 코드
            
        Returns:
            bool: 보유 여부
        """
        return self.portfolio.has_position(code)
    
    def get_holdings_summary(self) -> Dict[str, Any]:
        """
        보유 종목 요약 정보 반환 (도메인 엔터티 사용)
        
        Returns:
            Dict: 보유 종목 요약
        """
        try:
            metrics = PortfolioDomainService.calculate_portfolio_metrics(self.portfolio)
            position_weights = PortfolioDomainService.get_position_weights(self.portfolio)
            
            return {
                "position_count": metrics.get("position_count", 0),
                "total_value": metrics.get("total_value", 0),
                "cash_ratio": metrics.get("cash_ratio", 0),
                "exposure_ratio": metrics.get("exposure_ratio", 0),
                "position_weights": position_weights,
                "holdings": [pos.stock.code for pos in self.portfolio.positions.values()]
            }
            
        except Exception as e:
            logger.error(f"보유 종목 요약 중 오류: {e}")
            return {
                "position_count": len(self.bought_list),
                "total_value": self.total_cash,
                "cash_ratio": 100.0,
                "exposure_ratio": 0.0,
                "position_weights": {},
                "holdings": self.bought_list
            }
    
    def update_portfolio(self) -> bool:
        """
        포트폴리오 업데이트 (계좌 정보 갱신)
        
        Returns:
            bool: 업데이트 성공 여부
        """
        return self.update_account_info()
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        포트폴리오 현재 상태 반환 (도메인 엔터티 사용)
        
        Returns:
            Dict: 포트폴리오 상태
        """
        try:
            return {
                "cash": self.portfolio.cash.to_float(),
                "total_value": self.portfolio.total_value().to_float(),
                "position_count": self.portfolio.get_position_count(),
                "exposure_ratio": self.portfolio.get_exposure_ratio().value,
                "positions": {
                    code: {
                        "name": pos.stock.name,
                        "quantity": pos.quantity.value,
                        "avg_price": pos.average_price.value.to_float(),
                        "current_price": pos.stock.current_price.value.to_float(),
                        "unrealized_pnl_pct": pos.unrealized_pnl_percentage().value
                    }
                    for code, pos in self.portfolio.positions.items()
                }
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 상태 조회 중 오류: {e}")
            return {
                "cash": self.total_cash,
                "total_value": self.total_cash,
                "position_count": len(self.bought_list),
                "exposure_ratio": 0.0,
                "positions": {}
            }
    
    # 도메인 엔터티 직접 접근 메서드들
    def get_domain_portfolio(self) -> Portfolio:
        """도메인 Portfolio 엔터티 반환"""
        return self.portfolio
    
    def get_domain_position(self, code: str) -> Optional[Position]:
        """특정 종목의 도메인 Position 엔터티 반환"""
        return self.portfolio.get_position(code)
    
    def add_position_legacy(self, code: str, entry_price: float, quantity: int = 0) -> None:
        """
        새로운 포지션 추가 (Legacy 백워드 호환성)
        
        Args:
            code: 종목 코드
            entry_price: 진입가
            quantity: 수량 (선택적)
        """
        try:
            # Stock 엔터티 생성 (기본값으로)
            stock = Stock(
                code=code,
                name=code,  # 이름은 추후 API에서 업데이트
                current_price=Price.won(entry_price),
                previous_close=Price.won(entry_price)
            )
            
            # Position 엔터티 생성
            position = Position(
                stock=stock,
                quantity=Quantity(quantity) if quantity > 0 else Quantity.zero(),
                average_price=Price.won(entry_price)
            )
            
            # 새로운 메서드 호출
            self.add_position(position)
            
        except Exception as e:
            logger.error(f"Legacy 포지션 추가 중 오류: {e}")
    
    def remove_position(self, code: str) -> bool:
        """
        포지션 제거 (도메인 엔터티 사용)
        
        Args:
            code: 종목 코드
        
        Returns:
            bool: 제거 성공 여부
        """
        try:
            # 도메인 포트폴리오에서 제거
            if code in self.portfolio.positions:
                del self.portfolio.positions[code]
            
            # 백워드 호환성 데이터 제거
            if code in self.bought_list:
                self.bought_list.remove(code)
            if code in self.entry_prices:
                del self.entry_prices[code]
            if code in self.stock_dict:
                del self.stock_dict[code]
            
            logger.info(f"포지션 제거: {code}")
            return True
                
        except Exception as e:
            logger.error(f"포지션 제거 중 오류: {e}")
            return False
    
    def get_position_info(self, code: str) -> Optional[Dict[str, Any]]:
        """
        특정 종목의 포지션 정보 조회 (도메인 엔터티 사용)
        
        Args:
            code: 종목 코드
            
        Returns:
            Dict: 포지션 정보 (없으면 None)
        """
        position = self.portfolio.get_position(code)
        if position is None:
            return None
        
        return {
            "code": code,
            "name": position.stock.name,
            "quantity": position.quantity.value,
            "entry_price": position.average_price.value.to_float(),
            "current_price": position.stock.current_price.value.to_float(),
            "total_cost": position.total_cost().to_float(),
            "current_value": position.current_value().to_float(),
            "unrealized_pnl": position.unrealized_pnl().to_float(),
            "unrealized_pnl_pct": position.unrealized_pnl_percentage().value,
            "is_profitable": position.is_profitable()
        } 