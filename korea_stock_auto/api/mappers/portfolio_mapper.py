"""
포트폴리오 API 응답 매퍼

계좌 잔고 API 응답을 Portfolio, Position 도메인 엔터티로 변환합니다:
- 계좌 잔고 조회 응답
- 보유 종목 정보
- 포트폴리오 상태
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .base_mapper import BaseMapper, MappingError
from .stock_mapper import StockMapper
from korea_stock_auto.domain.entities import Portfolio, Position, Stock
from korea_stock_auto.domain.value_objects import Money, Price, Quantity

logger = logging.getLogger(__name__)


class PortfolioMapper(BaseMapper[Portfolio]):
    """포트폴리오 API 응답 → Portfolio 엔터티 변환 매퍼"""
    
    def __init__(self, enable_cache: bool = True, cache_ttl_seconds: int = 120):
        """
        포트폴리오 매퍼 초기화
        
        Args:
            enable_cache: 캐시 사용 여부
            cache_ttl_seconds: 캐시 유효 시간 (포트폴리오 데이터는 2분)
        """
        super().__init__(enable_cache, cache_ttl_seconds)
        self.stock_mapper = StockMapper(enable_cache=True, cache_ttl_seconds=30)
    
    def map_single(self, api_response: Dict[str, Any]) -> Portfolio:
        """
        계좌 잔고 API 응답을 Portfolio 엔터티로 변환
        
        Args:
            api_response: 계좌 잔고 API 응답 데이터
            
        Returns:
            Portfolio: 변환된 Portfolio 엔터티
            
        Raises:
            MappingError: 매핑 실패 시
        """
        try:
            # 현금 정보 추출
            cash = self._extract_cash_amount(api_response)
            
            # 포트폴리오 생성
            portfolio = Portfolio(cash=cash)
            
            # 보유 종목 추가
            positions = self._extract_positions(api_response)
            for position in positions:
                portfolio.positions[position.stock.code] = position
            
            logger.debug(f"Portfolio 엔터티 생성 성공: 현금 {cash}, 보유종목 {len(positions)}개")
            return portfolio
            
        except Exception as e:
            logger.error(f"Portfolio 매핑 실패: {e}")
            raise MappingError(f"Portfolio 매핑 실패: {e}", api_response)
    
    def map_from_balance_response(self, balance_response: Dict[str, Any]) -> Portfolio:
        """
        한국투자증권 계좌 잔고 API 응답에서 Portfolio 엔터티 생성
        
        API 응답 구조:
        {
            "rt_cd": "0",
            "output1": [  # 보유종목 정보
                {
                    "pdno": "005930",           # 종목코드
                    "prdt_name": "삼성전자",      # 종목명
                    "hldg_qty": "10",           # 보유수량
                    "pchs_avg_pric": "74000",   # 매입평균가격
                    "prpr": "74200",            # 현재가
                    "evlu_amt": "742000",       # 평가금액
                    "pchs_amt": "740000",       # 매입금액
                    "evlu_pfls_amt": "2000",    # 평가손익금액
                    "evlu_pfls_rt": "0.27",     # 평가손익률
                    "sll_able_qty": "10"        # 매도가능수량
                }
            ],
            "output2": [  # 계좌 전체 정보
                {
                    "dnca_tot_amt": "1000000",     # 예수금총액(현금)
                    "scts_evlu_amt": "742000",     # 유가증권평가금액
                    "tot_evlu_amt": "1742000",     # 총평가금액
                    "nass_amt": "1742000",         # 순자산금액
                    "pchs_amt_smtl_amt": "740000", # 매입금액합계
                    "evlu_pfls_smtl_amt": "2000",  # 평가손익합계금액
                    "evlu_pfls_rt": "0.27"         # 평가손익률
                }
            ]
        }
        """
        try:
            # 계좌 정보에서 현금 추출
            cash = self._extract_cash_from_balance_response(balance_response)
            
            # 포트폴리오 생성
            portfolio = Portfolio(cash=cash)
            
            # 보유종목 정보 처리
            holdings = balance_response.get('output1', [])
            if isinstance(holdings, list):
                for holding_data in holdings:
                    try:
                        position = self._create_position_from_holding(holding_data)
                        if position and not position.is_empty():
                            portfolio.positions[position.stock.code] = position
                    except Exception as e:
                        logger.warning(f"보유종목 매핑 실패: {e}")
                        continue
            
            logger.info(f"Portfolio 엔터티 생성 성공: 현금 {cash}, 보유종목 {len(portfolio.positions)}개")
            return portfolio
            
        except Exception as e:
            logger.error(f"Balance 응답 매핑 실패: {e}")
            raise MappingError(f"Balance 응답 매핑 실패: {e}", balance_response)
    
    def _extract_cash_amount(self, api_response: Dict[str, Any]) -> Money:
        """API 응답에서 현금 금액 추출"""
        # 여러 구조 시도
        cash_fields = [
            'cash',
            'dnca_tot_amt',
            'cash_balance'
        ]
        
        # 직접 필드에서 추출
        for field in cash_fields:
            if field in api_response:
                amount = self.safe_get_value(api_response, field, 0, int)
                return Money.won(amount)
        
        # output2 구조에서 추출
        if 'output2' in api_response:
            output2 = api_response['output2']
            if isinstance(output2, list) and len(output2) > 0:
                amount = self.safe_get_value(output2[0], 'dnca_tot_amt', 0, int)
                return Money.won(amount)
        
        # account 구조에서 추출
        if 'account' in api_response:
            amount = self.safe_get_value(api_response['account'], 'cash', 0, int)
            return Money.won(amount)
        
        logger.warning("현금 정보를 찾을 수 없어 0으로 설정")
        return Money.zero()
    
    def _extract_cash_from_balance_response(self, balance_response: Dict[str, Any]) -> Money:
        """한국투자증권 잔고 응답에서 현금 추출"""
        output2 = balance_response.get('output2', [])
        
        if isinstance(output2, list) and len(output2) > 0:
            account_info = output2[0]
            cash_amount = self.safe_get_value(account_info, 'dnca_tot_amt', 0, int)
            return Money.won(cash_amount)
        elif isinstance(output2, dict):
            cash_amount = self.safe_get_value(output2, 'dnca_tot_amt', 0, int)
            return Money.won(cash_amount)
        
        return Money.zero()
    
    def _extract_positions(self, api_response: Dict[str, Any]) -> List[Position]:
        """API 응답에서 포지션 리스트 추출"""
        positions = []
        
        # stocks 필드에서 추출
        if 'stocks' in api_response:
            stocks_data = api_response['stocks']
            if isinstance(stocks_data, list):
                for stock_data in stocks_data:
                    try:
                        position = self._create_position_from_stock_data(stock_data)
                        if position:
                            positions.append(position)
                    except Exception as e:
                        logger.warning(f"포지션 생성 실패: {e}")
                        continue
        
        return positions
    
    def _create_position_from_holding(self, holding_data: Dict[str, Any]) -> Optional[Position]:
        """보유종목 데이터에서 Position 엔터티 생성"""
        try:
            # 수량 확인
            quantity_value = self.safe_get_value(holding_data, 'hldg_qty', 0, int)
            if quantity_value <= 0:
                return None  # 보유수량이 0이면 포지션 생성하지 않음
            
            # Stock 엔터티 생성
            stock = self.stock_mapper.map_from_balance_response(holding_data)
            
            # Position 생성에 필요한 데이터
            quantity = Quantity(quantity_value)
            average_price_value = self.safe_get_value(holding_data, 'pchs_avg_pric', 0, int)
            average_price = Price(Money.won(average_price_value))
            
            # Position 엔터티 생성
            position = Position(
                stock=stock,
                quantity=quantity,
                average_price=average_price,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            logger.debug(f"Position 생성 성공: {stock.code} {quantity} @ {average_price}")
            return position
            
        except Exception as e:
            logger.error(f"Position 생성 실패: {e}")
            return None
    
    def _create_position_from_stock_data(self, stock_data: Dict[str, Any]) -> Optional[Position]:
        """기존 stock_data 구조에서 Position 엔터티 생성 (백워드 호환성)"""
        try:
            # 수량 확인
            quantity_value = self.safe_get_value(stock_data, 'units', 0, int)
            if quantity_value <= 0:
                return None
            
            # Stock 엔터티 생성
            stock_info = {
                'code': self.safe_get_value(stock_data, 'code', '', str),
                'name': self.safe_get_value(stock_data, 'prdt_name', '', str),
                'current_price': self.safe_get_value(stock_data, 'current_price', 0, int),
                'previous_close': self.safe_get_value(stock_data, 'current_price', 0, int),  # 전일가 없으면 현재가 사용
                'volume': 0,
                'market_cap': 0
            }
            
            stock = self.stock_mapper.map_single({'stock_data': stock_info})
            
            # Position 생성
            quantity = Quantity(quantity_value)
            average_price_value = self.safe_get_value(stock_data, 'avg_buy_price', 0, int)
            average_price = Price(Money.won(average_price_value))
            
            position = Position(
                stock=stock,
                quantity=quantity,
                average_price=average_price,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            return position
            
        except Exception as e:
            logger.error(f"Stock data Position 생성 실패: {e}")
            return None


class PositionMapper(BaseMapper[Position]):
    """Position 전용 매퍼"""
    
    def __init__(self, stock_mapper: Optional[StockMapper] = None):
        super().__init__(enable_cache=True, cache_ttl_seconds=60)
        self.stock_mapper = stock_mapper or StockMapper()
    
    def map_single(self, api_response: Dict[str, Any]) -> Position:
        """개별 보유종목 데이터를 Position 엔터티로 변환"""
        try:
            # Stock 엔터티 생성
            stock = self.stock_mapper.map_from_balance_response(api_response)
            
            # Position 데이터 추출
            quantity_value = self.safe_get_value(api_response, 'hldg_qty', 0, int)
            average_price_value = self.safe_get_value(api_response, 'pchs_avg_pric', 0, int)
            
            if quantity_value <= 0:
                raise MappingError("보유수량이 0 이하입니다", api_response)
            
            # Position 엔터티 생성
            position = Position(
                stock=stock,
                quantity=Quantity(quantity_value),
                average_price=Price(Money.won(average_price_value)),
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            return position
            
        except Exception as e:
            logger.error(f"Position 매핑 실패: {e}")
            raise MappingError(f"Position 매핑 실패: {e}", api_response) 