"""
계좌 API 응답 매퍼

계좌 관련 API 응답을 도메인 객체로 변환합니다:
- 계좌 잔고 정보
- 매수 가능 금액
- 계좌 상태
"""

import logging
from typing import Dict, Any, Optional, NamedTuple
from datetime import datetime

from .base_mapper import BaseMapper, MappingError
from korea_stock_auto.domain.value_objects import Money

logger = logging.getLogger(__name__)


class AccountBalance(NamedTuple):
    """계좌 잔고 정보"""
    cash: Money
    stock_value: Money
    total_assets: Money
    total_profit_loss: Money
    profit_loss_rate: float
    updated_at: datetime


class BuyableAmount(NamedTuple):
    """매수 가능 금액 정보"""
    cash: Money
    max_buy_amount: Money
    stock_code: str
    stock_price: int
    max_quantity: int


class AccountMapper(BaseMapper[AccountBalance]):
    """계좌 API 응답 → AccountBalance 변환 매퍼"""
    
    def __init__(self, enable_cache: bool = True, cache_ttl_seconds: int = 60):
        """
        계좌 매퍼 초기화
        
        Args:
            enable_cache: 캐시 사용 여부
            cache_ttl_seconds: 캐시 유효 시간
        """
        super().__init__(enable_cache, cache_ttl_seconds)
    
    def map_single(self, api_response: Dict[str, Any]) -> AccountBalance:
        """
        계좌 잔고 API 응답을 AccountBalance로 변환
        
        Args:
            api_response: 계좌 API 응답 데이터
            
        Returns:
            AccountBalance: 변환된 계좌 잔고 정보
            
        Raises:
            MappingError: 매핑 실패 시
        """
        try:
            # 한국투자증권 API 구조에서 계좌 정보 추출
            account_info = self._extract_account_info(api_response)
            
            # AccountBalance 객체 생성
            balance = AccountBalance(
                cash=Money.won(self.safe_get_value(account_info, 'cash', 0, int)),
                stock_value=Money.won(self.safe_get_value(account_info, 'stocks_value', 0, int)),
                total_assets=Money.won(self.safe_get_value(account_info, 'total_assets', 0, int)),
                total_profit_loss=Money.won(self.safe_get_value(account_info, 'total_profit_loss', 0, int)),
                profit_loss_rate=self.safe_get_value(account_info, 'total_profit_loss_rate', 0.0, float),
                updated_at=datetime.now()
            )
            
            logger.debug(f"AccountBalance 생성 성공: 총자산 {balance.total_assets}")
            return balance
            
        except Exception as e:
            logger.error(f"AccountBalance 매핑 실패: {e}")
            raise MappingError(f"AccountBalance 매핑 실패: {e}", api_response)
    
    def map_from_balance_response(self, balance_response: Dict[str, Any]) -> AccountBalance:
        """
        한국투자증권 계좌 잔고 API 응답에서 AccountBalance 생성
        
        API 응답 구조:
        {
            "rt_cd": "0",
            "output2": [{
                "dnca_tot_amt": "1000000",     # 예수금총액
                "scts_evlu_amt": "742000",     # 유가증권평가금액
                "tot_evlu_amt": "1742000",     # 총평가금액
                "evlu_pfls_smtl_amt": "2000",  # 평가손익합계금액
                "evlu_pfls_rt": "0.27"         # 평가손익률
            }]
        }
        """
        try:
            # output2에서 계좌 정보 추출
            output2 = balance_response.get('output2', [])
            
            if isinstance(output2, list) and len(output2) > 0:
                account_data = output2[0]
            elif isinstance(output2, dict):
                account_data = output2
            else:
                raise MappingError("output2 계좌 정보가 없습니다", balance_response)
            
            # AccountBalance 생성
            balance = AccountBalance(
                cash=Money.won(self.safe_get_value(account_data, 'dnca_tot_amt', 0, int)),
                stock_value=Money.won(self.safe_get_value(account_data, 'scts_evlu_amt', 0, int)),
                total_assets=Money.won(self.safe_get_value(account_data, 'tot_evlu_amt', 0, int)),
                total_profit_loss=Money.won(self.safe_get_value(account_data, 'evlu_pfls_smtl_amt', 0, int)),
                profit_loss_rate=self.safe_get_value(account_data, 'evlu_pfls_rt', 0.0, float),
                updated_at=datetime.now()
            )
            
            logger.debug(f"Balance 응답에서 AccountBalance 생성: 총자산 {balance.total_assets}")
            return balance
            
        except Exception as e:
            logger.error(f"Balance 응답 AccountBalance 매핑 실패: {e}")
            raise MappingError(f"Balance 응답 매핑 실패: {e}", balance_response)
    
    def _extract_account_info(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """API 응답에서 계좌 정보 추출"""
        # 직접 계좌 정보가 있는 경우
        if 'account' in api_response:
            return api_response['account']
        
        # output2 구조인 경우
        if 'output2' in api_response:
            output2 = api_response['output2']
            if isinstance(output2, list) and len(output2) > 0:
                return output2[0]
            elif isinstance(output2, dict):
                return output2
        
        # 직접 필드들이 있는 경우
        return api_response


class BuyableAmountMapper(BaseMapper[BuyableAmount]):
    """매수 가능 금액 API 응답 → BuyableAmount 변환 매퍼"""
    
    def map_single(self, api_response: Dict[str, Any]) -> BuyableAmount:
        """
        매수 가능 금액 API 응답을 BuyableAmount로 변환
        
        Args:
            api_response: 매수 가능 금액 API 응답
            
        Returns:
            BuyableAmount: 변환된 매수 가능 정보
        """
        try:
            # 한국투자증권 매수 가능 금액 API 구조
            output = api_response.get('output', {})
            
            if not output:
                raise MappingError("output 필드가 없습니다", api_response)
            
            # 매수 가능 정보 추출
            cash = self.safe_get_value(output, 'ord_psbl_cash', 0, int)  # 주문가능현금
            max_buy_amount = self.safe_get_value(output, 'max_buy_amt', 0, int)  # 최대매수금액
            stock_code = self.safe_get_value(api_response, 'stock_code', '', str)  # 요청 시 포함된 종목코드
            stock_price = self.safe_get_value(api_response, 'stock_price', 0, int)  # 요청 시 포함된 주식가격
            
            # 최대 매수 가능 수량 계산
            max_quantity = 0
            if stock_price > 0:
                max_quantity = max_buy_amount // stock_price
            
            buyable = BuyableAmount(
                cash=Money.won(cash),
                max_buy_amount=Money.won(max_buy_amount),
                stock_code=stock_code,
                stock_price=stock_price,
                max_quantity=max_quantity
            )
            
            logger.debug(f"BuyableAmount 생성 성공: {stock_code} 최대 {max_quantity}주")
            return buyable
            
        except Exception as e:
            logger.error(f"BuyableAmount 매핑 실패: {e}")
            raise MappingError(f"BuyableAmount 매핑 실패: {e}", api_response) 