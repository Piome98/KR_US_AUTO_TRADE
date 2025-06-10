"""
한국 주식 자동매매 - 주문 서비스

주식 매수, 매도, 정정, 취소 등 주문 관련 기능을 제공합니다.
"""

import json
import requests
import logging
from typing import Dict, Optional, Any, TYPE_CHECKING, List, Tuple

from korea_stock_auto.config import get_config
from korea_stock_auto.utils.utils import send_message, hashkey

# 도메인 엔터티 import
from korea_stock_auto.domain import Order, Stock, OrderType, Price, Quantity

# API 매퍼 통합
from korea_stock_auto.api.mappers import OrderMapper, StockMapper, MappingError

if TYPE_CHECKING:
    from korea_stock_auto.api.client import KoreaInvestmentApiClient

logger = logging.getLogger("stock_auto")

class OrderService:
    """
    주문 처리 서비스 (API 매퍼 통합)
    
    주식 매수, 매도, 정정, 취소 등 주문 관련 기능을 제공합니다.
    """
    
    def __init__(self, api_client: 'KoreaInvestmentApiClient'):
        """
        OrderService 초기화
        
        Args:
            api_client: KoreaInvestmentApiClient 인스턴스
        """
        self.api_client = api_client
        self.config = get_config()
        
        # API 매퍼 초기화
        self.order_mapper = OrderMapper(
            enable_cache=False,  # 주문은 실시간성이 중요하므로 캐시 비활성화
            cache_ttl_seconds=30
        )
        self.stock_mapper = StockMapper(
            enable_cache=True,
            cache_ttl_seconds=30
        )
        
        logger.debug("OrderService 초기화 완료 (API 매퍼 통합)")

    def buy_stock(self, code: str, qty: int, price: Optional[int] = None) -> bool:
        """
        주식 매수 함수 (백워드 호환성 유지)
        
        Args:
            code: 종목 코드
            qty: 주문 수량
            price: 주문 가격 (지정가시 필수, None 또는 0 이하인 경우 시장가 주문)
            
        Returns:
            bool: 매수 성공 여부
            
        Examples:
            >>> order_service.buy_stock("005930", 10, 70000)  # 삼성전자 10주 70,000원에 지정가 매수
            >>> order_service.buy_stock("005930", 10)  # 삼성전자 10주 시장가 매수
        """
        try:
            # 매수 주문 실행 및 Order 엔터티 반환
            success, order_result = self.buy_stock_as_entity(code, qty, price)
            return success
            
        except Exception as e:
            logger.error(f"매수 주문 처리 중 예외: {e}")
            return False
    
    def buy_stock_as_entity(self, code: str, qty: int, price: Optional[int] = None) -> Tuple[bool, Optional[Order]]:
        """
        주식 매수 함수 (Order 엔터티 반환, NEW)
        
        Args:
            code: 종목 코드
            qty: 주문 수량
            price: 주문 가격 (지정가시 필수, None 또는 0 이하인 경우 시장가 주문)
            
        Returns:
            Tuple[bool, Optional[Order]]: (성공 여부, Order 엔터티)
        """
        try:
            # 원시 API 응답 조회
            raw_response = self._buy_stock_raw(code, qty, price)
            if not raw_response:
                return False, None
            
            # Stock 엔터티 생성 (주문에 필요)
            stock = self._create_stock_entity(code, price or 0)
            if not stock:
                logger.error(f"{code} Stock 엔터티 생성 실패")
                return False, None
            
            # OrderMapper를 통한 Order 엔터티 생성 시도
            try:
                order_type = OrderType.BUY
                target_price = price if price and price > 0 else 0
                
                order = self.order_mapper.map_from_order_submit_response(
                    raw_response, stock, order_type, qty, target_price
                )
                
                order_type_str = "시장가" if target_price == 0 else f"지정가({target_price:,}원)"
                logger.info(f"{code} 매수 주문 성공 (매퍼): {qty}주 {order_type_str}")
                
                return True, order
                
            except MappingError as e:
                logger.warning(f"Order 매핑 실패, 수동 생성: {e}")
                # 백워드 호환성: 수동으로 Order 엔터티 생성
                return self._create_order_entity_legacy(raw_response, stock, OrderType.BUY, qty, price)
                
        except Exception as e:
            logger.error(f"매수 주문 처리 중 오류: {e}")
            return False, None
    
    def _buy_stock_raw(self, code: str, qty: int, price: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        주식 매수 원시 API 응답 조회
        
        Args:
            code: 종목 코드
            qty: 주문 수량
            price: 주문 가격
            
        Returns:
            dict or None: 원시 API 응답
        """
        path = "uapi/domestic-stock/v1/trading/order-cash"
        url = f"{self.config.current_api.base_url}/{path}"
        
        # 시장가 또는 지정가 결정
        if price is None or price <= 0:
            ord_dvsn = "01"  # 시장가
            price = 0
        else:
            ord_dvsn = "00"  # 지정가
        
        data = {
            "CANO": self.config.current_api.account_number,
            "ACNT_PRDT_CD": self.config.current_api.account_product_code,
            "PDNO": code,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": str(price),
        }
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC0802U" if self.config.use_realtime_api else "VTTC0802U"
        
        # 해시키 생성
        hash_val = hashkey(data, self.config.current_api.app_key, self.config.current_api.app_secret, self.config.current_api.base_url)
        headers = self.api_client._get_headers(tr_id, hashkey_val=hash_val)
        
        try:
            # API 호출 속도 제한 적용
            self.api_client._rate_limit()
            
            order_type = "시장가" if ord_dvsn == "01" else f"지정가({price:,}원)"
            logger.info(f"{code} {qty}주 매수 주문({order_type}) 요청")
            
            res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            res_json = self.api_client._handle_response(res, "매수 주문 실패")
            
            if not res_json:
                return None
            
            if res_json.get("rt_cd") == "0":
                order_no = res_json.get("output", {}).get("ODNO", "알 수 없음")
                success_msg = f"[매수 성공] {code} {qty}주 {order_type} (주문번호: {order_no})"
                logger.info(success_msg)
                send_message(success_msg, self.config.notification.discord_webhook_url)
                return res_json
            else:
                error_msg = f"[매수 실패] {res_json}"
                logger.error(error_msg)
                send_message(error_msg, self.config.notification.discord_webhook_url)
                return None
                
        except Exception as e:
            error_msg = f"매수 주문 중 예외 발생: {e}"
            logger.error(error_msg, exc_info=True)
            send_message(f"[오류] 매수 주문 실패: {e}", self.config.notification.discord_webhook_url)
            return None

    def sell_stock(self, code: str, qty: int, price: Optional[int] = None) -> bool:
        """
        주식 매도 함수 (백워드 호환성 유지)
        
        Args:
            code: 종목 코드
            qty: 주문 수량
            price: 주문 가격 (지정가시 필수, None 또는 0 이하인 경우 시장가 주문)
            
        Returns:
            bool: 매도 성공 여부
            
        Examples:
            >>> order_service.sell_stock("005930", 10, 70000)  # 삼성전자 10주 70,000원에 지정가 매도
            >>> order_service.sell_stock("005930", 10)  # 삼성전자 10주 시장가 매도
        """
        try:
            # 매도 주문 실행 및 Order 엔터티 반환
            success, order_result = self.sell_stock_as_entity(code, qty, price)
            return success
            
        except Exception as e:
            logger.error(f"매도 주문 처리 중 예외: {e}")
            return False
    
    def sell_stock_as_entity(self, code: str, qty: int, price: Optional[int] = None) -> Tuple[bool, Optional[Order]]:
        """
        주식 매도 함수 (Order 엔터티 반환, NEW)
        
        Args:
            code: 종목 코드
            qty: 주문 수량
            price: 주문 가격 (지정가시 필수, None 또는 0 이하인 경우 시장가 주문)
            
        Returns:
            Tuple[bool, Optional[Order]]: (성공 여부, Order 엔터티)
        """
        try:
            # 원시 API 응답 조회
            raw_response = self._sell_stock_raw(code, qty, price)
            if not raw_response:
                return False, None
            
            # Stock 엔터티 생성 (주문에 필요)
            stock = self._create_stock_entity(code, price or 0)
            if not stock:
                logger.error(f"{code} Stock 엔터티 생성 실패")
                return False, None
            
            # OrderMapper를 통한 Order 엔터티 생성 시도
            try:
                order_type = OrderType.SELL
                target_price = price if price and price > 0 else 0
                
                order = self.order_mapper.map_from_order_submit_response(
                    raw_response, stock, order_type, qty, target_price
                )
                
                order_type_str = "시장가" if target_price == 0 else f"지정가({target_price:,}원)"
                logger.info(f"{code} 매도 주문 성공 (매퍼): {qty}주 {order_type_str}")
                
                return True, order
                
            except MappingError as e:
                logger.warning(f"Order 매핑 실패, 수동 생성: {e}")
                # 백워드 호환성: 수동으로 Order 엔터티 생성
                return self._create_order_entity_legacy(raw_response, stock, OrderType.SELL, qty, price)
                
        except Exception as e:
            logger.error(f"매도 주문 처리 중 오류: {e}")
            return False, None
    
    def _sell_stock_raw(self, code: str, qty: int, price: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        주식 매도 원시 API 응답 조회
        
        Args:
            code: 종목 코드
            qty: 주문 수량
            price: 주문 가격
            
        Returns:
            dict or None: 원시 API 응답
        """
        path = "uapi/domestic-stock/v1/trading/order-cash"
        url = f"{self.config.current_api.base_url}/{path}"
        
        # 시장가 또는 지정가 결정
        if price is None or price <= 0:
            ord_dvsn = "01"  # 시장가
            price = 0
        else:
            ord_dvsn = "00"  # 지정가
        
        data = {
            "CANO": self.config.current_api.account_number,
            "ACNT_PRDT_CD": self.config.current_api.account_product_code,
            "PDNO": code,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": str(price),
        }
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름 (매도는 다른 TR_ID)
        tr_id = "TTTC0801U" if self.config.use_realtime_api else "VTTC0801U"
        
        # 해시키 생성
        hash_val = hashkey(data, self.config.current_api.app_key, self.config.current_api.app_secret, self.config.current_api.base_url)
        headers = self.api_client._get_headers(tr_id, hashkey_val=hash_val)
        
        try:
            # API 호출 속도 제한 적용
            self.api_client._rate_limit()
            
            order_type = "시장가" if ord_dvsn == "01" else f"지정가({price:,}원)"
            logger.info(f"{code} {qty}주 매도 주문({order_type}) 요청")
            
            res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            res_json = self.api_client._handle_response(res, "매도 주문 실패")
            
            if not res_json:
                return None
            
            if res_json.get("rt_cd") == "0":
                order_no = res_json.get("output", {}).get("ODNO", "알 수 없음")
                success_msg = f"[매도 성공] {code} {qty}주 {order_type} (주문번호: {order_no})"
                logger.info(success_msg)
                send_message(success_msg, self.config.notification.discord_webhook_url)
                return res_json
            else:
                error_msg = f"[매도 실패] {res_json}"
                logger.error(error_msg)
                send_message(error_msg, self.config.notification.discord_webhook_url)
                return None
                
        except Exception as e:
            error_msg = f"매도 주문 중 예외 발생: {e}"
            logger.error(error_msg, exc_info=True)
            send_message(f"[오류] 매도 주문 실패: {e}", self.config.notification.discord_webhook_url)
            return None

    def _create_stock_entity(self, code: str, price: int) -> Optional[Stock]:
        """
        Stock 엔터티 생성 (주문용)
        
        Args:
            code: 종목 코드
            price: 가격
            
        Returns:
            Stock: Stock 엔터티 또는 None
        """
        try:
            # StockMapper를 통한 Stock 엔터티 생성 시도 (최소 정보로)
            stock_data = {
                'code': code,
                'name': code,  # 기본값
                'current_price': max(price, 1),  # 0이면 1로 설정
                'previous_close': max(price, 1),
                'volume': 0,
                'market_cap': 0
            }
            
            # 🔧 이미 가공된 데이터로 직접 Stock 엔터티 생성 (StockMapper 우회)
            from korea_stock_auto.domain.entities import Stock
            from korea_stock_auto.domain.value_objects import Price, Money, Quantity
            from datetime import datetime
            
            stock = Stock(
                code=stock_data['code'],
                name=stock_data.get('name', stock_data['code']),
                current_price=Price(stock_data['current_price']),
                previous_close=Price(stock_data.get('previous_close', stock_data['current_price'])),
                market_cap=Money(stock_data.get('market_cap', 0)),
                volume=Quantity(stock_data.get('volume', 0)),
                updated_at=datetime.now()
            )
            return stock
            
        except Exception as e:
            logger.warning(f"StockMapper를 통한 Stock 엔터티 생성 실패, 직접 생성: {e}")
            # 백워드 호환성: 직접 생성
            try:
                stock = Stock(
                    code=code,
                    name=code,
                    current_price=Price.won(max(price, 1)),
                    previous_close=Price.won(max(price, 1))
                )
                return stock
            except Exception as e2:
                logger.error(f"Stock 엔터티 직접 생성도 실패: {e2}")
                return None
    
    def _create_order_entity_legacy(self, raw_response: Dict[str, Any], stock: Stock, 
                                   order_type: OrderType, qty: int, price: Optional[int]) -> Tuple[bool, Optional[Order]]:
        """수동으로 Order 엔터티 생성 (백워드 호환성)"""
        try:
            target_price = Price.won(price) if price and price > 0 else stock.current_price
            quantity = Quantity(qty)
            
            if order_type == OrderType.BUY:
                order = Order.create_buy_order(stock, quantity, target_price)
            else:
                order = Order.create_sell_order(stock, quantity, target_price)
            
            # API 성공 응답이면 주문 제출 상태로 설정
            if raw_response.get("rt_cd") == "0":
                order.submit()
                
                # 주문번호 설정 (있으면)
                order_no = raw_response.get("output", {}).get("ODNO")
                if order_no:
                    order.api_order_id = order_no
            
            return True, order
            
        except Exception as e:
            logger.error(f"Order 엔터티 수동 생성 실패: {e}")
            return False, None

    def fetch_buyable_amount(self, code: str, price: int = 0) -> Optional[Dict[str, Any]]:
        """
        주식 매수 가능 금액 조회
        
        Args:
            code: 종목 코드
            price: 주문 가격 (0인 경우 현재가로 계산)
            
        Returns:
            dict or None: 매수 가능 정보 (실패 시 None)
            
        Examples:
            >>> order_service.fetch_buyable_amount("005930", 70000)  # 삼성전자 70,000원일 때 매수 가능 금액 조회
        """
        path = "uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        url = f"{self.config.current_api.base_url}/{path}"
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC8908R" if self.config.use_realtime_api else "VTTC8908R"
        
        params = {
            "CANO": self.config.current_api.account_number,
            "ACNT_PRDT_CD": self.config.current_api.account_product_code,
            "PDNO": code,
            "ORD_UNPR": str(price),
            "ORD_DVSN": "02",
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "N"
        }
        
        headers = self.api_client._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self.api_client._rate_limit()
            
            logger.info(f"{code} 매수 가능 금액 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self.api_client._handle_response(res, f"{code} 매수 가능 금액 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", {})
            
            # 결과 가공
            buyable_info = {
                "code": code,
                "price": price,
                "max_amount": int(output.get("nrcvb_buy_amt", "0").replace(',', '')),  # 최대 매수 가능 금액
                "max_qty": int(output.get("max_buy_qty", "0").replace(',', '')),       # 최대 매수 가능 수량
                "deposited_cash": int(output.get("dnca_tot_amt", "0").replace(',', '')),  # 예수금 총액
                "available_cash": int(output.get("prvs_rcdl_excc_amt", "0").replace(',', '')),  # 가용 현금
                "asset_value": int(output.get("tot_evlu_amt", "0").replace(',', ''))   # 총평가금액
            }
            
            logger.info(f"{code} 매수 가능 금액 조회 성공: {buyable_info['max_amount']:,}원 ({buyable_info['max_qty']:,}주)")
            return buyable_info
            
        except Exception as e:
            logger.error(f"매수 가능 금액 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 매수 가능 금액 조회 실패: {e}", self.config.notification.discord_webhook_url)
            return None

    def clear_mappers_cache(self) -> None:
        """매퍼 캐시 전체 삭제"""
        self.order_mapper.clear_cache()
        self.stock_mapper.clear_cache()
        logger.debug("OrderMapper, StockMapper 캐시 삭제 완료")
    
    def get_mappers_cache_stats(self) -> Dict[str, Any]:
        """매퍼 캐시 통계 조회"""
        return {
            'order_mapper': self.order_mapper.get_cache_stats(),
            'stock_mapper': self.stock_mapper.get_cache_stats()
        }
    
    def get_orders_as_entities(self, date: Optional[str] = None) -> List[Order]:
        """
        주문 조회 (Order 엔터티 리스트 반환, NEW)
        
        Args:
            date: 조회 날짜 (YYYYMMDD, None이면 당일)
            
        Returns:
            List[Order]: Order 엔터티 리스트
        """
        try:
            # 원시 API 응답 조회 (실제 구현 필요)
            # raw_response = self._get_orders_raw(date)
            # if not raw_response:
            #     return []
            
            # OrderMapper를 통한 Order 엔터티 생성
            # orders = self.order_mapper.map_from_order_inquiry_response(raw_response)
            # return orders
            
            # 현재는 빈 리스트 반환 (실제 API 구현 필요)
            logger.warning("주문 조회 API 미구현")
            return []
            
        except Exception as e:
            logger.error(f"Order 엔터티 조회 실패: {e}")
            return [] 