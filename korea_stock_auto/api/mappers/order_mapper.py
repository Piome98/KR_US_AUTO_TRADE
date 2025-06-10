"""
주문 API 응답 매퍼

주문 관련 API 응답을 Order 도메인 엔터티로 변환합니다:
- 주문 제출 응답
- 주문 조회 응답
- 주문 상태 업데이트
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from .base_mapper import BaseMapper, MappingError
from .stock_mapper import StockMapper
from korea_stock_auto.domain.entities import Order, Stock, OrderType, OrderStatus
from korea_stock_auto.domain.value_objects import Price, Money, Quantity

logger = logging.getLogger(__name__)


class OrderMapper(BaseMapper[Order]):
    """주문 API 응답 → Order 엔터티 변환 매퍼"""
    
    def __init__(self, enable_cache: bool = False, cache_ttl_seconds: int = 30):
        """
        주문 매퍼 초기화
        
        Args:
            enable_cache: 캐시 사용 여부 (주문은 보통 캐시하지 않음)
            cache_ttl_seconds: 캐시 유효 시간
        """
        super().__init__(enable_cache, cache_ttl_seconds)
        self.stock_mapper = StockMapper(enable_cache=True, cache_ttl_seconds=60)
    
    def map_single(self, api_response: Dict[str, Any]) -> Order:
        """
        주문 API 응답을 Order 엔터티로 변환
        
        Args:
            api_response: 주문 API 응답 데이터
            
        Returns:
            Order: 변환된 Order 엔터티
            
        Raises:
            MappingError: 매핑 실패 시
        """
        try:
            # 주문 정보 추출
            order_info = self._extract_order_info(api_response)
            
            # 필수 필드 검증
            self.validate_api_response(order_info, ['stock_code', 'order_type', 'quantity'])
            
            # Stock 엔터티 생성 또는 조회
            stock = self._create_stock_from_order_info(order_info)
            
            # OrderType 변환
            order_type = self._convert_order_type(order_info['order_type'])
            
            # Order 엔터티 생성
            order = Order(
                id=uuid4(),  # 새 주문 ID 생성
                stock=stock,
                order_type=order_type,
                quantity=Quantity(self.safe_get_value(order_info, 'quantity', 0, int)),
                target_price=Price(Money.won(self.safe_get_value(order_info, 'price', 0, int))),
                status=self._convert_order_status(order_info.get('status', 'PENDING')),
                created_at=datetime.now()
            )
            
            # 추가 정보 설정
            if 'order_id' in order_info:
                # API에서 제공하는 주문 ID가 있으면 사용 (실제로는 문자열이므로 메타데이터로 저장)
                order.error_message = f"API_ORDER_ID:{order_info['order_id']}"
            
            logger.debug(f"Order 엔터티 생성 성공: {order.order_type.value} {order.stock.code} {order.quantity}")
            return order
            
        except Exception as e:
            logger.error(f"Order 매핑 실패: {e}")
            raise MappingError(f"Order 매핑 실패: {e}", api_response)
    
    def map_from_order_submit_response(self, submit_response: Dict[str, Any], 
                                     stock: Stock, order_type: OrderType, 
                                     quantity: int, price: int) -> Order:
        """
        주문 제출 API 응답에서 Order 엔터티 생성
        
        API 응답 구조:
        {
            "rt_cd": "0",
            "msg_cd": "MCA00000",
            "msg1": "정상처리 되었습니다.",
            "output": {
                "KRX_FWDG_ORD_ORGNO": "",
                "ODNO": "0000117057",     # 주문번호
                "ORD_TMD": "121052"       # 주문시각
            }
        }
        """
        try:
            # 주문 성공 여부 확인
            rt_cd = submit_response.get('rt_cd', '1')
            if rt_cd != '0':
                status = OrderStatus.REJECTED
                error_msg = submit_response.get('msg1', '주문 실패')
            else:
                status = OrderStatus.SUBMITTED
                error_msg = None
            
            # 주문 번호 추출
            output = submit_response.get('output', {})
            order_id = self.safe_get_value(output, 'ODNO', '', str)
            order_time = self.safe_get_value(output, 'ORD_TMD', '', str)
            
            # Order 엔터티 생성
            order = Order(
                id=uuid4(),
                stock=stock,
                order_type=order_type,
                quantity=Quantity(quantity),
                target_price=Price(Money.won(price)),
                status=status,
                created_at=datetime.now(),
                submitted_at=datetime.now() if status == OrderStatus.SUBMITTED else None,
                error_message=f"API_ORDER_ID:{order_id}" if order_id else error_msg
            )
            
            logger.info(f"주문 제출 응답 매핑 성공: {order_type.value} {stock.code} {quantity}주")
            return order
            
        except Exception as e:
            logger.error(f"주문 제출 응답 매핑 실패: {e}")
            raise MappingError(f"주문 제출 응답 매핑 실패: {e}", submit_response)
    
    def map_from_order_inquiry_response(self, inquiry_response: Dict[str, Any]) -> Order:
        """
        주문 조회 API 응답에서 Order 엔터티 생성
        
        API 응답 구조:
        {
            "rt_cd": "0",
            "output": [
                {
                    "pdno": "005930",           # 종목코드
                    "prdt_name": "삼성전자",     # 종목명
                    "ord_qty": "10",            # 주문수량
                    "ord_unpr": "74000",        # 주문단가
                    "ord_tmd": "121052",        # 주문시각
                    "ord_gno_brno": "01234",    # 주문채번지점번호
                    "orgn_odno": "0000117057",  # 원주문번호
                    "ord_dvsn_cd": "00",        # 주문구분코드 (00: 지정가)
                    "sll_buy_dvsn_cd": "02",    # 매도매수구분코드 (01: 매도, 02: 매수)
                    "ord_psbl_yn": "Y",         # 주문가능여부
                    "ord_st": "02",             # 주문상태 (02: 접수, 10: 전량체결)
                    "ccld_qty": "10",           # 체결수량
                    "ccld_amt": "740000",       # 체결금액
                    "avg_prvs": "74000"         # 평균가
                }
            ]
        }
        """
        try:
            output = inquiry_response.get('output', [])
            if not output or not isinstance(output, list):
                raise MappingError("주문 조회 응답에 주문 정보가 없습니다", inquiry_response)
            
            # 첫 번째 주문 정보 사용 (일반적으로 하나의 주문 조회)
            order_data = output[0]
            
            # Stock 정보 생성
            stock_info = {
                'code': self.safe_get_value(order_data, 'pdno', '', str),
                'name': self.safe_get_value(order_data, 'prdt_name', '', str),
                'current_price': self.safe_get_value(order_data, 'ord_unpr', 0, int),
                'previous_close': self.safe_get_value(order_data, 'ord_unpr', 0, int),
                'volume': 0,
                'market_cap': 0
            }
            stock = self.stock_mapper.map_single({'stock_data': stock_info})
            
            # 주문 정보 추출
            order_qty = self.safe_get_value(order_data, 'ord_qty', 0, int)
            order_price = self.safe_get_value(order_data, 'ord_unpr', 0, int)
            
            # 주문 구분 코드로 매수/매도 판단
            sll_buy_code = self.safe_get_value(order_data, 'sll_buy_dvsn_cd', '02', str)
            order_type = OrderType.BUY if sll_buy_code == '02' else OrderType.SELL
            
            # 주문 상태 변환
            ord_status = self.safe_get_value(order_data, 'ord_st', '02', str)
            status = self._convert_api_order_status(ord_status)
            
            # 체결 정보
            filled_qty = self.safe_get_value(order_data, 'ccld_qty', 0, int)
            filled_amount = self.safe_get_value(order_data, 'ccld_amt', 0, int)
            
            # Order 엔터티 생성
            order = Order(
                id=uuid4(),
                stock=stock,
                order_type=order_type,
                quantity=Quantity(order_qty),
                target_price=Price(Money.won(order_price)),
                status=status,
                filled_quantity=Quantity(filled_qty),
                filled_price=Price(Money.won(filled_amount // filled_qty)) if filled_qty > 0 else None,
                created_at=datetime.now(),
                submitted_at=datetime.now(),
                filled_at=datetime.now() if status == OrderStatus.FILLED else None,
                error_message=f"API_ORDER_ID:{self.safe_get_value(order_data, 'orgn_odno', '', str)}"
            )
            
            logger.debug(f"주문 조회 응답 매핑 성공: {order}")
            return order
            
        except Exception as e:
            logger.error(f"주문 조회 응답 매핑 실패: {e}")
            raise MappingError(f"주문 조회 응답 매핑 실패: {e}", inquiry_response)
    
    def _extract_order_info(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """API 응답에서 주문 정보 추출"""
        # 직접 주문 정보가 있는 경우
        if 'order_info' in api_response:
            return api_response['order_info']
        
        # output 구조인 경우
        if 'output' in api_response:
            output = api_response['output']
            if isinstance(output, list) and len(output) > 0:
                return output[0]
            elif isinstance(output, dict):
                return output
        
        # 직접 필드들이 있는 경우
        return api_response
    
    def _create_stock_from_order_info(self, order_info: Dict[str, Any]) -> Stock:
        """주문 정보에서 Stock 엔터티 생성"""
        stock_data = {
            'code': self.safe_get_value(order_info, 'stock_code', '', str),
            'name': self.safe_get_value(order_info, 'stock_name', 'Unknown', str),
            'current_price': self.safe_get_value(order_info, 'price', 0, int),
            'previous_close': self.safe_get_value(order_info, 'price', 0, int),
            'volume': 0,
            'market_cap': 0
        }
        
        return self.stock_mapper.map_single({'stock_data': stock_data})
    
    def _convert_order_type(self, order_type_str: str) -> OrderType:
        """주문 구분 문자열을 OrderType으로 변환"""
        order_type_upper = str(order_type_str).upper()
        
        if order_type_upper in ['BUY', '02', 'LONG']:
            return OrderType.BUY
        elif order_type_upper in ['SELL', '01', 'SHORT']:
            return OrderType.SELL
        else:
            logger.warning(f"알 수 없는 주문 구분: {order_type_str}, BUY로 기본 설정")
            return OrderType.BUY
    
    def _convert_order_status(self, status_str: str) -> OrderStatus:
        """주문 상태 문자열을 OrderStatus로 변환"""
        status_upper = str(status_str).upper()
        
        status_mapping = {
            'PENDING': OrderStatus.PENDING,
            'SUBMITTED': OrderStatus.SUBMITTED,
            'FILLED': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'PARTIAL_FILLED': OrderStatus.PARTIAL_FILLED,
            'PARTIAL': OrderStatus.PARTIAL_FILLED
        }
        
        return status_mapping.get(status_upper, OrderStatus.PENDING)
    
    def _convert_api_order_status(self, api_status: str) -> OrderStatus:
        """한국투자증권 API 주문 상태 코드를 OrderStatus로 변환"""
        # 한국투자증권 주문 상태 코드 매핑
        api_status_mapping = {
            '01': OrderStatus.SUBMITTED,    # 주문접수
            '02': OrderStatus.SUBMITTED,    # 주문접수
            '10': OrderStatus.FILLED,       # 전량체결
            '11': OrderStatus.PARTIAL_FILLED,  # 부분체결
            '20': OrderStatus.CANCELLED,    # 주문취소
            '21': OrderStatus.CANCELLED,    # 취소접수
            '22': OrderStatus.CANCELLED,    # 취소체결
            '30': OrderStatus.REJECTED,     # 주문거부
            '31': OrderStatus.REJECTED,     # 거부접수
        }
        
        return api_status_mapping.get(api_status, OrderStatus.PENDING) 