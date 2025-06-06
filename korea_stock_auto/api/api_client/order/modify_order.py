"""
한국 주식 자동매매 - 주문 정정/취소 모듈

주식 주문 정정 및 취소 관련 기능을 제공합니다.
"""

import json
import requests
import logging
from typing import Dict, Optional, Any, TYPE_CHECKING, cast

from korea_stock_auto.config import get_config
from korea_stock_auto.utils.utils import send_message, hashkey

# 타입 힌트만을 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

# 로깅 설정
logger = logging.getLogger("stock_auto")

class OrderModificationMixin:
    """
    주문 정정/취소 관련 기능 Mixin
    
    주문 정정 및 취소 관련 기능을 제공합니다.
    """
    
    def modify_order(self, org_order_no: str, code: str, qty: int, price: int, order_type: str = "00") -> bool:
        """
        주문 정정 요청
        
        Args:
            org_order_no: 원주문번호
            code: 종목코드
            qty: 정정 수량
            price: 정정 가격
            order_type: 주문구분(00: 지정가, 01: 시장가)
            
        Returns:
            bool: 정정 성공 여부
            
        Notes:
            모의투자 지원 함수입니다.
            
        Examples:
            >>> api_client.modify_order("XXXXXXXX", "005930", 10, 70000)  # 삼성전자 주문 정정
        """

        # 설정 로드
        config = get_config()
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/trading/order-rvsecncl"
        url = f"{config.current_api.base_url}/{path}"
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC0803U" if config.use_realtime_api else "VTTC0803U"
        
        data = {
            "CANO": config.current_api.account_number,
            "ACNT_PRDT_CD": config.current_api.account_product_code,
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": org_order_no,
            "RVSE_CNCL_DVSN_CD": "01",  # 정정(01)
            "ORD_DVSN": order_type,
            "PDNO": code,
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price),
            "QTY_ALL_ORD_YN": "N"
        }
        
        # 해시키 생성
        hash_val = hashkey(data, config.current_api.app_key, config.current_api.app_secret, config.current_api.base_url)
        headers = self._get_headers(tr_id, hashkey_val=hash_val)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            order_type_name = "시장가" if order_type == "01" else f"지정가({price:,}원)"
            logger.info(f"주문번호 {org_order_no} {code} {qty}주 정정 주문({order_type_name}) 요청")
            
            res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            res_json = self._handle_response(res, "주문 정정 실패")
            
            if not res_json:
                return False
            
            if res_json.get("rt_cd") == "0":
                order_no = res_json.get("output", {}).get("ODNO", "알 수 없음")
                success_msg = f"[정정 성공] {code} {qty}주 {order_type_name} (주문번호: {order_no})"
                logger.info(success_msg)
                send_message(success_msg, config.notification.discord_webhook_url)
                return True
            else:
                error_msg = f"[정정 실패] {res_json}"
                logger.error(error_msg)
                send_message(error_msg, config.notification.discord_webhook_url)
                return False
                
        except Exception as e:
            error_msg = f"주문 정정 중 예외 발생: {e}"
            logger.error(error_msg, exc_info=True)
            send_message(f"[오류] 주문 정정 실패: {e}", config.notification.discord_webhook_url)
            return False
    
    def cancel_order(self, order_no: str, code: str, qty: int) -> bool:
        """
        주문 취소 요청
        
        Args:
            order_no: 주문번호
            code: 종목코드
            qty: 취소 수량
            
        Returns:
            bool: 취소 성공 여부
            
        Notes:
            모의투자 지원 함수입니다.
            
        Examples:
            >>> api_client.cancel_order("XXXXXXXX", "005930", 10)  # 삼성전자 주문 취소
        """
        # 설정 로드
        config = get_config()
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/trading/order-rvsecncl"
        url = f"{config.current_api.base_url}/{path}"
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC0803U" if config.use_realtime_api else "VTTC0803U"
        
        data = {
            "CANO": config.current_api.account_number,
            "ACNT_PRDT_CD": config.current_api.account_product_code,
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": order_no,
            "RVSE_CNCL_DVSN_CD": "02",  # 취소(02)
            "ORD_DVSN": "00",
            "PDNO": code,
            "ORD_QTY": str(qty),
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y"  # 잔량 전부 취소
        }
        
        # 해시키 생성
        hash_val = hashkey(data, config.current_api.app_key, config.current_api.app_secret, config.current_api.base_url)
        headers = self._get_headers(tr_id, hashkey_val=hash_val)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"주문번호 {order_no} {code} {qty}주 취소 요청")
            
            res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            res_json = self._handle_response(res, "주문 취소 실패")
            
            if not res_json:
                return False
            
            if res_json.get("rt_cd") == "0":
                success_msg = f"[취소 성공] {code} {qty}주 (주문번호: {order_no})"
                logger.info(success_msg)
                send_message(success_msg, config.notification.discord_webhook_url)
                return True
            else:
                error_msg = f"[취소 실패] {res_json}"
                logger.error(error_msg)
                send_message(error_msg, config.notification.discord_webhook_url)
                return False
                
        except Exception as e:
            error_msg = f"주문 취소 중 예외 발생: {e}"
            logger.error(error_msg, exc_info=True)
            send_message(f"[오류] 주문 취소 실패: {e}", config.notification.discord_webhook_url)
            return False
    
    def cancel_all_orders(self) -> bool:
        """
        미체결 주문 전체 취소
        
        모든 미체결 주문을 일괄 취소합니다.
        
        Returns:
            bool: 취소 성공 여부
            
        Notes:
            모의투자에서는 지원되지 않습니다.
            
        Examples:
            >>> api_client.cancel_all_orders()  # 모든 미체결 주문 취소
        """
        # 설정 로드
        config = get_config()
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # 모의투자에서는 기능 미지원
        if not config.use_realtime_api:
            logger.warning("미체결 주문 전체 취소 기능은 모의투자에서 지원되지 않습니다.")
            send_message("[안내] 미체결 주문 전체 취소 기능은 모의투자에서 지원되지 않습니다.", config.notification.discord_webhook_url)
            return False
        
        # 미체결 주문 조회
        pending_orders = self.get_pending_orders()
        if not pending_orders:
            logger.info("취소할 미체결 주문이 없습니다.")
            return True
        
        # 각 미체결 주문에 대해 취소 요청
        success_count = 0
        for order in pending_orders:
            order_no = order.get("order_no")
            code = order.get("code")
            qty = order.get("remaining_qty")
            
            if self.cancel_order(order_no, code, qty):
                success_count += 1
        
        # 모든 주문 취소 성공 여부 반환
        if success_count == len(pending_orders):
            logger.info(f"전체 {success_count}건의 미체결 주문 취소 완료")
            return True
        else:
            logger.warning(f"전체 {len(pending_orders)}건 중 {success_count}건만 취소 성공")
            return False 