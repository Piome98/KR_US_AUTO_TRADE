"""
한국 주식 자동매매 - 매수 주문 모듈

주식 매수 관련 기능을 제공합니다.
"""

import json
import requests
import logging
from typing import Dict, Optional, Any, TYPE_CHECKING, cast

from korea_stock_auto.config import (
    URL_BASE, CANO, ACNT_PRDT_CD, USE_REALTIME_API
)
from korea_stock_auto.utils.utils import send_message, hashkey

# 타입 힌트만을 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

# 로깅 설정
logger = logging.getLogger("stock_auto")

class BuyOrderMixin:
    """
    주식 매수 관련 기능 Mixin
    
    주식 매수 관련 기능을 제공합니다.
    """
    
    def buy_stock(self, code: str, qty: int, price: Optional[int] = None) -> bool:
        """
        주식 매수 함수
        
        Args:
            code: 종목 코드
            qty: 주문 수량
            price: 주문 가격 (지정가시 필수, None 또는 0 이하인 경우 시장가 주문)
            
        Returns:
            bool: 매수 성공 여부
            
        Examples:
            >>> api_client.buy_stock("005930", 10, 70000)  # 삼성전자 10주 70,000원에 지정가 매수
            >>> api_client.buy_stock("005930", 10)  # 삼성전자 10주 시장가 매수
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/trading/order-cash"
        url = f"{URL_BASE}/{path}"
        
        # 시장가 또는 지정가 결정
        if price is None or price <= 0:
            ord_dvsn = "01"  # 시장가
            price = 0
        else:
            ord_dvsn = "00"  # 지정가
        
        data = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "PDNO": code,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": str(price),
        }
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC0802U" if USE_REALTIME_API else "VTTC0802U"
        
        # 해시키 생성
        hash_val = hashkey(data)
        headers = self._get_headers(tr_id, hashkey_val=hash_val)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            order_type = "시장가" if ord_dvsn == "01" else f"지정가({price:,}원)"
            logger.info(f"{code} {qty}주 매수 주문({order_type}) 요청")
            
            res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            res_json = self._handle_response(res, "매수 주문 실패")
            
            if not res_json:
                return False
            
            if res_json.get("rt_cd") == "0":
                order_no = res_json.get("output", {}).get("ODNO", "알 수 없음")
                success_msg = f"[매수 성공] {code} {qty}주 {order_type} (주문번호: {order_no})"
                logger.info(success_msg)
                send_message(success_msg)
                return True
            else:
                error_msg = f"[매수 실패] {res_json}"
                logger.error(error_msg)
                send_message(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"매수 주문 중 예외 발생: {e}"
            logger.error(error_msg, exc_info=True)
            send_message(f"[오류] 매수 주문 실패: {e}")
            return False
    
    def fetch_buyable_amount(self, code: str, price: int = 0) -> Optional[Dict[str, Any]]:
        """
        주식 매수 가능 금액 조회
        
        Args:
            code: 종목 코드
            price: 주문 가격 (0인 경우 현재가로 계산)
            
        Returns:
            dict or None: 매수 가능 정보 (실패 시 None)
            
        Notes:
            모의투자 지원 함수입니다.
            
        Examples:
            >>> api_client.fetch_buyable_amount("005930", 70000)  # 삼성전자 70,000원일 때 매수 가능 금액 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        url = f"{URL_BASE}/{path}"
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC8908R" if USE_REALTIME_API else "VTTC8908R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "PDNO": code,
            "ORD_UNPR": str(price),
            "ORD_DVSN": "02",
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "N"
        }
        
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} 매수 가능 금액 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} 매수 가능 금액 조회 실패")
            
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
            send_message(f"[오류] 매수 가능 금액 조회 실패: {e}")
            return None 