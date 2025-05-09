"""
한국 주식 자동매매 - 주식 주문 모듈
"""

import json
import requests
import logging
from typing import Dict, List, Optional, Any, Union

from korea_stock_auto.config import (
    URL_BASE, CANO, ACNT_PRDT_CD, USE_REALTIME_API
)
from korea_stock_auto.utils.utils import send_message, hashkey
from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

logger = logging.getLogger("stock_auto")

class StockOrderMixin:
    """주식 주문 관련 기능 Mixin"""
    
    def buy_stock(self, code: str, qty: int, price: Optional[int] = None) -> bool:
        """
        주식 매수 함수
        
        Args:
            code (str): 종목 코드
            qty (int): 주문 수량
            price (int, optional): 주문 가격 (지정가시 필수)
            
        Returns:
            bool: 매수 성공 여부
        """
        self: KoreaInvestmentApiClient  # type hint
        
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
            
            order_type = "시장가" if ord_dvsn == "01" else f"지정가({price}원)"
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
    
    def sell_stock(self, code: str, qty: int, price: Optional[int] = None) -> bool:
        """
        주식 매도 함수
        
        Args:
            code (str): 종목 코드
            qty (int): 주문 수량
            price (int, optional): 주문 가격 (지정가시 필수)
            
        Returns:
            bool: 매도 성공 여부
        """
        self: KoreaInvestmentApiClient  # type hint
        
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
        tr_id = "TTTC0801U" if USE_REALTIME_API else "VTTC0801U"
        
        # 해시키 생성
        hash_val = hashkey(data)
        headers = self._get_headers(tr_id, hashkey_val=hash_val)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            order_type = "시장가" if ord_dvsn == "01" else f"지정가({price}원)"
            logger.info(f"{code} {qty}주 매도 주문({order_type}) 요청")
            
            res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            res_json = self._handle_response(res, "매도 주문 실패")
            
            if not res_json:
                return False
            
            if res_json.get("rt_cd") == "0":
                order_no = res_json.get("output", {}).get("ODNO", "알 수 없음")
                success_msg = f"[매도 성공] {code} {qty}주 {order_type} (주문번호: {order_no})"
                logger.info(success_msg)
                send_message(success_msg)
                return True
            else:
                error_msg = f"[매도 실패] {res_json}"
                logger.error(error_msg)
                send_message(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"매도 주문 중 예외 발생: {e}"
            logger.error(error_msg, exc_info=True)
            send_message(f"[오류] 매도 주문 실패: {e}")
            return False
    
    def modify_order(self, org_order_no: str, code: str, qty: int, price: int, order_type: str = "00") -> bool:
        """
        주문 정정 요청
        
        Args:
            org_order_no (str): 원주문번호
            code (str): 종목코드
            qty (int): 정정 수량
            price (int): 정정 가격
            order_type (str): 주문구분(00: 지정가, 01: 시장가)
            
        Returns:
            bool: 정정 성공 여부
            
        Notes:
            모의투자 지원 함수입니다.
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/trading/order-rvsecncl"
        url = f"{URL_BASE}/{path}"
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC0803U" if USE_REALTIME_API else "VTTC0803U"
        
        data = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
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
        hash_val = hashkey(data)
        headers = self._get_headers(tr_id, hashkey_val=hash_val)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"주문 정정 요청: 원주문번호 {org_order_no}, 종목 {code}, 수량 {qty}주, 가격 {price}원")
            res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            result = self._handle_response(res, "주문 정정 실패")
            
            if not result:
                return False
            
            if result.get("rt_cd") == "0":
                output = result.get("output", {})
                new_order_no = output.get("ODNO", "알 수 없음")
                success_msg = f"주문 정정 성공: {code} {qty}주 {price}원 (원주문번호: {org_order_no}, 신규주문번호: {new_order_no})"
                logger.info(success_msg)
                send_message(success_msg)
                return True
            else:
                error_msg = f"주문 정정 실패: {result}"
                logger.error(error_msg)
                send_message(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"주문 정정 중 예외 발생: {e}"
            logger.error(error_msg, exc_info=True)
            send_message(f"[오류] 주문 정정 실패: {e}")
            return False
    
    def cancel_order(self, order_no: str, code: str, qty: int) -> bool:
        """
        주문 취소 요청
        
        Args:
            order_no (str): 원주문번호
            code (str): 종목코드
            qty (int): 취소 수량
            
        Returns:
            bool: 취소 성공 여부
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/trading/order-rvsecncl"
        url = f"{URL_BASE}/{path}"
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC0803U" if USE_REALTIME_API else "VTTC0803U"
        
        data = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": order_no,
            "RVSE_CNCL_DVSN_CD": "02",  # 취소(02)
            "ORD_DVSN": "00",
            "PDNO": code,
            "ORD_QTY": str(qty),
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "N"
        }
        
        # 해시키 생성
        hash_val = hashkey(data)
        headers = self._get_headers(tr_id, hashkey_val=hash_val)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"주문 취소 요청: 주문번호 {order_no}, 종목 {code}, 수량 {qty}")
            res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            result = self._handle_response(res, "주문 취소 실패")
            
            if not result:
                return False
            
            if result.get("rt_cd") == "0":
                output = result.get("output", {})
                new_order_no = output.get("ODNO", "알 수 없음")
                success_msg = f"주문 취소 성공: {code} {qty}주 (원주문번호: {order_no}, 신규주문번호: {new_order_no})"
                logger.info(success_msg)
                send_message(success_msg)
                return True
            else:
                error_msg = f"주문 취소 실패: {result}"
                logger.error(error_msg)
                send_message(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"주문 취소 중 예외 발생: {e}"
            logger.error(error_msg, exc_info=True)
            send_message(f"[오류] 주문 취소 실패: {e}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """
        모든 미체결 주문 취소
        
        Returns:
            bool: 전체 취소 성공 여부
            
        Notes:
            모의투자에서는 지원되지 않습니다.
        """
        self: KoreaInvestmentApiClient  # type hint
        
        # 미체결 주문 조회
        pending_orders = self.get_pending_orders()
        
        if pending_orders is None:
            logger.error("미체결 주문 조회에 실패하여 전체 취소를 진행할 수 없습니다.")
            send_message("[오류] 미체결 주문 조회에 실패하여 전체 취소를 진행할 수 없습니다.")
            return False
        
        if not pending_orders:
            logger.info("취소할 미체결 주문이 없습니다.")
            return True
        
        success_count = 0
        for order in pending_orders:
            if self.cancel_order(order["order_no"], order["code"], order["remaining_qty"]):
                success_count += 1
        
        if success_count == len(pending_orders):
            logger.info(f"모든 미체결 주문 {success_count}건 취소 성공")
            send_message(f"[알림] 모든 미체결 주문 {success_count}건 취소 성공")
            return True
        else:
            logger.warning(f"미체결 주문 취소 일부 실패: {success_count}/{len(pending_orders)}건 성공")
            send_message(f"[경고] 미체결 주문 취소 일부 실패: {success_count}/{len(pending_orders)}건 성공")
            return False
    
    def fetch_buyable_amount(self, code: str, price: int = 0) -> Optional[Dict[str, Any]]:
        """
        매수 가능 금액 조회
        
        Args:
            code (str): 종목 코드
            price (int): 주문 단가
            
        Returns:
            dict or None: 매수 가능 금액 정보
            
        Notes:
            모의투자 지원 함수입니다.
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        url = f"{URL_BASE}/{path}"
        
        tr_id = "TTTC8908R" if USE_REALTIME_API else "VTTC8908R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "PDNO": code,
            "ORD_UNPR": str(price),
            "ORD_DVSN": "02",  # 시장가
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
            buyable_info = {
                "max_amount": int(output.get("nrcvb_buy_amt", "0").replace(',', '')),
                "max_qty": int(output.get("max_buy_qty", "0").replace(',', '')),
                "price": int(output.get("buyable_price", "0").replace(',', '')) if output.get("buyable_price") else price
            }
            
            logger.info(f"{code} 매수 가능 금액 조회 성공: {buyable_info['max_amount']:,}원, {buyable_info['max_qty']:,}주")
            return buyable_info
            
        except Exception as e:
            logger.error(f"{code} 매수 가능 금액 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 매수 가능 금액 조회 실패: {e}")
            return None
    
    def fetch_sellable_quantity(self, code: str) -> Optional[Dict[str, Any]]:
        """
        매도 가능 수량 조회
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict or None: 매도 가능 수량 정보
            
        Notes:
            모의투자 지원 함수입니다.
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/trading/inquire-psbl-sell"
        url = f"{URL_BASE}/{path}"
        
        tr_id = "TTTC8209R" if USE_REALTIME_API else "VTTC8209R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "PDNO": code,
            "UNPR": "0",
            "SLL_TYPE": "01",  # 보통
            "ORD_DVSN": "01"   # 시장가
        }
        
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} 매도 가능 수량 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} 매도 가능 수량 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
            
            output = result.get("output", {})
            sellable_info = {
                "code": code,
                "name": output.get("prdt_name", ""),
                "sellable_qty": int(output.get("ord_psbl_qty", "0").replace(',', '')),
                "hldg_qty": int(output.get("hldg_qty", "0").replace(',', '')),
                "avg_purchase_price": int(output.get("pchs_avg_pric", "0").replace(',', ''))
            }
            
            logger.info(f"{code} 매도 가능 수량 조회 성공: {sellable_info['sellable_qty']:,}주")
            return sellable_info
            
        except Exception as e:
            logger.error(f"{code} 매도 가능 수량 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 매도 가능 수량 조회 실패: {e}")
            return None 