"""
한국 주식 자동매매 - 주문 정보 조회 모듈

미체결 주문, 일별 주문 내역 등 주문 정보 조회 관련 기능을 제공합니다.
"""

import json
import requests
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING, cast

from korea_stock_auto.config import (
    URL_BASE, CANO, ACNT_PRDT_CD, USE_REALTIME_API
)
from korea_stock_auto.utils.utils import send_message

# 타입 힌트만을 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

# 로깅 설정
logger = logging.getLogger("stock_auto")

class OrderInfoMixin:
    """
    주문 정보 조회 관련 기능 Mixin
    
    미체결 주문, 일별 주문 내역 등 주문 정보 조회 관련 기능을 제공합니다.
    """
    
    def get_pending_orders(self, code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        미체결 주문 조회
        
        Args:
            code: 종목 코드 (특정 종목만 조회, None인 경우 전체 조회)
            
        Returns:
            List[Dict[str, Any]]: 미체결 주문 목록
            
        Notes:
            모의투자 지원 함수입니다.
            
        Examples:
            >>> api_client.get_pending_orders()  # 전체 미체결 주문 조회
            >>> api_client.get_pending_orders("005930")  # 삼성전자 미체결 주문만 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl"
        url = f"{URL_BASE}/{path}"
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC8036R" if USE_REALTIME_API else "VTTC8036R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
            "INQR_DVSN_1": "0",
            "INQR_DVSN_2": "0"
        }
        
        if code:
            params["PDNO"] = code
        
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"미체결 주문 조회 요청 {code or '(전체)'}")
            
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, "미체결 주문 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return []
                
            output1 = result.get("output1", [])
            
            if not output1:
                logger.info("미체결 주문이 없습니다.")
                return []
            
            # 결과 가공
            pending_orders = []
            for item in output1:
                order_info = {
                    "order_no": item.get("odno", ""),
                    "order_time": item.get("ord_tmd", ""),
                    "code": item.get("pdno", ""),
                    "name": item.get("prdt_name", ""),
                    "order_type": item.get("sll_buy_dvsn_cd_name", ""),
                    "order_price": int(item.get("ord_unpr", "0").replace(',', '')),
                    "order_qty": int(item.get("ord_qty", "0").replace(',', '')),
                    "executed_qty": int(item.get("tot_ccld_qty", "0").replace(',', '')),
                    "remaining_qty": int(item.get("psbl_qty", "0").replace(',', '')),
                    "order_status": item.get("status_name", "")
                }
                pending_orders.append(order_info)
            
            logger.info(f"미체결 주문 조회 성공: {len(pending_orders)}건")
            return pending_orders
            
        except Exception as e:
            logger.error(f"미체결 주문 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 미체결 주문 조회 실패: {e}")
            return []
    
    def get_daily_orders(self, date: Optional[str] = None, code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        일별 주문 내역 조회
        
        Args:
            date: 조회 날짜 (YYYYMMDD 형식, None인 경우 당일)
            code: 종목 코드 (특정 종목만 조회, None인 경우 전체 조회)
            
        Returns:
            List[Dict[str, Any]]: 주문 내역 목록
            
        Notes:
            모의투자 지원 함수입니다.
            
        Examples:
            >>> api_client.get_daily_orders()  # 당일 전체 주문 내역 조회
            >>> api_client.get_daily_orders("20230901")  # 2023년 9월 1일 주문 내역 조회
            >>> api_client.get_daily_orders(code="005930")  # 삼성전자 당일 주문 내역 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        import datetime
        if date is None:
            date = datetime.datetime.now().strftime("%Y%m%d")
        
        path = "uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        url = f"{URL_BASE}/{path}"
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC8001R" if USE_REALTIME_API else "VTTC8001R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "INQR_STRT_DT": date,
            "INQR_END_DT": date,
            "SLL_BUY_DVSN_CD": "00",  # 전체(00), 매도(01), 매수(02)
            "INQR_DVSN": "00", # 역순(00), 정순(01)
            "PDNO": code or "",
            "CCLD_DVSN": "00", # 전체(00), 체결(01), 미체결(02)
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_3": "",
            "INQR_DVSN_1": ""
        }
        
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{date} 주문 내역 조회 요청 {code or '(전체)'}")
            
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{date} 주문 내역 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return []
                
            output = result.get("output", [])
            
            if not output:
                logger.info(f"{date} 주문 내역이 없습니다.")
                return []
            
            # 결과 가공
            orders = []
            for item in output:
                order_info = {
                    "order_no": item.get("odno", ""),
                    "order_time": item.get("ord_tmd", ""),
                    "execution_time": item.get("ccld_tmd", ""),
                    "code": item.get("pdno", ""),
                    "name": item.get("prdt_name", ""),
                    "order_type": item.get("sll_buy_dvsn_cd_name", ""),
                    "order_price": int(item.get("ord_unpr", "0").replace(',', '')),
                    "executed_price": int(item.get("avg_prvs", "0").replace(',', '')),
                    "order_qty": int(item.get("ord_qty", "0").replace(',', '')),
                    "executed_qty": int(item.get("tot_ccld_qty", "0").replace(',', '')),
                    "remaining_qty": int(item.get("rmn_qty", "0").replace(',', '')),
                    "order_status": item.get("status_name", ""),
                    "is_cancel": item.get("cncl_yn", "") == "Y"
                }
                orders.append(order_info)
            
            logger.info(f"{date} 주문 내역 조회 성공: {len(orders)}건")
            return orders
            
        except Exception as e:
            logger.error(f"{date} 주문 내역 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {date} 주문 내역 조회 실패: {e}")
            return [] 