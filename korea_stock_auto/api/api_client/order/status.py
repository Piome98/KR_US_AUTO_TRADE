"""
한국 주식 자동매매 - 주문 상태 조회 모듈

주문 체결 상태, 미체결 주문, 체결 내역 등 주문 상태 관련 조회 기능을 제공합니다.
"""

import requests
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING, cast
from datetime import datetime

from korea_stock_auto.config import (
    URL_BASE, CANO, ACNT_PRDT_CD, USE_REALTIME_API
)
from korea_stock_auto.utils.utils import send_message

# 타입 힌트만을 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

# 로깅 설정
logger = logging.getLogger("stock_auto")

class OrderStatusMixin:
    """
    주문 상태 조회 관련 기능 Mixin
    
    주문 체결 상태, 미체결 주문, 체결 내역 등 주문 상태 관련 조회 기능을 제공합니다.
    """
    
    def get_order_status(self, order_no: str) -> Optional[Dict[str, Any]]:
        """
        주문 체결 상태 조회
        
        Args:
            order_no: 주문 번호
            
        Returns:
            dict or None: 주문 체결 상태 정보 (실패 시 None)
            
        Examples:
            >>> api_client.get_order_status("XXXXXXXX")  # 특정 주문의 체결 상태 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/trading/inquire-order"
        url = f"{URL_BASE}/{path}"
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC8036R" if USE_REALTIME_API else "VTTC8036R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "INQR_DVSN_1": "0",
            "INQR_DVSN_2": "0",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
            "ODNO": order_no
        }
        
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"주문번호 {order_no} 체결 상태 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"주문 체결 상태 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", [])
            if not output:
                logger.warning(f"주문번호 {order_no}의 체결 정보가 없습니다.")
                return None
            
            # 결과 가공
            order_info = {
                "order_no": order_no,
                "code": output[0].get("pdno", ""),
                "name": output[0].get("prdt_name", ""),
                "order_qty": int(output[0].get("ord_qty", "0").replace(',', '')),
                "executed_qty": int(output[0].get("tot_ccld_qty", "0").replace(',', '')),
                "remaining_qty": int(output[0].get("rmn_qty", "0").replace(',', '')),
                "executed_price": int(output[0].get("avg_prvs", "0").replace(',', '')),
                "order_status": output[0].get("status_name", ""),
                "order_type": output[0].get("ord_dvsn_name", ""),
                "order_time": output[0].get("ord_tm", "")
            }
            
            logger.info(f"주문번호 {order_no} 체결 상태 조회 성공")
            return order_info
            
        except Exception as e:
            logger.error(f"주문 체결 상태 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 주문 체결 상태 조회 실패: {e}")
            return None
    
    def get_pending_orders(self) -> Optional[List[Dict[str, Any]]]:
        """
        미체결 주문 조회
        
        현재 계좌에 존재하는 미체결 주문 목록을 조회합니다.
        
        Returns:
            list or None: 미체결 주문 목록 (실패 시 None)
            
        Notes:
            모의투자에서는 지원되지 않습니다.
            
        Examples:
            >>> api_client.get_pending_orders()  # 미체결 주문 목록 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl"
        url = f"{URL_BASE}/{path}"
        
        # 모의투자에서는 기능 미지원
        if not USE_REALTIME_API:
            logger.warning("미체결 주문 조회 기능은 모의투자에서 지원되지 않습니다.")
            send_message("[안내] 미체결 주문 조회 기능은 모의투자에서 지원되지 않습니다.")
            return None
        
        tr_id = "TTTC8036R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
            "INQR_DVSN_1": "0", # 조회구분 전체:0, 매도:1, 매수:2
            "INQR_DVSN_2": "0"  # 조회구분2 전체:0, 체결:1, 미체결:2
        }
        
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info("미체결 주문 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, "미체결 주문 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
            
            output = result.get("output", [])
            if not output:
                logger.info("미체결 주문이 없습니다.")
                return []
            
            # 결과 가공
            pending_orders = []
            for item in output:
                if int(item.get("rmn_qty", "0").replace(',', '')) > 0:  # 미체결 수량이 있는 주문만 처리
                    order_info = {
                        "order_no": item.get("odno", ""),
                        "code": item.get("pdno", ""),
                        "name": item.get("prdt_name", ""),
                        "order_qty": int(item.get("ord_qty", "0").replace(',', '')),
                        "executed_qty": int(item.get("tot_ccld_qty", "0").replace(',', '')),
                        "remaining_qty": int(item.get("rmn_qty", "0").replace(',', '')),
                        "order_price": int(item.get("ord_unpr", "0").replace(',', '')),
                        "current_price": int(item.get("prpr", "0").replace(',', '')),
                        "order_type": item.get("ord_dvsn_name", ""),
                        "order_time": item.get("ord_tmd", ""),
                        "order_date": item.get("ord_dt", ""),
                        "order_status": item.get("status_name", ""),
                        "buy_sell_type": "매수" if item.get("sll_buy_dvsn_cd", "") == "02" else "매도"
                    }
                    pending_orders.append(order_info)
            
            logger.info(f"미체결 주문 {len(pending_orders)}건 조회 성공")
            return pending_orders
            
        except Exception as e:
            logger.error(f"미체결 주문 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 미체결 주문 조회 실패: {e}")
            return None
    
    def get_executed_orders(self, order_date: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        체결 내역 조회
        
        Args:
            order_date: 주문일자(YYYYMMDD), 기본값은 당일
            
        Returns:
            list or None: 체결 내역 목록 (실패 시 None)
            
        Notes:
            모의투자 지원 함수입니다.
            
        Examples:
            >>> api_client.get_executed_orders()  # 당일 체결 내역 조회
            >>> api_client.get_executed_orders("20230101")  # 특정 날짜 체결 내역 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        url = f"{URL_BASE}/{path}"
        
        tr_id = "TTTC8001R" if USE_REALTIME_API else "VTTC8001R"
        
        # 주문일자가 없으면 당일로 설정
        if not order_date:
            order_date = datetime.now().strftime("%Y%m%d")
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "INQR_STRT_DT": order_date,
            "INQR_END_DT": order_date,
            "SLL_BUY_DVSN_CD": "00",  # 전체
            "INQR_DVSN": "00",  # 역순
            "PDNO": "",
            "CCLD_DVSN": "00",  # 전체
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_3": "00",  # 전체
            "INQR_DVSN_1": "",
            "INQR_DVSN_2": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{order_date} 체결 내역 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, "체결 내역 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", [])
            if not output:
                logger.info(f"{order_date} 체결 내역이 없습니다.")
                return []
            
            # 결과 가공
            executed_orders = []
            for item in output:
                order_info = {
                    "order_no": item.get("odno", ""),
                    "code": item.get("pdno", ""),
                    "name": item.get("prdt_name", ""),
                    "order_qty": int(item.get("ord_qty", "0").replace(',', '')),
                    "executed_qty": int(item.get("tot_ccld_qty", "0").replace(',', '')),
                    "executed_price": int(item.get("avg_prvs", "0").replace(',', '')),
                    "order_type": item.get("ord_dvsn_name", ""),
                    "order_time": item.get("ord_tmd", ""),
                    "executed_time": item.get("ccld_cnfr_tm", ""),
                    "buy_sell_type": "매수" if item.get("sll_buy_dvsn_cd", "") == "02" else "매도",
                    "trade_amount": int(item.get("tot_ccld_amt", "0").replace(',', '')),
                    "fee": int(item.get("cmis", "0").replace(',', '')),
                    "tax": int(item.get("tax", "0").replace(',', '')),
                    "net_amount": int(item.get("tot_ccld_amt", "0").replace(',', '')) - 
                               int(item.get("cmis", "0").replace(',', '')) - 
                               int(item.get("tax", "0").replace(',', ''))
                }
                executed_orders.append(order_info)
            
            logger.info(f"{order_date} 체결 내역 {len(executed_orders)}건 조회 성공")
            return executed_orders
            
        except Exception as e:
            logger.error(f"체결 내역 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 체결 내역 조회 실패: {e}")
            return None
    
    def get_profit_loss(self, from_date: str, to_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        기간별 손익 현황 조회
        
        Args:
            from_date: 시작일(YYYYMMDD)
            to_date: 종료일(YYYYMMDD), 기본값은 당일
            
        Returns:
            dict or None: 손익 현황 정보 (실패 시 None)
            
        Notes:
            모의투자 지원 함수입니다.
            
        Examples:
            >>> api_client.get_profit_loss("20230101")  # 2023년 1월 1일부터 현재까지의 손익 조회
            >>> api_client.get_profit_loss("20230101", "20230131")  # 2023년 1월 한 달간의 손익 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # 종료일이 없으면 당일로 설정
        if not to_date:
            to_date = datetime.now().strftime("%Y%m%d")
        
        path = "uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{URL_BASE}/{path}"
        
        tr_id = "TTTC8434R" if USE_REALTIME_API else "VTTC8434R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "N",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"손익 현황 조회 요청 ({from_date} ~ {to_date})")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, "손익 현황 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output1 = result.get("output1", {})
            output2 = result.get("output2", [])
            
            # 결과 가공
            profit_loss = {
                "period": f"{from_date} ~ {to_date}",
                "total_asset": int(output1.get("tot_evlu_amt", "0").replace(',', '')),
                "total_buy_amount": int(output1.get("pchs_amt_smtl_amt", "0").replace(',', '')),
                "total_eval_amount": int(output1.get("evlu_amt_smtl_amt", "0").replace(',', '')),
                "total_profit_loss": int(output1.get("evlu_pfls_smtl_amt", "0").replace(',', '')),
                "total_profit_loss_rate": float(output1.get("evlu_pfls_rt", "0").replace(',', '')),
                "total_purchase_fee": int(output1.get("pchs_fee_smtl_amt", "0").replace(',', '')),
                "total_holding_qty": int(output1.get("hldg_qty_smtl_qty", "0").replace(',', '')),
                "stocks": []
            }
            
            # 종목별 손익 정보
            for item in output2:
                stock_info = {
                    "code": item.get("pdno", ""),
                    "name": item.get("prdt_name", ""),
                    "holding_qty": int(item.get("hldg_qty", "0").replace(',', '')),
                    "avg_buy_price": int(item.get("pchs_avg_pric", "0").replace(',', '')),
                    "current_price": int(item.get("prpr", "0").replace(',', '')),
                    "buy_amount": int(item.get("pchs_amt", "0").replace(',', '')),
                    "eval_amount": int(item.get("evlu_amt", "0").replace(',', '')),
                    "profit_loss": int(item.get("evlu_pfls_amt", "0").replace(',', '')),
                    "profit_loss_rate": float(item.get("evlu_pfls_rt", "0").replace(',', '')),
                    "purchase_fee": int(item.get("pchs_fee", "0").replace(',', '')),
                    "holding_rate": float(item.get("evlu_amt_rt", "0").replace(',', ''))
                }
                profit_loss["stocks"].append(stock_info)
            
            logger.info(f"손익 현황 조회 성공: 총 {len(profit_loss['stocks'])}종목, 총손익 {profit_loss['total_profit_loss']:,}원 ({profit_loss['total_profit_loss_rate']}%)")
            return profit_loss
            
        except Exception as e:
            logger.error(f"손익 현황 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 손익 현황 조회 실패: {e}")
            return None
    
    def get_realized_profit_loss(self, from_date: str, to_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        실현 손익 조회
        
        특정 기간 동안의 매매를 통해 실현된 손익을 조회합니다.
        
        Args:
            from_date: 시작일(YYYYMMDD)
            to_date: 종료일(YYYYMMDD), 기본값은 당일
            
        Returns:
            dict or None: 실현 손익 정보 (실패 시 None)
            
        Notes:
            모의투자 지원 함수입니다.
            
        Examples:
            >>> api_client.get_realized_profit_loss("20230101")  # 2023년 1월 1일부터 현재까지의 실현 손익 조회
            >>> api_client.get_realized_profit_loss("20230101", "20230131")  # 2023년 1월 한 달간의 실현 손익 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # 종료일이 없으면 당일로 설정
        if not to_date:
            to_date = datetime.now().strftime("%Y%m%d")
        
        path = "uapi/domestic-stock/v1/trading/inquire-ccnl"
        url = f"{URL_BASE}/{path}"
        
        tr_id = "TTTC8494R" if USE_REALTIME_API else "VTTC8494R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "INQR_STRT_DT": from_date,
            "INQR_END_DT": to_date,
            "SLL_BUY_DVSN_CD": "00",  # 전체
            "CCLD_TYPE": "00",  # 전체
            "INQR_DVSN": "00",  # 역순
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"실현 손익 조회 요청 ({from_date} ~ {to_date})")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, "실현 손익 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output1 = result.get("output1", {})
            output2 = result.get("output2", [])
            
            # 결과 가공
            realized_profit_loss = {
                "period": f"{from_date} ~ {to_date}",
                "total_buy_amount": int(output1.get("pchs_amt_smtl_amt", "0").replace(',', '')),
                "total_sell_amount": int(output1.get("sll_amt_smtl_amt", "0").replace(',', '')),
                "total_profit_loss": int(output1.get("ccld_pfls_smtl_amt", "0").replace(',', '')),
                "total_fee": int(output1.get("tot_fee_amt_smtl_amt", "0").replace(',', '')),
                "total_tax": int(output1.get("tot_tax_amt_smtl_amt", "0").replace(',', '')),
                "transactions": []
            }
            
            # 거래별 손익 정보
            for item in output2:
                transaction_info = {
                    "code": item.get("pdno", ""),
                    "name": item.get("prdt_name", ""),
                    "order_date": item.get("ord_dt", ""),
                    "order_time": item.get("ord_tmd", ""),
                    "executed_date": item.get("sll_buy_ccld_dtm", ""),
                    "buy_sell_type": "매수" if item.get("sll_buy_dvsn_cd", "") == "02" else "매도",
                    "executed_qty": int(item.get("ccld_qty", "0").replace(',', '')),
                    "executed_price": int(item.get("ccld_unpr", "0").replace(',', '')),
                    "trade_amount": int(item.get("ccld_amt", "0").replace(',', '')),
                    "fee": int(item.get("fee_amt", "0").replace(',', '')),
                    "tax": int(item.get("tax_amt", "0").replace(',', '')),
                    "profit_loss": int(item.get("ccld_pfls_amt", "0").replace(',', ''))
                }
                realized_profit_loss["transactions"].append(transaction_info)
            
            logger.info(f"실현 손익 조회 성공: 총 {len(realized_profit_loss['transactions'])}건, 총손익 {realized_profit_loss['total_profit_loss']:,}원")
            return realized_profit_loss
            
        except Exception as e:
            logger.error(f"실현 손익 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 실현 손익 조회 실패: {e}")
            return None 