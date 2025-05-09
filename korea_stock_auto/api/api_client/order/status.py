"""
한국 주식 자동매매 - 주문 상태 조회 모듈
"""

import time
import requests
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from korea_stock_auto.config import (
    URL_BASE, CANO, ACNT_PRDT_CD, USE_REALTIME_API
)
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

logger = logging.getLogger("stock_auto")

class OrderStatusMixin:
    """주문 상태 조회 관련 기능 Mixin"""
    
    def get_order_status(self, order_no: str) -> Optional[Dict[str, Any]]:
        """
        주문 체결 상태 조회
        
        Args:
            order_no (str): 주문 번호
            
        Returns:
            dict or None: 주문 체결 상태 정보
        """
        self: KoreaInvestmentApiClient  # type hint
        
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
        
        Returns:
            list or None: 미체결 주문 목록
            
        Notes:
            모의투자에서는 지원되지 않습니다.
        """
        self: KoreaInvestmentApiClient  # type hint
        
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
            order_date (str, optional): 주문일자(YYYYMMDD), 기본값은 당일
            
        Returns:
            list or None: 체결 내역 목록
            
        Notes:
            모의투자 지원 함수입니다.
        """
        self: KoreaInvestmentApiClient  # type hint
        
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
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
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
                    "executed_qty": int(item.get("ccld_qty", "0").replace(',', '')),
                    "executed_price": int(item.get("avg_pvt", "0").replace(',', '')),
                    "order_type": item.get("ord_dvsn_name", ""),
                    "executed_time": item.get("ccld_tm", ""),
                    "executed_date": item.get("ord_dt", ""),
                    "buy_sell_type": "매수" if item.get("sll_buy_dvsn_cd", "") == "02" else "매도",
                    "amount": int(item.get("tot_ccld_amt", "0").replace(',', '')),
                    "fee": int(item.get("tot_ccld_amt", "0").replace(',', '')) * 0.0035  # 거래수수료 약 0.35% (추정값)
                }
                executed_orders.append(order_info)
            
            logger.info(f"{order_date} 체결 내역 {len(executed_orders)}건 조회 성공")
            return executed_orders
            
        except Exception as e:
            logger.error(f"체결 내역 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 체결 내역 조회 실패: {e}")
            return None
    
    def get_profit_loss(self, from_date: str, to_date: str = None) -> Optional[Dict[str, Any]]:
        """
        기간별 손익 조회
        
        Args:
            from_date (str): 시작일(YYYYMMDD)
            to_date (str, optional): 종료일(YYYYMMDD), 기본값은 당일
            
        Returns:
            dict or None: 기간별 손익 정보
            
        Notes:
            모의투자에서는 지원되지 않습니다.
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/trading/inquire-account-profit-loss"
        url = f"{URL_BASE}/{path}"
        
        # 모의투자에서는 기능 미지원
        if not USE_REALTIME_API:
            logger.warning("기간별 손익 조회 기능은 모의투자에서 지원되지 않습니다.")
            send_message("[안내] 기간별 손익 조회 기능은 모의투자에서 지원되지 않습니다.")
            return None
        
        tr_id = "TTTC8494R"
        
        # 종료일이 없으면 당일로 설정
        if not to_date:
            to_date = datetime.now().strftime("%Y%m%d")
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "INQR_STRT_DT": from_date,
            "INQR_END_DT": to_date,
            "INQR_DVSN": "00",  # 조회구분 - 00:기간별, 01:분기별
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"기간별 손익 조회 요청: {from_date} ~ {to_date}")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, "기간별 손익 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
            
            output1 = result.get("output1", {})
            
            profit_loss = {
                "total_purchase_amount": int(output1.get("asst_icdc_amt", "0").replace(',', '')),
                "total_evaluation_amount": int(output1.get("tot_evlu_amt", "0").replace(',', '')),
                "total_profit_loss": int(output1.get("tot_pfls_amt", "0").replace(',', '')),
                "total_profit_rate": float(output1.get("tot_pfls_amt_rate", "0").replace(',', '')),
                "deposit_increase": int(output1.get("dnca_adnt_icdc_amt", "0").replace(',', '')),
                "deposit_decrease": int(output1.get("dnca_adnt_dcrc_amt", "0").replace(',', '')),
                "stock_transaction_amount": int(output1.get("scts_trpf_amt", "0").replace(',', '')),
                "tax_and_fee": int(output1.get("trxs_csts_amt", "0").replace(',', '')),
                "period": f"{from_date} ~ {to_date}"
            }
            
            logger.info(f"기간별 손익 조회 성공: {profit_loss['total_profit_loss']:,}원 ({profit_loss['total_profit_rate']}%)")
            return profit_loss
            
        except Exception as e:
            logger.error(f"기간별 손익 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 기간별 손익 조회 실패: {e}")
            return None
    
    def get_realized_profit_loss(self, from_date: str, to_date: str = None) -> Optional[Dict[str, Any]]:
        """
        실현손익 조회
        
        Args:
            from_date (str): 시작일(YYYYMMDD)
            to_date (str, optional): 종료일(YYYYMMDD), 기본값은 당일
            
        Returns:
            dict or None: 실현손익 정보
            
        Notes:
            모의투자에서는 지원되지 않습니다.
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/trading/inquire-trade-profit"
        url = f"{URL_BASE}/{path}"
        
        # 모의투자에서는 기능 미지원
        if not USE_REALTIME_API:
            logger.warning("실현손익 조회 기능은 모의투자에서 지원되지 않습니다.")
            send_message("[안내] 실현손익 조회 기능은 모의투자에서 지원되지 않습니다.")
            return None
        
        tr_id = "TTTC8434R"
        
        # 종료일이 없으면 당일로 설정
        if not to_date:
            to_date = datetime.now().strftime("%Y%m%d")
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "PDNO": "",  # 전체 종목
            "ORD_STRT_DT": from_date,
            "ORD_END_DT": to_date,
            "SLL_BUY_DVSN": "00",  # 전체(00), 매도(01), 매수(02)
            "CCLD_NCCS_DVSN": "01",  # 정산 완료분
            "INQR_DVSN": "00",  # 조회 구분(00: 전체)
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"실현손익 조회 요청: {from_date} ~ {to_date}")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, "실현손익 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
            
            output = result.get("output", [])
            
            # 결과 가공
            total_profit = 0
            total_loss = 0
            trade_count = 0
            win_count = 0
            
            trades = []
            for item in output:
                profit = int(item.get("rofee_amt_smtl", "0").replace(',', ''))
                trade_info = {
                    "code": item.get("pdno", ""),
                    "name": item.get("prdt_name", ""),
                    "quantity": int(item.get("sll_qty", "0").replace(',', '')),
                    "buy_price": int(item.get("pchs_avg_pric", "0").replace(',', '')),
                    "sell_price": int(item.get("sll_prc", "0").replace(',', '')),
                    "profit": profit,
                    "profit_rate": float(item.get("evlu_pfls_rt", "0").replace(',', '')),
                    "trade_date": item.get("tot_ccld_amt", "")
                }
                trades.append(trade_info)
                
                trade_count += 1
                if profit > 0:
                    total_profit += profit
                    win_count += 1
                else:
                    total_loss += profit  # profit is negative for losses
            
            win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
            
            realized_profit_loss = {
                "trades": trades,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "net_profit": total_profit + total_loss,
                "trade_count": trade_count,
                "win_count": win_count,
                "win_rate": win_rate,
                "period": f"{from_date} ~ {to_date}"
            }
            
            logger.info(f"실현손익 조회 성공: 거래 {trade_count}건, 승률 {win_rate:.2f}%, 순손익 {realized_profit_loss['net_profit']:,}원")
            return realized_profit_loss
            
        except Exception as e:
            logger.error(f"실현손익 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 실현손익 조회 실패: {e}")
            return None 