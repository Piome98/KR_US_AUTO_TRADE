"""
한국 주식 자동매매 - 계좌 예수금 모듈
계좌 예수금 및 일별 잔고 조회 기능 제공
"""

import logging
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING, cast

from korea_stock_auto.config import (
    URL_BASE, CANO, ACNT_PRDT_CD, USE_REALTIME_API
)
from korea_stock_auto.utils.utils import send_message

# 타입 힌트만을 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

# 로깅 설정
logger = logging.getLogger("stock_auto")

class AccountDepositMixin:
    """
    계좌 예수금 관련 기능 Mixin
    
    계좌 예수금 조회, 일별 잔고 조회 등의 기능을 제공합니다.
    """
    
    def fetch_deposit(self) -> Optional[Dict[str, Any]]:
        """
        계좌 예수금 조회
        
        한국투자증권 API를 통해 계좌 예수금 정보를 조회합니다.
        
        Returns:
            dict or None: 계좌 예수금 정보 또는 조회 실패 시 None
            
        Notes:
            모의투자와 실전투자 모두 지원하는 함수입니다.
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/trading/inquire-psbl-order"
        url = f"{URL_BASE}/{path}"
        
        # 실전/모의투자 구분
        tr_id = "TTTC8908R" if USE_REALTIME_API else "VTTC8908R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "PDNO": "005930",  # 삼성전자를 기준으로 조회 (다른 종목 코드여도 무관)
            "ORD_UNPR": "65500",  # 가격은 의미 없음
            "ORD_DVSN": "01",  # 시장가
            "CMA_EVLU_AMT_ICLD_YN": "Y",  # CMA평가금액포함여부
            "OVRS_ICLD_YN": "N"  # 해외포함여부
        }
        
        try:
            headers = self._get_headers(tr_id)
            
            logger.info("계좌 예수금 조회 요청")
            result = self._request_get(url, headers, params, "계좌 예수금 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                logger.error("계좌 예수금 조회 결과가 유효하지 않습니다.")
                return None
                
            output = result.get("output", {})
            
            # 결과 가공 - 사용 가능한 예수금 정보
            deposit_info = {
                "d1_deposit": int(output.get("ord_psbl_cash", "0").replace(',', '')),  # D+1 예수금
                "d2_deposit": int(output.get("psbl_amt", "0").replace(',', '')),  # D+2 예수금
                "withdrawable": int(output.get("wdrl_psbl_amt", "0").replace(',', '')),  # 출금 가능 금액
                "order_executable_amount": int(output.get("nrcvb_buy_amt", "0").replace(',', '')),  # 매수 가능 금액
                "total_balance": int(output.get("tot_evlu_amt", "0").replace(',', '')),  # 총 평가 금액
                "deposit_used": int(output.get("scts_evlu_amt", "0").replace(',', '')),  # 유가증권 평가 금액
                "credit_order_amount": int(output.get("crdtl_ord_amt", "0").replace(',', ''))  # 신용 주문 가능 금액
            }
            
            logger.info(f"계좌 예수금 조회 성공: 주문 가능 금액 {deposit_info['order_executable_amount']:,}원")
            return deposit_info
            
        except Exception as e:
            logger.error(f"계좌 예수금 조회 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] 계좌 예수금 조회 실패: {e}")
            return None
    
    def fetch_daily_balance(self) -> Optional[Dict[str, Any]]:
        """
        계좌 일별 잔고 조회
        
        한국투자증권 API를 통해 계좌의 일별 잔고 정보를 조회합니다.
        
        Returns:
            dict or None: {
                'summary': {계좌 종합 정보},
                'stocks': [보유 종목 목록],
                'stock_count': 보유 종목 수
            } 또는 조회 실패 시 None
            
        Notes:
            모의투자와 실전투자 모두 지원하는 함수입니다.
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/trading/inquire-daily-balance"
        url = f"{URL_BASE}/{path}"
        
        # 실전/모의투자 구분
        tr_id = "TTTC8434R" if USE_REALTIME_API else "VTTC8434R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "INQR_DVSN_1": "1",  # 조회구분 1 (1:조회순서 2:종목코드순)
            "INQR_DVSN_2": "1",  # 조회구분 2 (1:순번 2:주식잔고 3:종목평가)
            "CTX_AREA_FK100": "",  # 연속조회검색조건
            "CTX_AREA_NK100": ""   # 연속조회키
        }
        
        try:
            headers = self._get_headers(tr_id)
            
            logger.info("계좌 일별 잔고 조회 요청")
            result = self._request_get(url, headers, params, "계좌 일별 잔고 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                logger.error("계좌 일별 잔고 조회 결과가 유효하지 않습니다.")
                return None
                
            output1 = result.get("output1", {})
            output2 = result.get("output2", [])
            
            # 계좌 종합 정보
            summary = {
                "total_deposit": int(output1.get("dnca_tot_amt", "0").replace(',', '')),  # 예수금총금액
                "withdrawal_available": int(output1.get("prvs_rcdl_excc_amt", "0").replace(',', '')),  # 가수도 정산 금액
                "d1_withdraw": int(output1.get("thdt_sll_amt", "0").replace(',', '')),  # 금일 매도 금액
                "d2_withdraw": int(output1.get("nxdy_auto_rdpt_amt", "0").replace(',', '')),  # 익일 자동 환매 금액
                "total_stocks_value": int(output1.get("tot_asst_evlu_amt", "0").replace(',', '')),  # 총자산평가금액
                "total_profit_loss": int(output1.get("asst_icdc_amt", "0").replace(',', '')),  # 자산증감액
                "total_profit_rate": float(output1.get("asst_icdc_erng_rt", "0").replace(',', ''))  # 자산증감수익률
            }
            
            # 보유 종목 정보
            stocks = []
            for stock in output2:
                if not stock.get("pdno"):  # 종목코드가 없는 경우 건너뛰기
                    continue
                    
                stock_info = {
                    "code": stock.get("pdno", ""),
                    "name": stock.get("prdt_name", ""),
                    "quantity": int(stock.get("hldg_qty", "0").replace(',', '')),
                    "purchase_price": int(stock.get("pchs_avg_pric", "0").replace(',', '')),
                    "current_price": int(stock.get("prpr", "0").replace(',', '')),
                    "earning_rate": float(stock.get("evlu_pfls_rt", "0").replace(',', '')),
                    "purchase_amount": int(stock.get("pchs_amt", "0").replace(',', '')),
                    "evaluation_amount": int(stock.get("evlu_amt", "0").replace(',', '')),
                    "profit_loss": int(stock.get("evlu_pfls_amt", "0").replace(',', '')),
                    "purchase_date": stock.get("thdt_buyqty", "") or None
                }
                stocks.append(stock_info)
            
            daily_balance = {
                "summary": summary,
                "stocks": stocks,
                "stock_count": len(stocks)
            }
            
            logger.info(f"계좌 일별 잔고 조회 성공: 자산총액 {summary['total_stocks_value']:,}원, 수익률 {summary['total_profit_rate']}%")
            return daily_balance
            
        except Exception as e:
            logger.error(f"계좌 일별 잔고 조회 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] 계좌 일별 잔고 조회 실패: {e}")
            return None 