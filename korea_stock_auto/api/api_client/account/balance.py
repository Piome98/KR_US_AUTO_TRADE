"""
한국 주식 자동매매 - 계좌 잔고 모듈
"""

import requests
import logging
from typing import Dict, List, Optional, Any, Union

from korea_stock_auto.config import (
    URL_BASE, CANO, ACNT_PRDT_CD, USE_REALTIME_API
)
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

logger = logging.getLogger("stock_auto")

class AccountBalanceMixin:
    """계좌 잔고 관련 기능 Mixin"""
    
    def fetch_balance(self) -> Optional[Dict[str, Any]]:
        """
        계좌 잔고 정보 조회
        
        Returns:
            dict or None: 계좌 잔고 정보
            
        Notes:
            모의투자 지원 함수입니다.
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{URL_BASE}/{path}"
        
        tr_id = "TTTC8434R" if USE_REALTIME_API else "VTTC8434R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "N",
            "INQR_DVSN": "01",
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
            
            logger.info("계좌 잔고 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, "계좌 잔고 조회 실패")
            
            if result:
                logger.info("계좌 잔고 조회 성공")
            
            return result
            
        except Exception as e:
            logger.error(f"계좌 잔고 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 계좌 잔고 조회 실패: {e}")
            return None
    
    def get_stock_balance(self) -> Optional[Dict[str, Any]]:
        """
        주식 보유 종목 조회
        
        Returns:
            dict or None: 주식 보유 정보
        """
        self: KoreaInvestmentApiClient  # type hint
        
        balance_data = self.fetch_balance()
        
        if not balance_data:
            logger.error("계좌 잔고 조회 실패")
            return None
        
        try:
            output1 = balance_data.get("output1", {})
            output2 = balance_data.get("output2", [])
            
            # 총액, 평가금액, 수익률 등 정보
            total_data = {
                "dnca_tot_amt": float(output1.get("dnca_tot_amt", "0").replace(',', '')),  # 예수금총금액
                "scts_evlu_amt": float(output1.get("scts_evlu_amt", "0").replace(',', '')),  # 유가증권평가금액
                "tot_evlu_amt": float(output1.get("tot_evlu_amt", "0").replace(',', '')),  # 총평가금액
                "nass_amt": float(output1.get("nass_amt", "0").replace(',', '')),  # 순자산금액
                "pchs_amt_smtl_amt": float(output1.get("pchs_amt_smtl_amt", "0").replace(',', '')),  # 매입금액합계금액
                "evlu_pfls_smtl_amt": float(output1.get("evlu_pfls_smtl_amt", "0").replace(',', '')),  # 평가손익합계금액
                "evlu_pfls_rt": float(output1.get("evlu_pfls_rt", "0").replace(',', '')),  # 평가손익률
                "total_profit_loss_rate": float(output1.get("evlu_pfls_rt", "0").replace(',', '')),  # 총수익률
                "cash": float(output1.get("dnca_tot_amt", "0").replace(',', '')),  # 예수금총액
                "stocks_value": float(output1.get("scts_evlu_amt", "0").replace(',', '')),  # 주식평가금액
                "total_assets": float(output1.get("tot_evlu_amt", "0").replace(',', '')),  # 총자산
            }
            
            # 보유종목 목록
            stock_list = []
            for stock in output2:
                if stock.get("pdno") == "":
                    continue
                    
                stock_data = {
                    "prdt_name": stock.get("prdt_name", ""),  # 종목명
                    "code": stock.get("pdno", ""),  # 종목코드
                    "units": int(stock.get("hldg_qty", "0").replace(',', '')),  # 보유수량
                    "avg_buy_price": int(stock.get("pchs_avg_pric", "0").replace(',', '')),  # 매입평균가격
                    "current_price": int(stock.get("prpr", "0").replace(',', '')),  # 현재가
                    "value": int(stock.get("evlu_amt", "0").replace(',', '')),  # 평가금액
                    "earning_rate": float(stock.get("evlu_pfls_rt", "0").replace(',', '')),  # 평가손익률
                    "purchase_amount": int(stock.get("pchs_amt", "0").replace(',', '')),  # 매입금액
                    "profit_loss": int(stock.get("evlu_pfls_amt", "0").replace(',', '')),  # 평가손익금액
                    "pnl_ratio": float(stock.get("evlu_pfls_rt", "0").replace(',', '')),  # 평가손익률
                    "sellable_units": int(stock.get("sll_able_qty", "0").replace(',', '')),  # 매도가능수량
                }
                
                stock_list.append(stock_data)
            
            result = {
                "account": total_data,
                "stocks": stock_list
            }
            
            logger.info(f"주식 보유 종목 조회 성공: {len(stock_list)}종목, 총평가액: {total_data['total_assets']:,.0f}원")
            return result
            
        except Exception as e:
            logger.error(f"주식 보유 종목 처리 실패: {e}", exc_info=True)
            send_message(f"[오류] 주식 보유 종목 처리 실패: {e}")
            return None
    
    def get_balance(self) -> Optional[Dict[str, float]]:
        """
        주식 계좌 잔고 조회 (현금, 주식평가금액, 총평가금액)
        
        Returns:
            dict or None: {
                'cash': 현금(원),
                'stocks_value': 주식평가금액(원),
                'total_assets': 총평가금액(원),
                'total_profit_loss': 총평가손익(원),
                'total_profit_loss_rate': 총수익률(%)
            }
        """
        self: KoreaInvestmentApiClient  # type hint
        
        balance_data = self.fetch_balance()
        
        if not balance_data:
            logger.error("계좌 잔고 조회 실패")
            return None
        
        try:
            output1 = balance_data.get("output1", {})
            
            result = {
                "cash": float(output1.get("dnca_tot_amt", "0").replace(',', '')),  # 예수금총액
                "stocks_value": float(output1.get("scts_evlu_amt", "0").replace(',', '')),  # 주식평가금액
                "total_assets": float(output1.get("tot_evlu_amt", "0").replace(',', '')),  # 총자산
                "total_profit_loss": float(output1.get("evlu_pfls_smtl_amt", "0").replace(',', '')),  # 총평가손익
                "total_profit_loss_rate": float(output1.get("evlu_pfls_rt", "0").replace(',', '')),  # 총수익률
            }
            
            logger.info(f"계좌 잔고 조회 성공: 현금={result['cash']:,.0f}원, "
                        f"주식={result['stocks_value']:,.0f}원, "
                        f"총평가={result['total_assets']:,.0f}원, "
                        f"수익률={result['total_profit_loss_rate']:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"계좌 잔고 정보 처리 실패: {e}", exc_info=True)
            send_message(f"[오류] 계좌 잔고 정보 처리 실패: {e}")
            return None 