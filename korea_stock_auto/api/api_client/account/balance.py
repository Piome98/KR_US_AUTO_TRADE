"""
한국 주식 자동매매 - 계좌 잔고 모듈
계좌 잔고 및 주식 보유 현황 조회 기능 제공
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

class AccountBalanceMixin:
    """
    계좌 잔고 관련 기능 Mixin
    
    계좌 잔고 조회, 주식 보유 현황 조회 등의 기능을 제공합니다.
    """
    
    def fetch_balance(self) -> Optional[Dict[str, Any]]:
        """
        계좌 잔고 정보 조회
        
        한국투자증권 API를 통해 계좌 잔고 정보를 조회합니다.
        
        Returns:
            dict or None: 계좌 잔고 정보 또는 조회 실패 시 None
            
        Notes:
            모의투자와 실전투자 모두 지원하는 함수입니다.
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{URL_BASE}/{path}"
        
        # 실전/모의투자 구분
        tr_id = "TTTC8434R" if USE_REALTIME_API else "VTTC8434R"
        
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "AFHR_FLPR_YN": "N",  # 시간외단일가여부
            "OFL_YN": "N",  # 오프라인여부
            "INQR_DVSN": "01",  # 조회구분: 01-요약
            "UNPR_DVSN": "01",  # 단가구분: 01-평균단가
            "FUND_STTL_ICLD_YN": "N",  # 펀드결제분포함여부
            "FNCG_AMT_AUTO_RDPT_YN": "N",  # 융자금액자동상환여부
            "PRCS_DVSN": "01",  # 처리구분: 01-조회
            "CTX_AREA_FK100": "",  # 연속조회검색조건
            "CTX_AREA_NK100": ""   # 연속조회키
        }
        
        try:
            headers = self._get_headers(tr_id)
            
            # 로그 레벨을 DEBUG로 변경하여 INFO 로그에는 출력되지 않도록 함
            logger.debug("계좌 잔고 조회 요청")
            result = self._request_get(url, headers, params, "계좌 잔고 조회 실패")
            
            if result:
                logger.debug("계좌 잔고 조회 성공")
            
            return result
            
        except Exception as e:
            logger.error(f"계좌 잔고 조회 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] 계좌 잔고 조회 실패: {e}")
            return None
    
    def get_stock_balance(self) -> Optional[Dict[str, Any]]:
        """
        주식 보유 종목 조회
        
        계좌에 보유 중인 주식 종목 목록과 상세 정보를 조회합니다.
        
        Returns:
            dict or None: {
                'account': {계좌 요약 정보},
                'stocks': [보유 종목 목록]
            } 또는 조회 실패 시 None
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        balance_data = self.fetch_balance()
        
        if not balance_data:
            logger.error("계좌 잔고 조회 실패")
            return None
        
        try:
            # API 응답 구조 파악 및 처리
            # output1, output2가 리스트 또는 딕셔너리인지 확인하고 처리
            output1 = balance_data.get("output1")
            output2 = balance_data.get("output2")
            
            # 응답 구조 로깅
            logger.info(f"output1 타입: {type(output1)}, output2 타입: {type(output2)}")
            
            # 계좌 정보 처리 (output1)
            account_info = {}
            if isinstance(output1, list) and len(output1) > 0:
                account_info = output1[0]
            elif isinstance(output1, dict):
                account_info = output1
            # output1이 비어있거나 다른 형태인 경우 output2의 첫 번째 항목 사용
            elif isinstance(output2, list) and len(output2) > 0:
                account_info = output2[0]
            elif isinstance(output2, dict):
                # output2가 리스트가 아니라 딕셔너리인 경우, 딕셔너리 자체를 사용
                account_info = output2
            else:
                logger.error("API 응답 형식이 예상과 다릅니다.")
                logger.error(f"balance_data: {balance_data}")
                return None
            
            # 계좌 정보가 딕셔너리인지 확인
            if not isinstance(account_info, dict):
                logger.error(f"계좌 정보가 딕셔너리가 아닙니다: {type(account_info)}")
                return None
            
            # 총액, 평가금액, 수익률 등 정보
            total_data = {
                "dnca_tot_amt": float(account_info.get("dnca_tot_amt", "0").replace(',', '')),  # 예수금총금액
                "scts_evlu_amt": float(account_info.get("scts_evlu_amt", "0").replace(',', '')),  # 유가증권평가금액
                "tot_evlu_amt": float(account_info.get("tot_evlu_amt", "0").replace(',', '')),  # 총평가금액
                "nass_amt": float(account_info.get("nass_amt", "0").replace(',', '')),  # 순자산금액
                "pchs_amt_smtl_amt": float(account_info.get("pchs_amt_smtl_amt", "0").replace(',', '')),  # 매입금액합계금액
                "evlu_pfls_smtl_amt": float(account_info.get("evlu_pfls_smtl_amt", "0").replace(',', '')),  # 평가손익합계금액
                "evlu_pfls_rt": float(account_info.get("evlu_pfls_rt", "0").replace(',', '')),  # 평가손익률
                "total_profit_loss_rate": float(account_info.get("evlu_pfls_rt", "0").replace(',', '')),  # 총수익률
                "cash": float(account_info.get("dnca_tot_amt", "0").replace(',', '')),  # 예수금총액
                "stocks_value": float(account_info.get("scts_evlu_amt", "0").replace(',', '')),  # 주식평가금액
                "total_assets": float(account_info.get("tot_evlu_amt", "0").replace(',', '')),  # 총자산
            }
            
            # 보유종목 목록 처리 (output2)
            stock_list = []
            
            # output2가 리스트인 경우 (일반적인 경우)
            if isinstance(output2, list):
                stock_items = output2[1:] if len(output2) > 1 else []
            # output2가 딕셔너리인 경우 (실전투자 환경에서 응답 형식이 다를 수 있음)
            elif isinstance(output2, dict):
                # output2가 단일 종목 정보를 포함하는 딕셔너리인 경우
                if output2.get("pdno"):  # 종목코드가 있는 경우
                    stock_items = [output2]
                else:
                    stock_items = []
            else:
                stock_items = []
                logger.warning(f"보유종목 정보 형식이 예상과 다릅니다: {type(output2)}")
            
            for stock in stock_items:
                if not isinstance(stock, dict) or not stock.get("pdno"):  # 종목코드가 없는 경우 건너뛰기
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
            
            # 로깅은 필요한 경우에만 수행 (Trader에서는 초기화 시에만)            
            return result
            
        except Exception as e:
            logger.error(f"주식 보유 종목 처리 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] 주식 보유 종목 처리 실패: {e}")
            return None
    
    def get_balance(self) -> Optional[Dict[str, float]]:
        """
        주식 계좌 잔고 요약 조회
        
        계좌의 현금, 주식평가금액, 총평가금액 등 요약 정보를 조회합니다.
        
        Returns:
            dict or None: {
                'cash': 현금(원),
                'stocks_value': 주식평가금액(원),
                'total_assets': 총평가금액(원),
                'total_profit_loss': 총평가손익(원),
                'total_profit_loss_rate': 총수익률(%)
            } 또는 조회 실패 시 None
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        balance_data = self.fetch_balance()
        
        if not balance_data:
            logger.error("계좌 잔고 조회 실패")
            return None
        
        try:
            # API 응답 구조 파악 및 처리
            output1 = balance_data.get("output1")
            output2 = balance_data.get("output2")
            
            # 계좌 정보 처리
            account_info = {}
            if isinstance(output1, list) and len(output1) > 0:
                account_info = output1[0]
            elif isinstance(output1, dict):
                account_info = output1
            # output1이 비어있거나 다른 형태인 경우 output2의 첫 번째 항목 사용
            elif isinstance(output2, list) and len(output2) > 0:
                account_info = output2[0]
            elif isinstance(output2, dict):
                account_info = output2
            else:
                logger.error("API 응답 형식이 예상과 다릅니다.")
                logger.error(f"balance_data: {balance_data}")
                return None
                
            # 계좌 정보가 딕셔너리인지 확인
            if not isinstance(account_info, dict):
                logger.error(f"계좌 정보가 딕셔너리가 아닙니다: {type(account_info)}")
                return None
            
            result = {
                "cash": float(account_info.get("dnca_tot_amt", "0").replace(',', '')),  # 예수금총액
                "stocks_value": float(account_info.get("scts_evlu_amt", "0").replace(',', '')),  # 주식평가금액
                "total_assets": float(account_info.get("tot_evlu_amt", "0").replace(',', '')),  # 총자산
                "total_profit_loss": float(account_info.get("evlu_pfls_smtl_amt", "0").replace(',', '')),  # 총평가손익
                "total_profit_loss_rate": float(account_info.get("evlu_pfls_rt", "0").replace(',', '')),  # 총수익률
            }
            
            # 로깅은 필요한 경우에만 수행 (Trader에서는 초기화 시에만)
            return result
            
        except Exception as e:
            logger.error(f"계좌 잔고 정보 처리 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] 계좌 잔고 정보 처리 실패: {e}")
            return None 