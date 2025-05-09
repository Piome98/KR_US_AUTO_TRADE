"""
한국 주식 자동매매 - API 클라이언트 모듈
한국투자증권 API 호출 기능
"""

import json
import requests
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

from korea_stock_auto.config import (
    APP_KEY, APP_SECRET, URL_BASE, 
    CANO, ACNT_PRDT_CD, USE_REALTIME_API
)
from korea_stock_auto.utils.utils import (
    send_message, hashkey, create_hmac_signature, 
    handle_http_error, rate_limit_wait, retry_on_failure
)
from korea_stock_auto.api.auth import (
    get_access_token, refresh_token_if_needed, 
    is_token_valid, verify_token_status
)

# 로깅 설정
logger = logging.getLogger(__name__)

class APIClient:
    """한국투자증권 API 클라이언트"""
    
    def __init__(self):
        """API 클라이언트 초기화"""
        self.access_token = get_access_token()
        self.last_request_time = 0
        self.min_interval = 0.2  # 초당 5회 이하로 요청 제한
        self.request_count = 0
        self.reset_time = time.time() + 60  # 1분마다 초기화
    
    def _rate_limit(self):
        """
        API 호출 속도 제한 (초당 5회, 분당 100회)
        """
        current_time = time.time()
        
        # 분당 요청 횟수 제한 (100회)
        if current_time > self.reset_time:
            self.request_count = 0
            self.reset_time = current_time + 60
        
        self.request_count += 1
        if self.request_count > 95:  # 여유있게 95회로 제한
            wait_time = self.reset_time - current_time
            logger.warning(f"분당 요청 한도 접근 중 ({self.request_count}/100). {wait_time:.1f}초 대기")
            time.sleep(wait_time)
            self.request_count = 1
            self.reset_time = time.time() + 60
        
        # 초당 요청 횟수 제한 (5회)
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _get_headers(self, tr_id: str, hashkey_val: Optional[str] = None) -> Dict[str, str]:
        """
        API 요청용 헤더 생성
        
        Args:
            tr_id (str): 거래 ID
            hashkey_val (str, optional): 해시키
            
        Returns:
            dict: 헤더 정보
        """
        # 토큰이 만료되었거나 곧 만료될 예정이면 갱신
        if not is_token_valid():
            logger.info("토큰 만료 또는 만료 예정으로 갱신 시도")
            if refresh_token_if_needed():
                self.access_token = get_access_token()
            else:
                logger.error("토큰 갱신 실패")
                send_message("토큰 갱신 실패")
        
        headers = {
            "content-type": "application/json; charset=utf-8", 
            "authorization": f"Bearer {self.access_token}",
            "appkey": APP_KEY,
            "appsecret": APP_SECRET,
            "tr_id": tr_id,
            "custtype": "P",  # 개인
        }
        
        if hashkey_val:
            headers["hashkey"] = hashkey_val
            
        return headers
    
    def _handle_response(self, res: requests.Response, error_msg: str = "API 요청 실패") -> Optional[Dict[str, Any]]:
        """
        API 응답 처리 공통 함수
        
        Args:
            res (requests.Response): API 응답 객체
            error_msg (str): 오류 시 표시할 메시지
            
        Returns:
            dict or None: 응답 데이터 또는 오류시 None
        """
        try:
            res.raise_for_status()
            
            response_data = res.json()
            
            # API 응답 오류 코드 확인 (rt_cd가 0이 아닌 경우 오류)
            if 'rt_cd' in response_data and response_data['rt_cd'] != '0':
                error_code = response_data.get('rt_cd', 'unknown')
                error_message = response_data.get('msg1', 'No error message provided')
                logger.error(f"API 오류 - 코드: {error_code}, 메시지: {error_message}")
                send_message(f"[API 오류] 코드: {error_code}, 메시지: {error_message}")
                
                # 토큰 만료 오류인 경우 토큰 재발급 시도
                if error_code in ['EGW00123', 'EGW00203', 'EGW00033', 'OPSB00006']:  # 토큰 만료 관련 에러 코드
                    logger.warning("토큰 만료 관련 오류 감지, 재발급 시도")
                    send_message("토큰이 만료되었습니다. 재발급을 시도합니다.")
                    if refresh_token_if_needed():
                        self.access_token = get_access_token()
                        logger.info("토큰 재발급 성공")
                    else:
                        logger.error("토큰 재발급 실패")
                        send_message("토큰 재발급 실패")
                
                # 일부 경고성 오류는 정상 처리
                if error_code in ['APBK0225', 'APBK0030']:  # 데이터 없음 등 경고성 메시지
                    logger.warning(f"경고성 오류 (정상 처리됨): {error_code} - {error_message}")
                    return response_data
                
                return None
            
            return response_data
            
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"{error_msg}: HTTP 오류 - {http_err}")
            send_message(f"[오류] {error_msg}: HTTP 오류 - {http_err}")
            
            # 응답 본문이 있으면 함께 로그
            if hasattr(res, 'text'):
                logger.error(f"응답 내용: {res.text}")
                send_message(f"응답 내용: {res.text}")
            
            return None
        except requests.exceptions.Timeout:
            logger.error(f"{error_msg}: 시간 초과")
            send_message(f"[오류] {error_msg}: 요청 시간 초과")
            return None
        except ValueError as json_err:  # JSON 파싱 오류
            logger.error(f"{error_msg}: JSON 파싱 오류 - {json_err}")
            send_message(f"[오류] {error_msg}: JSON 파싱 오류 - {json_err}")
            if hasattr(res, 'text'):
                logger.error(f"응답 내용: {res.text}")
                send_message(f"응답 내용: {res.text}")
            return None
        except Exception as e:
            logger.error(f"{error_msg}: {e}", exc_info=True)
            send_message(f"[오류] {error_msg}: {e}")
            return None
        
    def fetch_balance(self) -> Optional[Dict[str, Any]]:
        """
        계좌 잔고 정보 조회
        
        Returns:
            dict or None: 계좌 잔고 정보
            
        Notes:
            모의투자 지원 함수입니다.
        """
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
        
    def fetch_stock_current_price(self, code: str) -> Optional[Dict[str, Any]]:
        """
        주식 현재가 조회
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict or None: 주식 현재가 정보
        """
        path = "uapi/domestic-stock/v1/quotations/inquire-price"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": code,
        }
        
        headers = self._get_headers("FHKST01010100")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} 현재가 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} 현재가 조회 실패")
            
            if result and result.get("rt_cd") == "0":
                price = result.get("output", {}).get("stck_prpr", "0")
                logger.info(f"{code} 현재가 조회 성공: {price}원")
            
            return result
            
        except Exception as e:
            logger.error(f"{code} 현재가 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 현재가 조회 실패: {e}")
            return None
    
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
    
    def fetch_daily_price(self, code: str, date_from: str, date_to: str) -> Optional[Dict[str, Any]]:
        """
        일별 주가 조회
        
        Args:
            code (str): 종목 코드
            date_from (str): 시작일(YYYYMMDD)
            date_to (str): 종료일(YYYYMMDD)
            
        Returns:
            dict or None: 일별 주가 정보
        """
        path = "uapi/domestic-stock/v1/quotations/inquire-daily-price"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": code,
            "fid_period_div_code": "D",
            "fid_org_adj_prc": "1",
            "FID_INPUT_DATE_1": date_from,
            "FID_INPUT_DATE_2": date_to,
        }
        
        headers = self._get_headers("FHKST01010400")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} 일별 주가 조회 요청 ({date_from} ~ {date_to})")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            return self._handle_response(res, f"{code} 일별 주가 조회 실패")
            
        except Exception as e:
            logger.error(f"{code} 일별 주가 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 일별 주가 조회 실패: {e}")
            return None
    
    def get_stock_balance(self) -> Optional[Dict[str, Any]]:
        """
        보유 주식 잔고 조회 및 처리
        
        Returns:
            dict or None: 보유 주식 잔고 정보 (종목 코드를 키로 하는 딕셔너리)
        """
        try:
            balance_data = self.fetch_balance()
            if not balance_data or balance_data.get("rt_cd") != "0":
                logger.error("잔고 조회 실패 또는 오류 응답")
                send_message("잔고 조회에 실패했습니다.")
                return None
            
            output1 = balance_data.get("output1", [])
            output2 = balance_data.get("output2", [])
            
            # 예수금 정보
            dnca_tot_amt = 0  # 예수금
            prvs_rcdl_excc_amt = 0  # 가수도 정산 금액
            
            if output2:
                dnca_tot_amt = int(output2[0].get("dnca_tot_amt", "0").replace(',', ''))
                prvs_rcdl_excc_amt = int(output2[0].get("prvs_rcdl_excc_amt", "0").replace(',', ''))
            
            # 보유 종목 정보
            stock_dict = {}
            send_message("보유종목 현황:")
            
            for stock in output1:
                # 종목코드
                code = stock.get("pdno", "")
                # 종목명
                name = stock.get("prdt_name", "알 수 없음")
                # 보유수량
                qty = int(stock.get("hldg_qty", "0").replace(',', ''))
                # 매입가
                purchase_price = int(stock.get("pchs_avg_pric", "0").replace(',', ''))
                # 현재가
                current_price = int(stock.get("prpr", "0").replace(',', ''))
                # 평가금액
                eval_amt = int(stock.get("evlu_amt", "0").replace(',', ''))
                # 평가손익
                profit_loss = int(stock.get("evlu_pfls_amt", "0").replace(',', ''))
                # 수익률
                earning_rate = float(stock.get("evlu_pfls_rt", "0"))
                
                # 종목별 정보 저장
                stock_dict[code] = {
                    "name": name,
                    "qty": qty,
                    "purchase_price": purchase_price,
                    "current_price": current_price,
                    "eval_amt": eval_amt,
                    "profit_loss": profit_loss,
                    "earning_rate": earning_rate
                }
                
                # 결과 출력
                message = f"{name}({code}): {qty}주, {current_price}원" + \
                          f" (수익률: {earning_rate:.2f}%, 평가손익: {profit_loss:,}원)"
                send_message(message)
            
            # 예수금 및 총 평가금액 정보
            total_eval_amt = int(balance_data.get("output2", [{}])[0].get("tot_evlu_amt", "0").replace(',', ''))
            total_purchase_amt = int(balance_data.get("output2", [{}])[0].get("pchs_amt_smtl_amt", "0").replace(',', ''))
            total_profit_loss = int(balance_data.get("output2", [{}])[0].get("evlu_pfls_smtl_amt", "0").replace(',', ''))
            total_earning_rate = float(balance_data.get("output2", [{}])[0].get("evlu_pfls_rt", "0"))
            
            send_message(f"예수금: {dnca_tot_amt:,}원")
            send_message(f"총 평가금액: {total_eval_amt:,}원 (손익: {total_profit_loss:,}원, 수익률: {total_earning_rate:.2f}%)")
            
            # 예수금 정보도 포함하여 반환
            stock_dict["cash"] = {
                "deposit": dnca_tot_amt,
                "pending_settlement": prvs_rcdl_excc_amt,
                "total_eval": total_eval_amt,
                "total_purchase": total_purchase_amt,
                "total_profit_loss": total_profit_loss,
                "total_earning_rate": total_earning_rate
            }
            
            return stock_dict
            
        except Exception as e:
            logger.error(f"주식 잔고 처리 중 오류 발생: {e}", exc_info=True)
            send_message(f"[오류] 주식 잔고 처리 실패: {e}")
            return None
    
    def get_balance(self) -> Optional[Dict[str, float]]:
        """
        계좌 잔고 및 주문가능금액 조회
        
        Returns:
            dict or None: 계좌 잔고 정보 (총 평가금액, 예수금 등)
        """
        try:
            balance_data = self.fetch_balance()
            if not balance_data or balance_data.get("rt_cd") != "0":
                logger.error("잔고 조회 실패 또는 오류 응답")
                send_message("잔고 조회에 실패했습니다.")
                return None
            
            # output2에서 필요한 정보 추출
            output2 = balance_data.get("output2", [{}])[0]
            
            # 예수금
            dnca_tot_amt = int(output2.get("dnca_tot_amt", "0").replace(',', ''))
            # 주문가능현금
            ord_psbl_cash = int(output2.get("ord_psbl_cash", "0").replace(',', ''))
            # D+1 예상 예수금
            d1_dncl_amt = int(output2.get("d1_dncl_amt", "0").replace(',', ''))
            # D+2 예상 예수금
            d2_dncl_amt = int(output2.get("d2_dncl_amt", "0").replace(',', ''))
            # 총 평가금액
            tot_evlu_amt = int(output2.get("tot_evlu_amt", "0").replace(',', ''))
            # 총평가손익
            evlu_pfls_smtl_amt = int(output2.get("evlu_pfls_smtl_amt", "0").replace(',', ''))
            # 총수익률
            evlu_pfls_rt = float(output2.get("evlu_pfls_rt", "0"))
            # 대출금액
            loan_amt = int(output2.get("loan_amt", "0").replace(',', ''))
            
            # 결과 저장
            balance_info = {
                "예수금": dnca_tot_amt,
                "주문가능현금": ord_psbl_cash,
                "D+1예상예수금": d1_dncl_amt,
                "D+2예상예수금": d2_dncl_amt,
                "총평가금액": tot_evlu_amt,
                "총평가손익": evlu_pfls_smtl_amt,
                "총수익률": evlu_pfls_rt,
                "대출금액": loan_amt
            }
            
            # 결과 출력
            send_message(f"예수금: {dnca_tot_amt:,}원")
            send_message(f"주문가능현금: {ord_psbl_cash:,}원")
            send_message(f"D+2예상예수금: {d2_dncl_amt:,}원")
            send_message(f"총평가금액: {tot_evlu_amt:,}원 (손익: {evlu_pfls_smtl_amt:,}원, 수익률: {evlu_pfls_rt:.2f}%)")
            
            return balance_info
            
        except Exception as e:
            logger.error(f"계좌 잔고 처리 중 오류 발생: {e}", exc_info=True)
            send_message(f"[오류] 계좌 잔고 처리 실패: {e}")
            return None
    
    def get_top_traded_stocks(self, market_type: str = "0", top_n: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        거래량 상위 종목 조회
        
        Args:
            market_type (str): 시장 구분 (0:전체, 1:코스피, 2:코스닥)
            top_n (int): 조회할 종목 수 (최대 100)
            
        Returns:
            list or None: 거래량 상위 종목 목록
        """
        path = "uapi/domestic-stock/v1/quotations/volume-rank"
        url = f"{URL_BASE}/{path}"
        
        if top_n > 100:
            top_n = 100  # 최대 100개까지만 조회 가능
            
        params = {
            "FID_COND_MRKT_DIV_CODE": market_type,
            "FID_COND_SCR_DIV_CODE": "20171",
            "FID_INPUT_ISCD": "0000",
            "FID_DIV_CLS_CODE": "0",  # 0: 거래량, 1: 거래대금, 2: 거래량 급증, 3: 거래대금 급증
            "FID_BLNG_CLS_CODE": "0",
            "FID_TRGT_CLS_CODE": "111111111",
            "FID_TRGT_EXLS_CLS_CODE": "000000",
            "FID_INPUT_PRICE_1": "",
            "FID_INPUT_PRICE_2": "",
            "FID_VOL_CNT": str(top_n),
            "FID_INPUT_DATE_1": ""
        }
        
        headers = self._get_headers("FHPST01710000")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            market_name = {
                "0": "전체", 
                "1": "코스피", 
                "2": "코스닥"
            }.get(market_type, "전체")
            
            logger.info(f"{market_name} 거래량 상위 {top_n}개 종목 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"거래량 상위 종목 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            stocks = result.get("output", [])
            if not stocks:
                logger.warning("거래량 상위 종목이 없습니다.")
                return []
            
            # 결과 가공
            ranked_stocks = []
            for stock in stocks:
                stock_info = {
                    "rank": stock.get("no", "0"),
                    "code": stock.get("mksc_shrn_iscd", ""),
                    "name": stock.get("hts_kor_isnm", ""),
                    "price": int(stock.get("stck_prpr", "0").replace(',', '')),
                    "change_rate": float(stock.get("prdy_ctrt", "0")),
                    "volume": int(stock.get("acml_vol", "0").replace(',', '')),
                    "market_cap": int(stock.get("hts_avls", "0").replace(',', ''))
                }
                ranked_stocks.append(stock_info)
            
            logger.info(f"{market_name} 거래량 상위 {len(ranked_stocks)}개 종목 조회 성공")
            return ranked_stocks
            
        except Exception as e:
            logger.error(f"거래량 상위 종목 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 거래량 상위 종목 조회 실패: {e}")
            return None
    
    def get_stock_info(self, code: str) -> Optional[Dict[str, Any]]:
        """
        종목 기본 정보 조회
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict or None: 종목 기본 정보
        """
        path = "uapi/domestic-stock/v1/quotations/search-info"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code
        }
        
        headers = self._get_headers("CTPF1002R")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} 종목 기본 정보 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} 종목 정보 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", {})
            if not output:
                logger.warning(f"{code} 종목 정보가 없습니다.")
                return None
            
            # 결과 가공
            stock_info = {
                "code": code,
                "name": output.get("hts_kor_isnm", ""),
                "market": output.get("bstp_kor_isnm", ""),
                "sector": output.get("bstp_larg_div_name", ""),
                "industry": output.get("bstp_med_div_name", ""),
                "listed_shares": int(output.get("lstn_stcn", "0").replace(',', '')),
                "capital": int(output.get("cpfn", "0").replace(',', '')),
                "par_value": int(output.get("stck_fcam", "0").replace(',', '')),
                "foreign_rate": float(output.get("frgn_hldn_qty_rt", "0")),
                "per": float(output.get("per", "0")),
                "eps": float(output.get("eps", "0").replace(',', '')),
                "pbr": float(output.get("pbr", "0")),
                "bps": float(output.get("bps", "0").replace(',', ''))
            }
            
            logger.info(f"{code} 종목 기본 정보 조회 성공")
            return stock_info
            
        except Exception as e:
            logger.error(f"{code} 종목 정보 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 종목 정보 조회 실패: {e}")
            return None
    
    def get_stock_data(self, code: str, period_div_code: str = "D") -> Optional[List[Dict[str, Any]]]:
        """
        주식 시세 데이터 조회 (일/주/월 선택)
        
        Args:
            code (str): 종목 코드
            period_div_code (str): 조회 주기 (D:일, W:주, M:월)
            
        Returns:
            list or None: 시세 데이터 목록 (최신 순)
        """
        path = "uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        url = f"{URL_BASE}/{path}"
        
        period_name = {"D": "일봉", "W": "주봉", "M": "월봉"}.get(period_div_code, "일봉")
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": "",  # 빈 값으로 두면 최근 데이터부터 조회
            "FID_INPUT_DATE_2": "",
            "FID_PERIOD_DIV_CODE": period_div_code,
            "FID_ORG_ADJ_PRC": "0"
        }
        
        headers = self._get_headers("FHKST03010100")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} {period_name} 데이터 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} {period_name} 데이터 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output1 = result.get("output1", {})
            output2 = result.get("output2", [])
            
            if not output2:
                logger.warning(f"{code} {period_name} 데이터가 없습니다.")
                return []
            
            # 결과 가공
            chart_data = []
            for item in output2:
                data_item = {
                    "date": item.get("stck_bsop_date", ""),
                    "open": int(item.get("stck_oprc", "0").replace(',', '')),
                    "high": int(item.get("stck_hgpr", "0").replace(',', '')),
                    "low": int(item.get("stck_lwpr", "0").replace(',', '')),
                    "close": int(item.get("stck_clpr", "0").replace(',', '')),
                    "volume": int(item.get("acml_vol", "0").replace(',', '')),
                    "value": int(item.get("acml_tr_pbmn", "0").replace(',', ''))
                }
                chart_data.append(data_item)
            
            logger.info(f"{code} {period_name} 데이터 {len(chart_data)}건 조회 성공")
            return chart_data
            
        except Exception as e:
            logger.error(f"{code} {period_name} 데이터 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} {period_name} 데이터 조회 실패: {e}")
            return None
    
    def get_daily_data(self, code: str, days: int = 30) -> Optional[List[Dict[str, Any]]]:
        """
        일별 시세 데이터 조회
        
        Args:
            code (str): 종목 코드
            days (int): 조회할 일수
            
        Returns:
            list or None: 일별 시세 데이터 (최신 순)
        """
        data = self.get_stock_data(code, period_div_code="D")
        if data and days > 0 and days < len(data):
            return data[:days]
        return data
        
    def get_weekly_data(self, code: str, weeks: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        주별 시세 데이터 조회
        
        Args:
            code (str): 종목 코드
            weeks (int): 조회할 주 수
            
        Returns:
            list or None: 주별 시세 데이터 (최신 순)
        """
        data = self.get_stock_data(code, period_div_code="W")
        if data and weeks > 0 and weeks < len(data):
            return data[:weeks]
        return data
    
    def get_monthly_data(self, code: str, months: int = 12) -> Optional[List[Dict[str, Any]]]:
        """
        월별 시세 데이터 조회
        
        Args:
            code (str): 종목 코드
            months (int): 조회할 월 수
            
        Returns:
            list or None: 월별 시세 데이터 (최신 순)
        """
        data = self.get_stock_data(code, period_div_code="M")
        if data and months > 0 and months < len(data):
            return data[:months]
        return data
        
    def get_order_status(self, order_no: str) -> Optional[Dict[str, Any]]:
        """
        주문 체결 상태 조회
        
        Args:
            order_no (str): 주문 번호
            
        Returns:
            dict or None: 주문 체결 상태 정보
        """
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

    def get_pending_orders(self) -> Optional[List[Dict[str, Any]]]:
        """
        미체결 주문 조회
        
        Returns:
            list or None: 미체결 주문 목록
            
        Notes:
            모의투자에서는 지원되지 않습니다.
        """
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
                        "executed_qty": int(item.get("ccld_qty", "0").replace(',', '')),
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
        path = "uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        url = f"{URL_BASE}/{path}"
        
        tr_id = "TTTC8001R" if USE_REALTIME_API else "VTTC8001R"
        
        # 주문일자가 없으면 당일로 설정
        if not order_date:
            order_date = time.datetime.now().strftime("%Y%m%d")
        
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

    def cancel_all_orders(self) -> bool:
        """
        모든 미체결 주문 취소
        
        Returns:
            bool: 전체 취소 성공 여부
            
        Notes:
            모의투자에서는 지원되지 않습니다.
        """
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

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        계좌 기본 정보 조회
        
        Returns:
            dict or None: 계좌 기본 정보
            
        Notes:
            모의투자 지원 함수입니다.
        """
        path = "uapi/domestic-stock/v1/trading/inquire-account-balance"
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
            
            logger.info("계좌 기본 정보 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, "계좌 기본 정보 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
            
            # 출력2의 첫 번째 항목에 계좌 정보가 있음
            output2 = result.get("output2", [{}])[0]
            
            account_info = {
                "account_number": CANO + "-" + ACNT_PRDT_CD,
                "account_name": output2.get("acnt_name", ""),
                "account_type": output2.get("acnt_type_name", ""),
                "deposit": int(output2.get("dnca_tot_amt", "0").replace(',', '')),
                "withdrawal_available": int(output2.get("wdrs_psbl_amt", "0").replace(',', '')),
                "order_available": int(output2.get("ord_psbl_cash", "0").replace(',', '')),
                "total_balance": int(output2.get("tot_evlu_amt", "0").replace(',', '')),
                "stock_balance": int(output2.get("scts_evlu_amt", "0").replace(',', '')),
                "profit_loss": int(output2.get("evlu_pfls_smtl_amt", "0").replace(',', '')),
                "profit_rate": float(output2.get("evlu_pfls_rt", "0").replace(',', '')),
                "credit_available": int(output2.get("pldg_rmnd_amt", "0").replace(',', '')),
                "foreign_deposit": int(output2.get("ovrs_re_use_psbl_amt", "0").replace(',', '')),
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info("계좌 기본 정보 조회 성공")
            return account_info
            
        except Exception as e:
            logger.error(f"계좌 기본 정보 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 계좌 기본 정보 조회 실패: {e}")
            return None

    def get_buyable_cash(self) -> Optional[int]:
        """
        주문 가능 현금 조회
        
        Returns:
            int or None: 주문 가능 현금 금액
            
        Notes:
            모의투자 지원 함수입니다.
        """
        path = "uapi/domestic-stock/v1/trading/inquire-account-balance"
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
            
            logger.info("주문 가능 현금 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, "주문 가능 현금 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
            
            # 출력2의 첫 번째 항목에 현금 정보가 있음
            output2 = result.get("output2", [{}])[0]
            
            # 주문 가능 현금 금액 반환
            buyable_cash = int(output2.get("ord_psbl_cash", "0").replace(',', ''))
            
            logger.info(f"주문 가능 현금 조회 성공: {buyable_cash:,}원")
            return buyable_cash
            
        except Exception as e:
            logger.error(f"주문 가능 현금 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 주문 가능 현금 조회 실패: {e}")
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