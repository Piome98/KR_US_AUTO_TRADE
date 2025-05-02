"""
한국 주식 자동매매 - API 클라이언트 모듈
한국투자증권 API 호출 기능
"""

import json
import requests
import numpy as np

from korea_stock_auto.config import (
    APP_KEY, APP_SECRET, URL_BASE, 
    CANO, ACNT_PRDT_CD, USE_REALTIME_API
)
from korea_stock_auto.utils.utils import send_message, hashkey
from korea_stock_auto.api.auth import get_access_token

class APIClient:
    """한국투자증권 API 클라이언트"""
    
    def __init__(self):
        """API 클라이언트 초기화"""
        self.access_token = get_access_token()
    
    def _get_headers(self, tr_id, hashkey=None):
        """
        API 요청용 헤더 생성
        
        Args:
            tr_id (str): 거래 ID
            hashkey (str, optional): 해시키
            
        Returns:
            dict: 헤더 정보
        """
        headers = {
            "Content-Type": "application/json", 
            "authorization": f"Bearer {self.access_token}",
            "appKey": APP_KEY,
            "appSecret": APP_SECRET,
            "tr_id": tr_id,
            "custtype": "P",
        }
        
        if hashkey:
            headers["hashkey"] = hashkey
            
        return headers
    
    def _handle_response(self, res, error_msg="API 요청 실패"):
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
            return res.json()
        except Exception as e:
            send_message(f"[오류] {error_msg}: {e}")
            return None
    
    def get_stock_balance(self):
        """
        국내 주식 잔고 조회 함수
        
        Returns:
            dict: 종목코드를 키로, 보유 수량을 값으로 하는 딕셔너리
        """
        path = "domestic-stock/v1/trading/inquire-balance"
        url = f"{URL_BASE}/{path}"
        headers = self._get_headers("TTTC8434R" if USE_REALTIME_API else "VTTC8434R")
        params = {
            'CANO': CANO,
            'ACNT_PRDT_CD': ACNT_PRDT_CD,
            'PDNO': '',
            'ORD_UNPR': '',
            'ORD_DVSN': '01',
            'CMA_EVLU_AMT_ICLD_YN': 'N',
            'OVRS_ICLD_YN': 'N',
        }
        
        try:
            res = requests.get(url, headers=headers, params=params)
            res_json = self._handle_response(res, "잔고 조회 실패")
            if not res_json:
                return {}
            
            stock_list = res_json.get('output1', [])
            evaluation = res_json.get('output2', [])
            stock_dict = {}
            
            send_message('=====주식 보유 잔고=====')
            for stock in stock_list:
                try:
                    hldg_qty = int(stock.get('hldg_qty', 0))
                except Exception:
                    hldg_qty = 0
                    
                if hldg_qty > 0:
                    pdno = stock.get('pdno', 'N/A')
                    prdt_name = stock.get('prdt_name', 'N/A')
                    stock_dict[pdno] = hldg_qty
                    send_message(f"{prdt_name}({pdno}): {hldg_qty}주")
            
            if evaluation and isinstance(evaluation, list) and len(evaluation) > 0:
                tot_evlu_amt = evaluation[0].get('tot_evlu_amt', 'N/A')
                send_message(f"총 평가 금액: {tot_evlu_amt}원")
            
            send_message('=================')
            return stock_dict
        except Exception as e:
            send_message(f"[오류] 잔고 조회 실패: {e}")
            return {}
    
    def get_balance(self):
        """
        현금 잔고조회
        
        Returns:
            int: 주문 가능 현금 잔고
        """
        path = "domestic-stock/v1/trading/inquire-psbl-order"
        url = f"{URL_BASE}/{path}"
        headers = self._get_headers("TTTC8908R" if USE_REALTIME_API else "VTTC8908R")
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "PDNO": "005930",
            "ORD_UNPR": "65500",
            "ORD_DVSN": "01",
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "Y"
        }
        
        try:
            res = requests.get(url, headers=headers, params=params)
            res_json = self._handle_response(res, "현금 잔고 조회 실패")
            if not res_json:
                return 0
                
            cash = res_json['output']['ord_psbl_cash']
            send_message(f"주문 가능 현금 잔고: {cash}원")
            return int(cash)
        except Exception as e:
            send_message(f"[오류] 현금 잔고 조회 실패: {e}")
            return 0
    
    def buy_stock(self, code, qty):
        """
        주식 시장가 매수 함수
        
        Args:
            code (str): 종목 코드
            qty (int): 주문 수량
            
        Returns:
            bool: 매수 성공 여부
        """
        path = "domestic-stock/v1/trading/order-cash"
        url = f"{URL_BASE}/{path}"
        data = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "PDNO": code,
            "ORD_DVSN": "01",  # 시장가 주문
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": "0",
        }
        headers = self._get_headers(
            "TTTC0802U" if USE_REALTIME_API else "VTTC0802U", 
            hashkey=hashkey(data)
        )
        
        try:
            res = requests.post(url, headers=headers, data=json.dumps(data))
            res_json = self._handle_response(res, "매수 주문 실패")
            if not res_json:
                return False
            
            if res_json.get("rt_cd") == "0":
                send_message(f"[매수 성공] {code} {qty}주")
                return True
            else:
                send_message(f"[매수 실패] {res_json}")
                return False
        except Exception as e:
            send_message(f"[오류] 매수 주문 실패: {e}")
            return False
    
    def sell_stock(self, code, qty):
        """
        주식 시장가 매도 함수
        
        Args:
            code (str): 종목 코드
            qty (int): 주문 수량
            
        Returns:
            bool: 매도 성공 여부
        """
        path = "domestic-stock/v1/trading/order-cash"
        url = f"{URL_BASE}/{path}"
        data = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "PDNO": code,
            "ORD_DVSN": "01",  # 시장가 매도
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": "0",
        }
        headers = self._get_headers(
            "TTTC0801U" if USE_REALTIME_API else "VTTC0801U", 
            hashkey=hashkey(data)
        )
        
        try:
            res = requests.post(url, headers=headers, data=json.dumps(data))
            res_json = self._handle_response(res, "매도 주문 실패")
            if not res_json:
                return False
                
            if res_json.get("rt_cd") == "0":
                send_message(f"[매도 성공] {code} {qty}주")
                return True
            else:
                send_message(f"[매도 실패] {res_json}")
                return False
        except Exception as e:
            send_message(f"[오류] 매도 주문 실패: {e}")
            return False
    
    def get_top_traded_stocks(self):
        """
        거래량 상위 종목 조회
        
        Returns:
            list: 거래량 상위 종목 정보 리스트
        """
        path = "uapi/domestic-stock/v1/market/high-volume"
        url = f"{URL_BASE}/{path}"
        headers = self._get_headers("FHPST01710000")
        
        try:
            res = requests.get(url, headers=headers)
            res_json = self._handle_response(res, "상위 종목 조회 실패")
            if not res_json:
                return []
                
            return res_json.get("output", [])
        except Exception as e:
            send_message(f"[오류] 상위 종목 조회 실패: {e}")
            return []
    
    def get_stock_info(self, code):
        """
        종목 기본 정보 조회
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict: 종목 정보
        """
        path = "uapi/domestic-stock/v1/quotations/inquire-price"
        url = f"{URL_BASE}/{path}"
        headers = self._get_headers("FHKST01010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code
        }
        
        try:
            res = requests.get(url, headers=headers, params=params)
            res_json = self._handle_response(res, f"{code} 정보 조회 실패")
            if not res_json:
                return {}
                
            return res_json.get("output", {})
        except Exception as e:
            send_message(f"[오류] {code} 정보 조회 실패: {e}")
            return {}
    
    def get_stock_data(self, code, period_div_code):
        """
        종목 차트 데이터 조회
        
        Args:
            code (str): 종목 코드
            period_div_code (str): 기간 구분 코드 (D:일봉, W:주봉, M:월봉)
            
        Returns:
            list: 차트 데이터 리스트
        """
        path = "uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        url = f"{URL_BASE}/{path}"
        headers = self._get_headers("FHKST03010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_PERIOD_DIV_CODE": period_div_code,
            "FID_ORG_ADJ_PRC": "0"
        }
        
        try:
            res = requests.get(url, headers=headers, params=params)
            res_json = self._handle_response(res, f"{code} 차트 데이터 조회 실패")
            if not res_json:
                return []
                
            return res_json.get("output2", [])
        except Exception as e:
            send_message(f"[오류] {code} 차트 데이터 조회 실패: {e}")
            return []
    
    def get_daily_data(self, code):
        """
        일봉 데이터 조회
        
        Args:
            code (str): 종목 코드
            
        Returns:
            list: 일봉 데이터 리스트
        """
        return self.get_stock_data(code, "D")
    
    def get_monthly_data(self, code):
        """
        월봉 데이터 조회
        
        Args:
            code (str): 종목 코드
            
        Returns:
            list: 월봉 데이터 리스트
        """
        return self.get_stock_data(code, "M") 