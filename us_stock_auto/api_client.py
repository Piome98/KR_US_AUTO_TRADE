"""
미국 주식 자동매매 - API 클라이언트 모듈
한국투자증권 API 호출 기능
"""

import json
import requests

from us_stock_auto.config import APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD
from us_stock_auto.utils import send_message

class APIClient:
    """한국투자증권 API 클라이언트"""
    
    def __init__(self):
        """API 클라이언트 초기화"""
        self.access_token = None
        self.get_access_token()
    
    def get_access_token(self):
        """
        API 접근을 위한 토큰 발급
        """
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": APP_KEY, 
            "appsecret": APP_SECRET
        }
        path = "oauth2/tokenP"
        url = f"{URL_BASE}/{path}"
        
        try:
            res = requests.post(url, headers=headers, data=json.dumps(body))
            self.access_token = res.json()["access_token"]
            return self.access_token
        except Exception as e:
            send_message(f"[오류] 토큰 발급 실패: {e}")
            return None
    
    def hashkey(self, datas):
        """
        데이터 암호화 함수
        
        Args:
            datas (dict): 암호화할 데이터
            
        Returns:
            str: 암호화된 해시값
        """
        path = "uapi/hashkey"
        url = f"{URL_BASE}/{path}"
        headers = {
            'content-Type': 'application/json',
            'appKey': APP_KEY,
            'appSecret': APP_SECRET,
        }
        
        try:
            res = requests.post(url, headers=headers, data=json.dumps(datas))
            return res.json()["HASH"]
        except Exception as e:
            send_message(f"[오류] 해시키 생성 실패: {e}")
            return None
    
    def get_current_price(self, market="NAS", code="AAPL"):
        """
        현재가 조회 함수
        
        Args:
            market (str): 시장 코드 (NAS, NYS, AMS)
            code (str): 종목 코드
            
        Returns:
            float: 현재가
        """
        path = "uapi/overseas-price/v1/quotations/price"
        url = f"{URL_BASE}/{path}"
        headers = self._get_headers("HHDFS00000300")
        params = {
            "AUTH": "",
            "EXCD": market,
            "SYMB": code,
        }
        
        try:
            res = requests.get(url, headers=headers, params=params)
            return float(res.json()['output']['last'])
        except Exception as e:
            send_message(f"[오류] 현재가 조회 실패: {e}")
            return None
    
    def get_target_price(self, market="NAS", code="AAPL"):
        """
        변동성 돌파 전략 기반 매수 목표가 계산 함수
        
        Args:
            market (str): 시장 코드 (NAS, NYS, AMS)
            code (str): 종목 코드
            
        Returns:
            float: 목표 매수가
        """
        path = "uapi/overseas-price/v1/quotations/dailyprice"
        url = f"{URL_BASE}/{path}"
        headers = self._get_headers("HHDFS76240000")
        params = {
            "AUTH": "",
            "EXCD": market,
            "SYMB": code,
            "GUBN": "0",
            "BYMD": "",
            "MODP": "0"
        }
        
        try:
            res = requests.get(url, headers=headers, params=params)
            data = res.json()
            stck_oprc = float(data['output2'][0]['open'])  # 오늘 시가
            stck_hgpr = float(data['output2'][1]['high'])  # 전일 고가
            stck_lwpr = float(data['output2'][1]['low'])   # 전일 저가
            target_price = stck_oprc + (stck_hgpr - stck_lwpr) * 0.5
            return target_price
        except Exception as e:
            send_message(f"[오류] 목표가 계산 실패: {e}")
            return None
    
    def get_stock_balance(self):
        """
        해외주식 잔고조회
        
        Returns:
            dict: 보유종목 정보 (종목코드: 보유수량)
        """
        path = "uapi/overseas-stock/v1/trading/inquire-balance"
        url = f"{URL_BASE}/{path}"
        headers = self._get_headers("JTTT3012R")
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "OVRS_EXCG_CD": "NASD",
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": ""
        }
        
        try:
            res = requests.get(url, headers=headers, params=params)
            stock_list = res.json()['output1']
            evaluation = res.json()['output2']
            stock_dict = {}
            
            send_message("====주식 보유잔고====")
            for stock in stock_list:
                if int(stock['ovrs_cblc_qty']) > 0:
                    stock_dict[stock['ovrs_pdno']] = stock['ovrs_cblc_qty']
                    send_message(f"{stock['ovrs_item_name']}({stock['ovrs_pdno']}): {stock['ovrs_cblc_qty']}주")
            
            if evaluation:
                send_message(f"주식 평가 금액: ${evaluation['tot_evlu_pfls_amt']}")
                send_message(f"평가 손익 합계: ${evaluation['ovrs_tot_pfls']}")
            
            send_message("=================")
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
        path = "uapi/domestic-stock/v1/trading/inquire-psbl-order"
        url = f"{URL_BASE}/{path}"
        headers = self._get_headers("TTTC8908R")
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
            cash = res.json()['output']['ord_psbl_cash']
            send_message(f"주문 가능 현금 잔고: {cash}원")
            return int(cash)
        except Exception as e:
            send_message(f"[오류] 현금 잔고 조회 실패: {e}")
            return 0
    
    def buy(self, market="NASD", code="AAPL", qty="1", price="0"):
        """
        미국 주식 지정가 매수
        
        Args:
            market (str): 거래소 코드 (NASD, NYSE, AMEX)
            code (str): 종목 코드
            qty (str): 주문 수량
            price (str): 주문 가격
            
        Returns:
            bool: 매수 성공 여부
        """
        path = "uapi/overseas-stock/v1/trading/order"
        url = f"{URL_BASE}/{path}"
        data = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "OVRS_EXCG_CD": market,
            "PDNO": code,
            "ORD_DVSN": "00",
            "ORD_QTY": str(int(qty)),
            "OVRS_ORD_UNPR": f"{round(float(price), 2)}",
            "ORD_SVR_DVSN_CD": "0"
        }
        headers = self._get_headers("JTTT1002U", hashkey=self.hashkey(data))
        
        try:
            res = requests.post(url, headers=headers, data=json.dumps(data))
            if res.json()['rt_cd'] == '0':
                send_message(f"[매수 성공] {code} {qty}주 {price}달러")
                return True
            else:
                send_message(f"[매수 실패] {res.json()}")
                return False
        except Exception as e:
            send_message(f"[오류] 매수 주문 실패: {e}")
            return False
    
    def sell(self, market="NASD", code="AAPL", qty="1", price="0"):
        """
        미국 주식 지정가 매도
        
        Args:
            market (str): 거래소 코드 (NASD, NYSE, AMEX)
            code (str): 종목 코드
            qty (str): 주문 수량
            price (str): 주문 가격
            
        Returns:
            bool: 매도 성공 여부
        """
        path = "uapi/overseas-stock/v1/trading/order"
        url = f"{URL_BASE}/{path}"
        data = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "OVRS_EXCG_CD": market,
            "PDNO": code,
            "ORD_DVSN": "00",
            "ORD_QTY": str(int(qty)),
            "OVRS_ORD_UNPR": f"{round(float(price), 2)}",
            "ORD_SVR_DVSN_CD": "0"
        }
        headers = self._get_headers("JTTT1006U", hashkey=self.hashkey(data))
        
        try:
            res = requests.post(url, headers=headers, data=json.dumps(data))
            if res.json()['rt_cd'] == '0':
                send_message(f"[매도 성공] {code} {qty}주 {price}달러")
                return True
            else:
                send_message(f"[매도 실패] {res.json()}")
                return False
        except Exception as e:
            send_message(f"[오류] 매도 주문 실패: {e}")
            return False
    
    def get_exchange_rate(self):
        """
        환율 조회 함수
        
        Returns:
            float: USD/KRW 환율
        """
        path = "uapi/overseas-stock/v1/trading/inquire-present-balance"
        url = f"{URL_BASE}/{path}"
        headers = self._get_headers("CTRP6504R")
        params = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "OVRS_EXCG_CD": "NASD",
            "WCRC_FRCR_DVSN_CD": "01",
            "NATN_CD": "840",
            "TR_MKET_CD": "01",
            "INQR_DVSN_CD": "00"
        }
        
        exchange_rate = 1270.0  # 기본 환율 설정
        
        try:
            res = requests.get(url, headers=headers, params=params)
            if res.status_code == 200:
                res_data = res.json()
                if 'output2' in res_data and len(res_data['output2']) > 0:
                    exchange_rate = float(res_data['output2'][0]['frst_bltn_exrt'])
        except Exception as e:
            send_message(f"[오류] 환율 조회 실패: {e}")
        
        return exchange_rate
    
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