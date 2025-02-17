# 주식 정보 조회 목적 API 호출
# korea_stock_auto.trading.real_trade.kr_stock_api

import requests
import json
import time
import numpy as np
import queue
from korea_stock_auto.utility.kr_config import *
from korea_stock_auto.utility.kr_auth import get_access_token
from korea_stock_auto.utility.kr_utils import send_message, hashkey
from korea_stock_auto.shared.price_queue import price_queue

# 액세스 토큰 발급
ACCESS_TOKEN = get_access_token()

APP_KEY = APP_KEY_REAL if USE_REALTIME_API else APP_KEY_VTS
APP_SECRET = APP_SECRET_REAL if USE_REALTIME_API else APP_SECRET_VTS
URL_BASE = URL_BASE_REAL if USE_REALTIME_API else URL_BASE_VTS
CANO = CANO_REAL if USE_REALTIME_API else CANO_VTS


# 국내 주식 매수 가능 조회
def get_stock_balance():
    URL = f"{URL_BASE_REAL}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY_REAL,
        "appSecret": APP_SECRET_REAL,
        "tr_id": "TTTC8434R",
        "custtype": "P",
    }
    params = {
        'CANO': CANO,  # 실전/모의 계좌번호 자동 변경
        'ACNT_PRDT_CD': ACNT_PRDT_CD,
        'PDNO': '',  # 종목번호 (공란 입력 가능)
        'ORD_UNPR': '',  # 주문 단가 (공란 가능)
        'ORD_DVSN': '01',  # 주문 구분 (00: 지정가, 01: 시장가 등)
        'CMA_EVLU_AMT_ICLD_YN': 'N',  
        'OVRS_ICLD_YN': 'N',  # 해외 포함 여부 
    }

    try:
        res = requests.get(URL, headers=headers, params=params)
        res_json = res.json()
    except Exception as e:
        send_message(f"❌ 주식 보유 잔고 API JSON 파싱 오류: {str(e)}")
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
            time.sleep(0.1)

    if evaluation and isinstance(evaluation, list) and len(evaluation) > 0:
        tot_evlu_amt = evaluation[0].get('tot_evlu_amt', 'N/A')
        send_message(f"총 평가 금액: {tot_evlu_amt}원")
    send_message('=================')
    
    return stock_dict


# 국내 주식 실시간 시세
def get_current_price(code):
    """현재가 조회"""
    while True:
        try:
            # 큐에서 시세 데이터 가져오기 (전용 모듈에서 가져온 price_queue 사용)
            received_code, price = price_queue.get(timeout=10)  # 10초 대기
            if received_code == code:
                return price
        except queue.Empty:
            send_message("실시간 시세 데이터를 수신하지 못했습니다.")
            return None


# 종목 매수 주문
def buy_stock(code="005930", qty="1"):
    """주식 시장가 매수"""
    URL = f"{URL_BASE_REAL}/uapi/domestic-stock/v1/trading/order-cash"
    data = {
        "CANO": CANO_REAL,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": code,
        "ORD_DVSN": "01",
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": "0",
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY_REAL,
        "appSecret": APP_SECRET_REAL,
        "tr_id": "TTTC0802U",
        "custtype": "P",
        "hashkey": hashkey(data),
    }
    
    try:
        res = requests.post(URL, headers=headers, data=json.dumps(data))
        res_json = res.json()
    except Exception as e:
        send_message(f"[매수 실패] JSON 파싱 오류: {str(e)}")
        return False

    if res_json.get("rt_cd") == "0":
        send_message(f"[매수 성공] {res_json}")
        return True
    else:
        send_message(f"[매수 실패] {res_json}")
        return False


# 종목 매도 주문
def sell_stock(stock_code, quantity):
    """주식 매도 주문"""
    URL = f"{URL_BASE_REAL}/uapi/domestic-stock/v1/trading/order"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY_REAL,
        "appSecret": APP_SECRET_REAL,
        "tr_id": "TTTC0802U",  # 매도용 거래 ID
    }
    params = {
        "CANO": CANO_REAL,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": stock_code,
        "ORD_DVSN": "00",  # 시장가 매도
        "ORD_QTY": str(quantity),
    }
    res = requests.post(URL, headers=headers, json=params)
    if res.status_code == 200:
        try:
            return res.json()
        except Exception as e:
            send_message(f"매도 성공 후 JSON 파싱 오류: {str(e)}")
            return None
    else:
        try:
            error_json = res.json()
        except Exception:
            error_json = res.text
        raise Exception(f"매도 실패: {error_json}")


# 실시간 거래량 상위 종목 조회
def get_top_traded_stocks():
    URL = f"{URL_BASE_REAL}/uapi/domestic-stock/v1/quotations/volume-rank"
    send_message(f"📡 API 요청 URL: {URL}")

    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY_REAL,
        "appSecret": APP_SECRET_REAL,
        "tr_id": "FHPST01710000",  
    }
    params = {"fid_cond_mrkt_div_code": "J"}

    try:
        res = requests.get(URL, headers=headers, params=params)
    except Exception as e:
        send_message(f"❌ API 요청 실패: {str(e)}")
        return []

    if res.status_code != 200:
        send_message(f"❌ API 요청 실패 (HTTP {res.status_code}): {res.text}")
        return []

    try:
        res_json = res.json()
        send_message(f"✅ API 응답 확인: {res_json}")
        return res_json.get("output", [])
    except Exception as e:
        send_message(f"❌ JSONDecodeError: {str(e)} -> {res.text}")
        return []


# 시가총액 및 30일 평균 거래량 조회
def get_stock_info(code):
    URL = f"{URL_BASE_REAL}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY_REAL,
        "appSecret": APP_SECRET_REAL,
        "tr_id": "FHKST01010100",
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",         # 국내 주식 시장
        "FID_COND_SCR_DIV_CODE": "20171",        # 조건 화면 분류 코드
        "FID_INPUT_ISCD": "0000",                # 전체 종목
        "FID_DIV_CLS_CODE": "1",                 # 보통주주
        "FID_BLNG_CLS_CODE": "1",                # 거래 증가율 순 정렬
        "FID_TRGT_CLS_CODE": "111111111",        # 증거금 30~100% 포함
        "FID_TRGT_EXLS_CLS_CODE": "",            # 투자경고/주의 모두 제외
        "FID_INPUT_PRICE_1": "3000",             # 3000원 이상
        "FID_INPUT_PRICE_2": "",                 # 가격 제한 없음
        "FID_VOL_CNT": "",                       # 거래량 제한 없음
        "FID_INPUT_DATE_1": ""                   # 날짜 제한 없음
    }

    try:
        res = requests.get(URL, headers=headers, params=params)
    except Exception as e:
        send_message(f"❌ 주식 정보 요청 실패: {str(e)}")
        return []

    if res.status_code == 200:
        try:
            data = res.json()
            return data.get("output", [])
        except Exception as e:
            send_message(f"❌ JSON 파싱 오류: {str(e)}")
            return []
    else:
        send_message(f"❌ 거래량 상위 종목 조회 실패 (HTTP {res.status_code}): {res.text}")
        return []


def get_stock_data(code, period_div_code):
    """
    주어진 종목 코드와 기간 분류 코드에 따라 주식 데이터를 조회합니다.

    Parameters:
        code (str): 종목 코드 (6자리)
        period_div_code (str): 기간 분류 코드 ('D' - 일봉, 'M' - 월봉)

    Returns:
        list: 주식 데이터 리스트
    """
    url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST01010400",
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",  # 주식, ETF, ETN
        "FID_INPUT_ISCD": code,
        "FID_PERIOD_DIV_CODE": period_div_code,
        "FID_ORG_ADJ_PRC": "0",  # 수정주가 반영
    }

    try:
        response = requests.get(url, headers=headers, params=params)
    except Exception as e:
        send_message(f"HTTP 요청 오류: {str(e)}")
        return []

    if response.status_code == 200:
        try:
            data = response.json()
            if data.get('rt_cd') == '0':
                return data.get('output', [])
            else:
                send_message(f"API 응답 오류: {data.get('msg1')}")
        except Exception as e:
            send_message(f"❌ JSON 파싱 오류: {str(e)}")
    else:
        send_message(f"HTTP 요청 오류: {response.status_code}")
    return []


def get_daily_data(code):
    return get_stock_data(code, 'D')


def get_monthly_data(code):
    return get_stock_data(code, 'M')


def get_financial_info(code):
    """재무 정보 조회"""
    url = f"{URL_BASE}/uapi/domestic-stock/v1/finance/financial-ratio"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST66430300",
    }
    params = {
        "FID_DIV_CLS_CODE": "0",
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": code,
    }
    try:
        res = requests.get(url, headers=headers, params=params)
    except Exception as e:
        send_message(f"❌ 재무 정보 요청 실패: {str(e)}")
        return {}

    if res.status_code == 200:
        try:
            data = res.json().get("output", {})
            return data
        except Exception as e:
            send_message(f"❌ JSON 파싱 오류: {str(e)}")
            return {}
    else:
        send_message(f"❌ 재무 정보 조회 실패 (HTTP {res.status_code}): {res.text}")
        return {}
