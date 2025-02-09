# 실시간 시세 반영 -> 웹소켓 활용:

import websocket
import json
import threading
import requests

from korea_stock_auto.utility.kr_auth import get_access_token
from korea_stock_auto.utility.kr_config import *
from korea_stock_auto.utility.kr_utils import send_message
from korea_stock_auto.shared.price_queue import price_queue

# ✅ 실전투자 여부 설정
USE_REALTIME_API = True  # 실전투자는 True, 모의투자는 False

# ✅ 웹소켓 접속 URL 설정 (국내 주식용 주소 예시 – 실제 API 문서를 참고하여 수정하세요)
if USE_REALTIME_API:
    WS_URL = "https://openapi.koreainvestment.com:9443/oauth2/Approval"
else:
    WS_URL = "https://openapivts.koreainvestment.com:29443/oauth2/Approval"

# ✅ 웹소켓 접속키 요청 함수 (수정됨)
def request_ws_connection_key():
    """웹소켓 접속키 발급"""

    approval_url = WS_URL
    if USE_REALTIME_API:
        app_key = APP_KEY_REAL
        app_secret = APP_SECRET_REAL
    else:
        app_key = APP_KEY_VTS
        app_secret = APP_SECRET_VTS

    headers = {
        "Content-Type": "application/json"
    }
    # 요청 본문에 grant_type, appkey, secretkey를 정확하게 포함합니다.
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "secretkey": app_secret
    }
    send_message(f"WebSocket 접속키 요청 본문: {body}")  # 요청 본문 로그 출력 (디버깅 용)
    res = requests.post(approval_url, headers=headers, json=body)
    if res.status_code == 200:
        ws_conn_key = res.json().get("approval_key")
        send_message(f"✅ WebSocket 접속키 발급 성공: {ws_conn_key}")
        return ws_conn_key
    else:
        send_message(f"❌ WebSocket 접속키 발급 실패 (HTTP {res.status_code}): {res.text}")
        return None

# ✅ 웹소켓 접속키 발급
ACCESS_TOKEN = get_access_token()
WS_CONN_KEY = request_ws_connection_key()

# 전역 변수: 관심 종목 및 관련 변수
symbol_list = []
target_buy_price = {}
target_sell_price = {}
holding_stock = {}

# ✅ 웹소켓 이벤트 핸들러
def on_message(ws, message):
    """실시간 시세 데이터 수신"""
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        send_message("⚠️ JSON 파싱 오류 발생")
        return

    if "body" in data and "output" in data["body"]:
        msg_type = data["body"]["msg1"]

        if msg_type == "SUBSCRIBE SUCCESS":
            send_message("📡 웹소켓 구독 완료")
            return

        output = data["body"]["output"]
        for item in output:
            code = item["MKSC_SHRN_ISCD"]  # 종목코드
            price = int(item["STCK_PRPR"])  # 현재가
            volume = int(item["CNTG_VOL"])  # 체결 거래량

            send_message(f"📊 [{code}] 현재가: {price}원 | 거래량: {volume}")

            # 로컬 임포트: buy_stock, sell_stock (순환 참조 방지)
            from korea_stock_auto.trading.real_trade.kr_stock_api import buy_stock, sell_stock

            if target_buy_price.get(code) and price <= target_buy_price[code]:
                send_message(f"💰 {code} 매수 시도: {price}원")
                if buy_stock(code, 1):
                    holding_stock[code] = holding_stock.get(code, 0) + 1
                    target_buy_price[code] = None

            if target_sell_price.get(code) and price >= target_sell_price[code]:
                if holding_stock.get(code, 0) > 0:
                    send_message(f"📈 {code} 매도 시도: {price}원")
                    sell_stock(code, holding_stock[code])
                    holding_stock[code] = 0
                    target_sell_price[code] = None

def on_error(ws, error):
    send_message(f"⚠️ 웹소켓 오류: {error}")

def on_close(ws, close_status_code, close_msg):
    send_message("🔌 웹소켓 연결 종료")

def on_open(ws):
    """웹소켓 연결 후 종목 구독 요청"""
    if not WS_CONN_KEY:
        send_message("⚠️ WebSocket 접속키 없음. 웹소켓 구독 요청을 중단합니다.")
        return

    subscribe_data = {
        "header": {
            "approval_key": WS_CONN_KEY,
            "appkey": APP_KEY_REAL if USE_REALTIME_API else APP_KEY_VTS,
            "appsecret": APP_SECRET_REAL if USE_REALTIME_API else APP_SECRET_VTS,
            "tr_type": "1",  # 1: 실시간 시세
            "custtype": "P"
        },
        "body": {
            "input": [{"tr_id": "H0STCNT0", "tr_key": sym} for sym in symbol_list]
        }
    }
    ws.send(json.dumps(subscribe_data))
    send_message("📡 웹소켓 구독 요청 완료")

def start_websocket():
    if not symbol_list:
        send_message("⚠️ 관심 종목 없음! 웹소켓 구독 요청을 진행하지 않습니다.")
        return
    
    if not WS_CONN_KEY:
        send_message("⚠️ WebSocket 접속키가 없습니다. 웹소켓 실행을 중단합니다.")
        return
    
    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    ws.run_forever()

def run_realtime_ws():
    global symbol_list, target_buy_price, target_sell_price, holding_stock, WS_CONN_KEY
    from korea_stock_auto.trading.real_trade.kr_trade_logic import select_interest_stocks

    symbol_list = select_interest_stocks() or []
    target_buy_price = {sym: None for sym in symbol_list}
    target_sell_price = {sym: None for sym in symbol_list}
    holding_stock = {sym: 0 for sym in symbol_list}

    WS_CONN_KEY = request_ws_connection_key()

    thread = threading.Thread(target=start_websocket)
    thread.daemon = True
    thread.start()
