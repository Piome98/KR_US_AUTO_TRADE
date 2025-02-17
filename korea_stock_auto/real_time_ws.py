from websocket import WebSocketApp  # websocket-client 패키지 사용
import json
import threading
import requests
import time

from korea_stock_auto.utility.kr_auth import get_access_token
from korea_stock_auto.utility.kr_config import *
from korea_stock_auto.utility.kr_utils import send_message
from korea_stock_auto.shared.price_queue import price_queue

USE_REALTIME_API = True

# WS_URL 수정: 실전 환경은 ops 도메인 사용
if USE_REALTIME_API:
    WS_URL = "wss://ops.koreainvestment.com:21000/tryitout/H0STCNT0"
else:
    WS_URL = "wss://ops.koreainvestment.com:31000/tryitout/H0STCNT0"

def request_ws_connection_key():
    if USE_REALTIME_API:
        approval_url = "https://openapi.koreainvestment.com:9443/oauth2/Approval"
        app_key = APP_KEY_REAL
        app_secret = APP_SECRET_REAL
    else:
        approval_url = "https://openapivts.koreainvestment.com:29443/oauth2/Approval"
        app_key = APP_KEY_VTS
        app_secret = APP_SECRET_VTS

    headers = {"Content-Type": "application/json"}
    body = {"grant_type": "client_credentials", "appkey": app_key, "secretkey": app_secret}
    send_message(f"WebSocket 접속키 요청 본문: {body}")
    res = requests.post(approval_url, headers=headers, json=body)
    if res.status_code == 200:
        ws_conn_key = res.json().get("approval_key")
        send_message(f"✅ WebSocket 접속키 발급 성공: {ws_conn_key}")
        return ws_conn_key
    else:
        send_message(f"❌ WebSocket 접속키 발급 실패 (HTTP {res.status_code}): {res.text}")
        return None

ACCESS_TOKEN = get_access_token()
WS_CONN_KEY = request_ws_connection_key()

# 전역 변수: 관심종목 및 관련 변수 (symbol_list는 별도의 전역 변수 모듈로 관리하거나 kr_main.py에서 설정)
target_buy_price = {}
target_sell_price = {}
holding_stock = {}

def on_message(ws, message):
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        send_message("⚠️ JSON 파싱 오류 발생")
        return
    if "body" in data and "output" in data["body"]:
        msg_type = data["body"].get("msg1", "")
        if msg_type == "SUBSCRIBE SUCCESS":
            send_message("📡 웹소켓 구독 완료")
            return
        output = data["body"]["output"]
        for item in output:
            code = item.get("MKSC_SHRN_ISCD")
            try:
                price = int(item.get("STCK_PRPR", 0))
            except Exception:
                price = None
            send_message(f"📊 [{code}] 현재가: {price}원")
            if code and price is not None:
                price_queue.put((code, price))
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
    # 구독 메시지는 API 문서에 따라 아래와 같이 구성합니다.
    if not WS_CONN_KEY:
        send_message("⚠️ WebSocket 접속키 없음. 웹소켓 구독 요청 중단.")
        return
    # 전역 symbol_list는 kr_main.py 또는 별도의 글로벌 변수 모듈에서 관리됨.
    # 구독 메시지: header에 approval_key, custtype, tr_type, content-type (문서에서는 "utf-8"으로 명시)
    from korea_stock_auto.shared.global_vars import symbol_list  # 전역 변수 모듈 사용
    subscribe_data = {
        "header": {
            "approval_key": WS_CONN_KEY,
            "custtype": "P",
            "tr_type": "1",
            "content-type": "utf-8"
        },
        "body": {
            "input": [{"tr_id": "H0STCNT0", "tr_key": sym} for sym in symbol_list]
        }
    }
    ws.send(json.dumps(subscribe_data))
    send_message("📡 웹소켓 구독 요청 완료")

def start_websocket():
    from korea_stock_auto.shared.global_vars import symbol_list
    while True:
        if not symbol_list:
            send_message("⚠️ 관심 종목 없음! 웹소켓 구독 요청 진행 중단.")
            return
        if not WS_CONN_KEY:
            send_message("⚠️ WebSocket 접속키 없음. 웹소켓 실행 중단.")
            return
        try:
            ws = WebSocketApp(
                WS_URL,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.on_open = on_open
            # 연결 유지: ping_interval 및 ping_timeout 추가 (예: 30초마다 ping)
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            send_message(f"⚠️ WebSocket 실행 오류: {e}")
        send_message("웹소켓 연결이 종료되었습니다. 10초 후 재연결 시도합니다.")
        time.sleep(10)

def run_websocket():
    thread = threading.Thread(target=start_websocket)
    thread.daemon = True
    thread.start()

def run_realtime_ws():
    global target_buy_price, target_sell_price, holding_stock, WS_CONN_KEY
    # 관심종목은 kr_main.py에서 설정한 전역 변수로 관리.
    from korea_stock_auto.shared.global_vars import symbol_list
    target_buy_price = {sym: None for sym in symbol_list}
    target_sell_price = {sym: None for sym in symbol_list}
    holding_stock = {sym: 0 for sym in symbol_list}
    WS_CONN_KEY = request_ws_connection_key()
    run_websocket()
