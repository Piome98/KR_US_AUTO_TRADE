# API 인증 목적 파일
# korea_stock_auto.utility.real_utils.kr_auth

import requests
import json
import time
from korea_stock_auto.utility.kr_config import *
from korea_stock_auto.utility.kr_utils import *

ACCESS_TOKEN = None
TOKEN_TIMESTAMP = 0

APP_KEY = APP_KEY_REAL if USE_REALTIME_API else APP_KEY_VTS
APP_SECRET = APP_SECRET_REAL if USE_REALTIME_API else APP_SECRET_VTS
URL_BASE = URL_BASE_REAL if USE_REALTIME_API else URL_BASE_VTS

def get_access_token():
    global ACCESS_TOKEN, TOKEN_TIMESTAMP

    # 1분 이내에 요청된 경우 기존 토큰 사용
    if ACCESS_TOKEN and (time.time() - TOKEN_TIMESTAMP) < 60:
        return ACCESS_TOKEN

    headers = {'content-type': 'application/json'}
    body = {'grant_type': 'client_credentials', 'appkey': APP_KEY, 'appsecret': APP_SECRET}
    URL = f'{URL_BASE}/oauth2/tokenP'

    res = requests.post(URL, headers=headers, json=body)

    if res.status_code == 200:
        access_token = res.json().get("access_token")
        TOKEN_TIMESTAMP = time.time()   # 토큰 발급 후 타임스탬프 업데이트
        ACCESS_TOKEN = access_token       # 전역 변수에 저장
        print(f"✅ 액세스 토큰 발급 성공 ({'실전' if USE_REALTIME_API else '모의투자'}): {access_token}")
        return access_token
    else:
        print(f"❌ 액세스 토큰 발급 실패 ({'실전' if USE_REALTIME_API else '모의투자'}): {res.text}")
        return None
