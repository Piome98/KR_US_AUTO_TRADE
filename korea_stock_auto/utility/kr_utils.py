# 유틸리티 함수 모음
# API request 모듈로 REST API 기능 이용
# korea_stock_auto.utility.real_utils.kr_utils

import requests
import json
import datetime
import time
from korea_stock_auto.utility.kr_config import DISCORD_WEBHOOK_URL, APP_KEY, APP_SECRET, URL_BASE

def send_message(msg):
    """
    디스코드 웹훅을 이용해 매수/매도 체결 및 작동 중이라는 정보 전달.
    예외 발생 시 에러를 출력합니다.
    """
    now = datetime.datetime.now()
    message = {"content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, data=message)
        response.raise_for_status()
    except Exception as e:
        print(f"Error sending Discord message: {e}")
    print(message)
    time.sleep(1)

def hashkey(datas):
    """
    API 요청을 위한 해시 키 생성.
    API 호출 중 오류 발생 시 에러 메시지를 로깅하고 빈 문자열을 반환합니다.
    """
    url = f"{URL_BASE}/uapi/hashkey"
    headers = {"Content-Type": "application/json", "appKey": APP_KEY, "appSecret": APP_SECRET}
    try:
        res = requests.post(url, headers=headers, data=json.dumps(datas))
        res.raise_for_status()
        result = res.json()
        return result.get("HASH", "")
    except Exception as e:
        send_message(f"Error generating hashkey: {e}")
        return ""
