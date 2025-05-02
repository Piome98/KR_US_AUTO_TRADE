"""
한국 주식 자동매매 - 인증 모듈
API 토큰 인증 및 관리
"""

import requests
import json
import time

from korea_stock_auto.config import (
    APP_KEY, APP_SECRET, URL_BASE, USE_REALTIME_API
)
from korea_stock_auto.utils.utils import send_message

# 토큰 관련 전역 변수
ACCESS_TOKEN = None
TOKEN_TIMESTAMP = 0

def get_access_token():
    """
    API 접근을 위한 토큰 발급
    
    Returns:
        str: 발급된 액세스 토큰
    """
    global ACCESS_TOKEN, TOKEN_TIMESTAMP
    
    # 1분 이내에 요청된 경우 기존 토큰 사용
    if ACCESS_TOKEN and (time.time() - TOKEN_TIMESTAMP) < 60:
        return ACCESS_TOKEN
    
    headers = {'content-type': 'application/json'}
    body = {'grant_type': 'client_credentials', 'appkey': APP_KEY, 'appsecret': APP_SECRET}
    url = f'{URL_BASE}/oauth2/tokenP'
    
    try:
        res = requests.post(url, headers=headers, json=body)
        res.raise_for_status()
        
        access_token = res.json().get("access_token")
        if access_token:
            TOKEN_TIMESTAMP = time.time()   # 토큰 발급 후 타임스탬프 업데이트
            ACCESS_TOKEN = access_token     # 전역 변수에 저장
            send_message(f"액세스 토큰 발급 성공 ({'실전' if USE_REALTIME_API else '모의투자'}): {access_token}")
            return access_token
        else:
            send_message(f"액세스 토큰 발급 실패: 토큰 정보 없음")
            return None
    except Exception as e:
        send_message(f"액세스 토큰 발급 실패: {e}")
        return None

def request_ws_connection_key():
    """
    WebSocket 접속키 요청 함수
    
    Returns:
        str: 발급된 WebSocket 접속키
    """
    approval_url = f"{URL_BASE}/oauth2/Approval"
    headers = {"Content-Type": "application/json"}
    body = {"grant_type": "client_credentials", "appkey": APP_KEY, "secretkey": APP_SECRET}
    
    try:
        res = requests.post(approval_url, headers=headers, json=body)
        res.raise_for_status()
        
        ws_conn_key = res.json().get("approval_key")
        if ws_conn_key:
            send_message(f"WebSocket 접속키 발급 성공: {ws_conn_key}")
            return ws_conn_key
        else:
            send_message(f"WebSocket 접속키 발급 실패: 접속키 정보 없음")
            return None
    except Exception as e:
        send_message(f"WebSocket 접속키 발급 실패: {e}")
        return None 