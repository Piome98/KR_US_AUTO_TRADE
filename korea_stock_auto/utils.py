"""
한국 주식 자동매매 - 유틸리티 모듈
메시지 전송, 시간 관련 유틸리티
"""

import datetime
import requests
import time
import json

from korea_stock_auto.config import DISCORD_WEBHOOK_URL, APP_KEY, APP_SECRET, URL_BASE

def send_message(msg):
    """
    디스코드 웹훅을 통한 메시지 전송
    
    Args:
        msg (str): 전송할 메시지
    """
    now = datetime.datetime.now()
    message = {"content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {str(msg)}"}
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, data=message)
        response.raise_for_status()
        print(message)
    except Exception as e:
        print(f"Error sending Discord message: {e}")
        print(message)
    
    # API 호출 제한을 피하기 위한 대기
    time.sleep(1)

def hashkey(datas):
    """
    API 요청을 위한 해시 키 생성
    
    Args:
        datas (dict): 암호화할 데이터
    
    Returns:
        str: 생성된 해시 키
    """
    url = f"{URL_BASE}/uapi/hashkey"
    headers = {
        "Content-Type": "application/json", 
        "appKey": APP_KEY,
        "appSecret": APP_SECRET
    }
    
    try:
        res = requests.post(url, headers=headers, data=json.dumps(datas))
        res.raise_for_status()
        result = res.json()
        return result.get("HASH", "")
    except Exception as e:
        send_message(f"Error generating hashkey: {e}")
        return ""

def wait(seconds=1):
    """
    지정된 시간만큼 대기
    
    Args:
        seconds (int): 대기 시간(초)
    """
    time.sleep(seconds) 