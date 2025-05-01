"""
미국 주식 자동매매 - 유틸리티 모듈
메시지 전송, 시간 관련 유틸리티
"""

import datetime
import requests
from pytz import timezone
import time

from us_stock_auto.config import DISCORD_WEBHOOK_URL, TIMEZONE, MARKET_TIME

def send_message(msg):
    """
    디스코드 웹훅을 통한 메시지 전송
    
    Args:
        msg (str): 전송할 메시지
    """
    now = datetime.datetime.now()
    message = {"content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {str(msg)}"}
    
    try:
        requests.post(DISCORD_WEBHOOK_URL, data=message)
        print(message)
    except Exception as e:
        print(f"메시지 전송 실패: {e}")
        print(message)

def get_market_time(time_key):
    """
    특정 시장 시간 조회 함수
    
    Args:
        time_key (str): 시간 설정 키 ('open', 'buy_start', 'sell_start', 'exit')
        
    Returns:
        datetime: 해당 시간 객체
    """
    hour, minute = MARKET_TIME[time_key]
    now = datetime.datetime.now(timezone(TIMEZONE))
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

def is_market_open():
    """
    현재 시장이 열려있는지 확인
    
    Returns:
        bool: 시장 오픈 여부
    """
    now = datetime.datetime.now(timezone(TIMEZONE))
    t_open = get_market_time('open')
    t_exit = get_market_time('exit')
    today = now.weekday()
    
    # 주말(토,일) 또는 장 시간 외에는 Fasle 반환
    if today >= 5 or now < t_open or now > t_exit:
        return False
    return True

def wait(seconds=1):
    """
    지정된 시간만큼 대기
    
    Args:
        seconds (int): 대기 시간(초)
    """
    time.sleep(seconds) 