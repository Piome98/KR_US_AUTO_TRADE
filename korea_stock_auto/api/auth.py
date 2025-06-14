"""
한국 주식 자동매매 - 인증 모듈
API 토큰 인증 및 관리
"""

import requests
import json
import time
import datetime
import logging
from typing import Optional, Dict, Any, Tuple, Union

import os
import sys
import yaml

# 설정 직접 로드
def _load_config():
    """config.yaml에서 직접 설정 로드"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # API 설정 섹션 가져오기
        api_config = config.get('API', {})
        USE_REALTIME_API = api_config.get('USE_REALTIME_API', True)
        
        if USE_REALTIME_API:
            real_config = api_config.get('REAL', {})
            APP_KEY = real_config.get('APP_KEY', '')
            APP_SECRET = real_config.get('APP_SECRET', '')
            URL_BASE = real_config.get('BASE_URL', 'https://openapi.koreainvestment.com:9443')
        else:
            vts_config = api_config.get('VTS', {})
            APP_KEY = vts_config.get('APP_KEY', '')
            APP_SECRET = vts_config.get('APP_SECRET', '')
            URL_BASE = vts_config.get('BASE_URL', 'https://openapivts.koreainvestment.com:29443')
        
        return APP_KEY, APP_SECRET, URL_BASE, USE_REALTIME_API
    except Exception as e:
        # logger가 아직 정의되지 않았을 수 있으므로 print 사용
        print(f"설정 로드 실패: {e}")
        return "", "", "", False

APP_KEY, APP_SECRET, URL_BASE, USE_REALTIME_API = _load_config()
from korea_stock_auto.utils.utils import send_message

# 로깅 설정
logger = logging.getLogger(__name__)

# 토큰 관련 전역 변수
ACCESS_TOKEN = None
TOKEN_EXPIRES_AT = 0  # 토큰 만료 시간 (timestamp)
TOKEN_TIMESTAMP = 0   # 토큰 발급 시간 (timestamp)
REFRESH_TOKEN = None  # 리프레시 토큰

def get_access_token(force_refresh: bool = False) -> Optional[str]:
    """
    API 접근을 위한 토큰 발급 또는 갱신
    
    Args:
        force_refresh (bool): 강제로 토큰을 새로 발급받을지 여부
    
    Returns:
        str: 발급된 액세스 토큰, 실패 시 None
    """
    global ACCESS_TOKEN, TOKEN_TIMESTAMP, TOKEN_EXPIRES_AT, REFRESH_TOKEN
    
    # 현재 시간
    current_time = time.time()
    
    # 토큰이 유효하고 강제 갱신이 아니라면 기존 토큰 사용
    if not force_refresh and ACCESS_TOKEN and (TOKEN_EXPIRES_AT - current_time > 300):
        logger.debug("기존 토큰 사용 (만료까지 %d초 남음)", TOKEN_EXPIRES_AT - current_time)
        return ACCESS_TOKEN
    
    # 토큰 발급 요청
    headers = {'content-type': 'application/json'}
    body = {'grant_type': 'client_credentials', 'appkey': APP_KEY, 'appsecret': APP_SECRET}
    
    # 모의투자와 실전투자 모두 동일한 엔드포인트 사용
    token_endpoint = "oauth2/tokenP"
    url = f'{URL_BASE}/{token_endpoint}'
    
    try:
        logger.info("토큰 발급 요청 중...")
        res = requests.post(url, headers=headers, json=body, timeout=10)
        res.raise_for_status()
        
        # 응답 처리
        response_data = res.json()
        access_token = response_data.get("access_token")
        expires_in = response_data.get("expires_in", 86400)  # 기본값 1일
        refresh_token = response_data.get("refresh_token")
        
        if access_token:
            TOKEN_TIMESTAMP = current_time  # 토큰 발급 시간 기록
            TOKEN_EXPIRES_AT = current_time + expires_in  # 토큰 만료 시간 계산
            ACCESS_TOKEN = access_token  # 전역 변수에 저장
            
            if refresh_token:
                REFRESH_TOKEN = refresh_token
                logger.info("리프레시 토큰 저장 완료")
            
            # 만료 시간 포맷팅
            expiry_time = datetime.datetime.fromtimestamp(TOKEN_EXPIRES_AT).strftime('%Y-%m-%d %H:%M:%S')
            
            send_message(f"액세스 토큰 발급 성공 ({'실전' if USE_REALTIME_API else '모의투자'})")
            send_message(f"토큰 만료 시간: {expiry_time}")
            logger.info("액세스 토큰 발급 성공, 만료 시간: %s", expiry_time)
            
            return access_token
        else:
            error_msg = f"액세스 토큰 발급 실패: 토큰 정보 없음"
            send_message(error_msg)
            logger.error(error_msg)
            return None
    except requests.exceptions.HTTPError as http_err:
        error_msg = f"액세스 토큰 발급 HTTP 오류: {http_err}"
        send_message(error_msg)
        logger.error(error_msg)
        if res and hasattr(res, 'text'):
            logger.error("응답 내용: %s", res.text)
            send_message(f"응답 내용: {res.text}")
        return None
    except requests.exceptions.Timeout:
        error_msg = "액세스 토큰 발급 시간 초과"
        send_message(error_msg)
        logger.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"액세스 토큰 발급 실패: {e}"
        send_message(error_msg)
        logger.error(error_msg)
        return None

def refresh_access_token() -> Optional[str]:
    """
    리프레시 토큰을 사용하여 액세스 토큰 갱신
    
    Returns:
        str: 갱신된 액세스 토큰, 실패 시 None
    """
    global ACCESS_TOKEN, TOKEN_TIMESTAMP, TOKEN_EXPIRES_AT, REFRESH_TOKEN
    
    if not REFRESH_TOKEN:
        logger.warning("리프레시 토큰이 없어 새 토큰을 발급합니다.")
        return get_access_token(force_refresh=True)
    
    # 토큰 갱신 요청
    headers = {'content-type': 'application/json'}
    body = {
        'grant_type': 'refresh_token', 
        'appkey': APP_KEY, 
        'appsecret': APP_SECRET,
        'refresh_token': REFRESH_TOKEN
    }
    
    # 모의투자와 실전투자 모두 동일한 엔드포인트 사용
    token_endpoint = "oauth2/tokenP"
    url = f'{URL_BASE}/{token_endpoint}'
    
    try:
        logger.info("토큰 갱신 요청 중...")
        res = requests.post(url, headers=headers, json=body, timeout=10)
        res.raise_for_status()
        
        # 응답 처리
        response_data = res.json()
        access_token = response_data.get("access_token")
        expires_in = response_data.get("expires_in", 86400)  # 기본값 1일
        new_refresh_token = response_data.get("refresh_token")
        
        if access_token:
            current_time = time.time()
            TOKEN_TIMESTAMP = current_time  # 토큰 발급 시간 기록
            TOKEN_EXPIRES_AT = current_time + expires_in  # 토큰 만료 시간 계산
            ACCESS_TOKEN = access_token  # 전역 변수에 저장
            
            if new_refresh_token:
                REFRESH_TOKEN = new_refresh_token
            
            # 만료 시간 포맷팅
            expiry_time = datetime.datetime.fromtimestamp(TOKEN_EXPIRES_AT).strftime('%Y-%m-%d %H:%M:%S')
            
            send_message(f"액세스 토큰 갱신 성공")
            send_message(f"토큰 만료 시간: {expiry_time}")
            logger.info("액세스 토큰 갱신 성공, 만료 시간: %s", expiry_time)
            
            return access_token
        else:
            error_msg = "액세스 토큰 갱신 실패: 토큰 정보 없음"
            send_message(error_msg)
            logger.error(error_msg)
            
            # 갱신 실패 시 새로운 토큰 발급
            logger.info("갱신 실패로 새 토큰 발급 시도")
            return get_access_token(force_refresh=True)
    except Exception as e:
        error_msg = f"액세스 토큰 갱신 실패: {e}"
        send_message(error_msg)
        logger.error(error_msg)
        
        # 갱신 실패 시 새로운 토큰 발급
        logger.info("갱신 실패로 새 토큰 발급 시도")
        return get_access_token(force_refresh=True)

def revoke_token() -> bool:
    """
    액세스 토큰 폐기 함수
    
    Returns:
        bool: 폐기 성공 여부
    """
    global ACCESS_TOKEN, REFRESH_TOKEN
    
    if not ACCESS_TOKEN:
        logger.warning("폐기할 액세스 토큰이 없습니다.")
        return False
    
    # 토큰 폐기 요청
    headers = {
        'content-type': 'application/json',
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }
    body = {'appkey': APP_KEY, 'appsecret': APP_SECRET}
    
    # 모의투자와 실전투자 모두 동일한 엔드포인트 사용
    revoke_endpoint = "oauth2/revoke"
    url = f'{URL_BASE}/{revoke_endpoint}'
    
    try:
        logger.info("토큰 폐기 요청 중...")
        res = requests.post(url, headers=headers, json=body, timeout=10)
        res.raise_for_status()
        
        response_data = res.json()
        result = response_data.get("result") == "success"
        
        if result:
            send_message("액세스 토큰 폐기 성공")
            logger.info("액세스 토큰 폐기 성공")
            ACCESS_TOKEN = None
            REFRESH_TOKEN = None
            TOKEN_EXPIRES_AT = 0
            TOKEN_TIMESTAMP = 0
            return True
        else:
            error_msg = f"액세스 토큰 폐기 실패: {response_data}"
            send_message(error_msg)
            logger.error(error_msg)
            return False
    except Exception as e:
        error_msg = f"액세스 토큰 폐기 중 오류 발생: {e}"
        send_message(error_msg)
        logger.error(error_msg)
        return False

def is_token_valid() -> bool:
    """
    현재 토큰의 유효성 확인
    
    Returns:
        bool: 토큰 유효 여부
    """
    global ACCESS_TOKEN, TOKEN_EXPIRES_AT
    
    if not ACCESS_TOKEN:
        return False
    
    # 토큰 만료 5분 전까지만 유효하다고 간주
    return time.time() < (TOKEN_EXPIRES_AT - 300)

def refresh_token_if_needed() -> bool:
    """
    필요 시 토큰 갱신
    
    Returns:
        bool: 갱신 성공 여부
    """
    if not is_token_valid():
        logger.info("토큰이 유효하지 않아 갱신을 시도합니다.")
        if REFRESH_TOKEN:
            logger.info("리프레시 토큰을 사용해 토큰 갱신 시도")
            new_token = refresh_access_token()
        else:
            logger.info("리프레시 토큰이 없어 새 토큰 발급 시도")
            new_token = get_access_token(force_refresh=True)
        
        return new_token is not None
    
    logger.debug("토큰이 아직 유효함 (갱신 불필요)")
    return True


def verify_token_status(token: Optional[str] = None) -> Dict[str, Any]:
    """
    토큰 상태 확인 함수
    
    Args:
        token (str, optional): 확인할 토큰. 기본값은 현재 저장된 토큰
        
    Returns:
        Dict[str, Any]: 토큰 상태 정보
    """
    if token is None:
        token = ACCESS_TOKEN
        
    if not token:
        return {"valid": False, "error": "토큰이 없습니다."}
    
    # 모의투자와 실전투자 모두 동일한 엔드포인트 사용
    status_endpoint = "oauth2/token/status"
    status_url = f"{URL_BASE}/{status_endpoint}"
    
    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    body = {"appkey": APP_KEY, "appsecret": APP_SECRET}
    
    try:
        res = requests.post(status_url, headers=headers, json=body, timeout=10)
        res.raise_for_status()
        
        response_data = res.json()
        return {
            "valid": True,
            "expires_in": response_data.get("expires_in", 0),
            "issued_at": response_data.get("issued_at", ""),
            "access_token": token
        }
    except Exception as e:
        error_msg = f"토큰 상태 확인 중 오류 발생: {e}"
        logger.error(error_msg)
        return {"valid": False, "error": str(e)} 