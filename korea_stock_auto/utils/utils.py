"""
한국 주식 자동매매 - 유틸리티 모듈

메시지 전송, 시간 관련 유틸리티, API 요청 관련 유틸리티 함수를 제공합니다.
"""

import datetime
import requests
import time
import json
import random
import logging
import hashlib
import base64
import hmac
import os
from typing import Dict, Any, Optional, Union, List, Callable, TypeVar, cast

# 타입 변수 정의
T = TypeVar('T')
R = TypeVar('R')

# 로깅 설정
logger = logging.getLogger(__name__)

def setup_logger(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    로깅 설정
    
    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 로그 파일 경로 (None이면 콘솔에만 출력)
    """
    # 로그 레벨 매핑
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level_num = level_map.get(log_level.upper(), logging.INFO)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_num)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_num)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러 추가 (지정된 경우)
    if log_file:
        # 로그 디렉토리 생성
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level_num)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 모듈별 로거 설정
    stock_logger = logging.getLogger("stock_auto")
    stock_logger.setLevel(log_level_num)
    stock_logger.propagate = True
    
    logger.info(f"로깅 설정 완료 (레벨: {log_level}, 파일: {log_file if log_file else '없음'})")

def send_message(msg: str, webhook_url: Optional[str] = None) -> None:
    """
    디스코드 웹훅을 통한 메시지 전송
    
    Args:
        msg: 전송할 메시지
        webhook_url: 웹훅 URL (None이면 콘솔에만 출력)
    """
    now = datetime.datetime.now()
    message = {"content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {str(msg)}"}
    
    try:
        if webhook_url:
            response = requests.post(webhook_url, data=message)
            response.raise_for_status()
        print(message)
    except Exception as e:
        print(f"Discord 메시지 전송 오류: {e}")
        print(message)
    
    # API 호출 제한을 피하기 위한 대기
    time.sleep(0.5)

def hashkey(datas: Dict[str, Any], app_key: str, app_secret: str, url_base: str) -> str:
    """
    API 요청을 위한 해시 키 생성
    
    Args:
        datas: 암호화할 데이터
        app_key: 앱 키
        app_secret: 앱 시크릿
        url_base: 기본 URL
    
    Returns:
        생성된 해시 키
    """
    url = f"{url_base}/uapi/hashkey"
    headers = {
        "Content-Type": "application/json", 
        "appKey": app_key,
        "appSecret": app_secret
    }
    
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            logger.info("해시키 생성 요청 중...")
            res = requests.post(url, headers=headers, data=json.dumps(datas), timeout=10)
            res.raise_for_status()
            result = res.json()
            hash_value = result.get("HASH", "")
            
            if not hash_value:
                raise ValueError("해시키가 비어있습니다")
            
            logger.info("해시키 생성 성공")
            return hash_value
            
        except requests.exceptions.HTTPError as http_err:
            retry_count += 1
            wait_time = 1 + (retry_count * 2)
            error_msg = f"해시키 생성 HTTP 오류: {http_err}, {retry_count}/{max_retries} 재시도 ({wait_time}초 대기)"
            logger.error(error_msg)
            print(f"오류: {error_msg}")
            
            if res and hasattr(res, 'text'):
                logger.error("응답 내용: %s", res.text)
            
            time.sleep(wait_time)
        
        except requests.exceptions.Timeout:
            retry_count += 1
            wait_time = 1 + (retry_count * 2)
            error_msg = f"해시키 생성 시간 초과, {retry_count}/{max_retries} 재시도 ({wait_time}초 대기)"
            logger.error(error_msg)
            print(f"오류: {error_msg}")
            time.sleep(wait_time)
            
        except Exception as e:
            retry_count += 1
            wait_time = 1 + (retry_count * 2)
            error_msg = f"해시키 생성 오류: {e}, {retry_count}/{max_retries} 재시도 ({wait_time}초 대기)"
            logger.error(error_msg)
            print(f"오류: {error_msg}")
            time.sleep(wait_time)
    
    error_msg = "해시키 생성 최대 재시도 횟수 초과"
    logger.error(error_msg)
    print(f"오류: {error_msg}")
    return ""

def create_hmac_signature(data_to_sign: Union[str, Dict[str, Any]], secret_key: str) -> str:
    """
    HMAC 서명 생성 (한국투자증권 API 요청에 필요한 경우가 있음)
    
    Args:
        data_to_sign: 서명할 데이터 (문자열 또는 딕셔너리)
        secret_key: 비밀키
        
    Returns:
        생성된 HMAC 서명
    """
    if isinstance(data_to_sign, dict):
        data_to_sign = json.dumps(data_to_sign)
    
    byte_key = bytes(secret_key, 'UTF-8')
    byte_data = bytes(data_to_sign, 'UTF-8')
    
    hash_obj = hmac.new(byte_key, byte_data, hashlib.sha256)
    hmac_signature = base64.b64encode(hash_obj.digest()).decode('UTF-8')
    
    logger.debug("HMAC 서명 생성 완료")
    return hmac_signature

def format_query_params(params: Dict[str, Any]) -> str:
    """
    API 요청용 쿼리 파라미터 포맷팅
    
    Args:
        params: 쿼리 파라미터
        
    Returns:
        포맷팅된 쿼리 문자열
    """
    return '&'.join([f"{k}={v}" for k, v in params.items() if v is not None])

def wait(seconds: float = 1.0, jitter: float = 0.5) -> None:
    """
    지정된 시간만큼 대기 (API 레이트 리밋 방지용)
    
    Args:
        seconds: 기본 대기 시간(초)
        jitter: 무작위 변동폭 최대값(초)
    """
    # 지터 추가: 동시 요청 방지
    jitter_amount = random.uniform(0, jitter)
    wait_time = seconds + jitter_amount
    time.sleep(wait_time)

def rate_limit_wait(client_or_base_time: Any = 2.0) -> None:
    """
    API 레이트 리밋 방지를 위한 대기
    
    Args:
        client_or_base_time: 클라이언트 객체 또는 기본 대기 시간(초)
    """
    # 클라이언트 객체가 전달된 경우 기본값 사용
    if not isinstance(client_or_base_time, (int, float)):
        base_time = 2.0  # 기본 2초로 증가 (초당 거래건수 제한 회피)
    else:
        base_time = client_or_base_time
    
    # 1.0~2.0초의 랜덤 지터 추가 (더 안전한 간격)
    jitter = random.uniform(1.0, 2.0)
    wait_time = base_time + jitter
    time.sleep(wait_time)
    logger.debug(f"API 레이트 리밋 대기: {wait_time:.2f}초")

def handle_http_error(response: requests.Response, error_msg_prefix: str = "API 요청 오류") -> Dict[str, Any]:
    """
    HTTP 오류 응답 처리
    
    Args:
        response: 응답 객체
        error_msg_prefix: 오류 메시지 접두사
        
    Returns:
        오류 정보를 담은 딕셔너리
    """
    error_info = {
        "success": False,
        "status_code": response.status_code,
        "error": f"{error_msg_prefix}: HTTP {response.status_code}"
    }
    
    try:
        # JSON 응답이 있는 경우
        error_data = response.json()
        error_info["error_data"] = error_data
        
        # 한국투자증권 API 에러 메시지 형식
        if "error_description" in error_data:
            error_info["message"] = error_data["error_description"]
        elif "rt_cd" in error_data:
            error_info["rt_cd"] = error_data["rt_cd"]
            error_info["msg_cd"] = error_data.get("msg_cd", "")
            error_info["message"] = error_data.get("msg1", "알 수 없는 오류")
    except:
        # JSON 아닌 경우 텍스트 응답
        error_info["response_text"] = response.text
    
    error_msg = f"{error_msg_prefix}: {response.status_code} - {error_info.get('message', response.text)}"
    logger.error(error_msg)
    # Note: webhook_url은 호출하는 쪽에서 별도로 전송해야 함
    
    return error_info
    
def format_datetime(dt: Optional[datetime.datetime] = None) -> str:
    """
    날짜 시간 포맷팅
    
    Args:
        dt: 포맷팅할 날짜시간, None이면 현재시간
        
    Returns:
        포맷팅된 날짜시간 문자열 (YYYY-MM-DD HH:MM:SS)
    """
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime('%Y-%m-%d %H:%M:%S')
    
def format_date(dt: Optional[datetime.datetime] = None) -> str:
    """
    날짜 포맷팅
    
    Args:
        dt: 포맷팅할 날짜, None이면 현재날짜
        
    Returns:
        포맷팅된 날짜 문자열 (YYYYMMDD)
    """
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime('%Y%m%d')

def retry_on_failure(max_retries: int = 3, base_wait: float = 1.0, 
                    error_msg_prefix: str = "작업 실패"):
    """
    실패 시 재시도 데코레이터 팩토리
    
    Args:
        max_retries: 최대 재시도 횟수
        base_wait: 기본 대기 시간
        error_msg_prefix: 오류 메시지 접두사
        
    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable[..., R]) -> Callable[..., Optional[R]]:
        def wrapper(*args: Any, **kwargs: Any) -> Optional[R]:
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    if retry_count > 0:
                        logger.info(f"재시도 {retry_count}/{max_retries} 중...")
                    
                    result = func(*args, **kwargs)
                    if retry_count > 0:
                        logger.info(f"재시도 성공 ({retry_count}/{max_retries})")
                    
                    return result
                    
                except Exception as e:
                    retry_count += 1
                    wait_time = base_wait * (2 ** (retry_count - 1))  # 지수 백오프
                    
                    if retry_count <= max_retries:
                        error_msg = f"{error_msg_prefix}: {e}, {retry_count}/{max_retries} 재시도 ({wait_time:.1f}초 대기)"
                        logger.warning(error_msg)
                        time.sleep(wait_time)
                    else:
                        error_msg = f"{error_msg_prefix}: {e}, 최대 재시도 횟수 초과"
                        logger.error(error_msg)
                        # Note: webhook_url은 호출하는 쪽에서 별도로 전송해야 함
                        return None
            
            return None
        
        return cast(Callable[..., Optional[R]], wrapper)
    
    return decorator 