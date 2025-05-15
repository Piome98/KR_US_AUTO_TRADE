"""
한국 주식 자동매매 - 기본 API 클라이언트 모듈
한국투자증권 API 호출 기능
"""

import json
import requests
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, TypeVar, cast
from datetime import datetime, timedelta
import os

from korea_stock_auto.config import (
    APP_KEY, APP_SECRET, URL_BASE, 
    CANO, ACNT_PRDT_CD, USE_REALTIME_API
)
from korea_stock_auto.utils.utils import (
    send_message, hashkey, create_hmac_signature, 
    handle_http_error, rate_limit_wait, retry_on_failure
)
from korea_stock_auto.api.auth import (
    get_access_token, refresh_token_if_needed, 
    is_token_valid, verify_token_status
)

# 로깅 설정
logger = logging.getLogger("stock_auto")

# 타입 힌트를 위한 제네릭 타입 정의
T = TypeVar('T')
ResponseDict = Dict[str, Any]

# 클래스 정의를 먼저 하고 나중에 Mixin 클래스들을 적용
class KoreaInvestmentApiClient:
    """
    한국투자증권 API 클라이언트
    
    한국투자증권 Open API를 호출하기 위한 기본 클래스입니다.
    인증, 요청 생성, 응답 처리 등의 공통 기능을 제공합니다.
    """
    
    def __init__(self) -> None:
        """
        API 클라이언트 초기화
        
        액세스 토큰을 발급받고 기본 설정을 초기화합니다.
        """
        self.access_token: Optional[str] = None
        self.token_expired_at: Optional[datetime] = None
        self.last_request_time: float = 0
        self.request_interval: float = 0.1  # 최소 API 호출 간격(초)
        
        # 액세스 토큰 발급
        self.issue_access_token()
    
    def _get_cache_path(self, filename: str) -> str:
        """
        캐시 파일 경로 반환
        
        Args:
            filename: 캐시 파일명
            
        Returns:
            str: 캐시 파일의 전체 경로
        """
        return os.path.join(os.path.dirname(__file__), '../../../../data/cache', filename)
    
    def issue_access_token(self) -> bool:
        """
        액세스 토큰 발급
        
        한국투자증권 API에 인증하여 액세스 토큰을 발급받습니다.
        
        Returns:
            bool: 토큰 발급 성공 여부
        """
        try:
            access_token = get_access_token()
            if not access_token:
                logger.error("액세스 토큰 발급 결과가 없습니다.")
                return False
            
            # 토큰 저장
            self.access_token = access_token
            
            # 토큰 만료 시간 설정 (기본 1일, 60초 일찍 만료되도록 설정)
            self.token_expired_at = datetime.now() + timedelta(seconds=86400 - 60)
            
            logger.info("액세스 토큰 발급 성공")
            return True
        except Exception as e:
            logger.error(f"액세스 토큰 발급 실패: {e}", exc_info=True)
            send_message(f"[오류] 액세스 토큰 발급 실패: {e}")
            return False
    
    def _get_headers(self, tr_id: str, hashkey_val: Optional[str] = None) -> Dict[str, str]:
        """
        API 요청 헤더 생성
        
        Args:
            tr_id: 트랜잭션 ID
            hashkey_val: 해시키 값 (주문 API 등에서 사용)
            
        Returns:
            dict: API 요청에 필요한 헤더 정보
        
        Raises:
            RuntimeError: 토큰이 없거나 갱신에 실패한 경우
        """
        # 토큰 만료 여부 확인 및 갱신
        if not self.access_token or (self.token_expired_at and datetime.now() > self.token_expired_at):
            # 토큰이 없거나 만료된 경우 새로 발급
            if not self.issue_access_token():
                raise RuntimeError("액세스 토큰 발급 및 갱신 실패")
        
        if not self.access_token:
            raise RuntimeError("액세스 토큰이 없습니다")
            
        headers = {
            "authorization": f"Bearer {self.access_token}",
            "appKey": APP_KEY,
            "appSecret": APP_SECRET,
            "tr_id": tr_id,
            "custtype": "P",
            "content-type": "application/json; charset=utf-8"
        }
        
        if hashkey_val:
            headers["hashkey"] = hashkey_val
            
        return headers
    
    def _rate_limit(self) -> None:
        """
        API 호출 속도 제한 적용
        
        API 호출 간격을 유지하기 위해 필요한 경우 대기합니다.
        """
        rate_limit_wait(self)
        self.last_request_time = time.time()
    
    def _handle_response(self, 
                        response: requests.Response, 
                        error_msg: str = "API 요청 실패") -> Optional[ResponseDict]:
        """
        API 응답 처리
        
        Args:
            response: HTTP 응답 객체
            error_msg: 에러 발생시 로그 메시지
            
        Returns:
            dict or None: 응답 JSON 또는 오류 시 None
        """
        try:
            if response.status_code == 200:
                result = response.json()
                
                # API 오류 코드 확인 (rt_cd가 0이 아닌 경우 오류)
                if "rt_cd" in result and result["rt_cd"] != "0":
                    error_detail = f"{result.get('msg_cd', '')}: {result.get('msg1', '')}"
                    logger.error(f"{error_msg} - {error_detail}")
                    send_message(f"[오류] {error_msg} - {error_detail}")
                    return None
                
                return result
            else:
                handle_http_error(response, error_msg)
                return None
        except json.JSONDecodeError:
            logger.error(f"{error_msg}: 응답을 JSON으로 파싱할 수 없습니다", exc_info=True)
            send_message(f"[오류] {error_msg}: 응답 형식 오류")
            return None
        except Exception as e:
            logger.error(f"{error_msg}: {e}", exc_info=True)
            send_message(f"[오류] {error_msg}: {e}")
            return None
    
    @retry_on_failure(max_retries=3, base_wait=1)
    def _request_get(self, 
                    url: str, 
                    headers: Dict[str, str], 
                    params: Dict[str, str], 
                    error_msg: str) -> Optional[ResponseDict]:
        """
        GET 요청 수행
        
        Args:
            url: 요청 URL
            headers: 요청 헤더
            params: 요청 파라미터
            error_msg: 오류 메시지
            
        Returns:
            dict or None: 응답 데이터 또는 오류 시 None
        """
        self._rate_limit()
        response = requests.get(url, headers=headers, params=params, timeout=10)
        return self._handle_response(response, error_msg)
    
    @retry_on_failure(max_retries=3, base_wait=1)
    def _request_post(self, 
                     url: str, 
                     headers: Dict[str, str], 
                     data: Dict[str, Any], 
                     error_msg: str) -> Optional[ResponseDict]:
        """
        POST 요청 수행
        
        Args:
            url: 요청 URL
            headers: 요청 헤더
            data: 요청 데이터
            error_msg: 오류 메시지
            
        Returns:
            dict or None: 응답 데이터 또는 오류 시 None
        """
        self._rate_limit()
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
        return self._handle_response(response, error_msg)

# 클래스 정의 후에 Mixin 클래스들을 임포트하고 적용
from korea_stock_auto.api.api_client.account.balance import AccountBalanceMixin
from korea_stock_auto.api.api_client.account.deposit import AccountDepositMixin
from korea_stock_auto.api.api_client.market.stock_info import StockInfoMixin
from korea_stock_auto.api.api_client.market.price import MarketPriceMixin
from korea_stock_auto.api.api_client.market.chart import ChartDataMixin
from korea_stock_auto.api.api_client.order.stock import StockOrderMixin
from korea_stock_auto.api.api_client.order.status import OrderStatusMixin
from korea_stock_auto.api.api_client.sector.index import SectorIndexMixin
from korea_stock_auto.api.api_client.sector.info import SectorInfoMixin

# 다중 상속을 통해 Mixin 클래스들의 메서드를 KoreaInvestmentApiClient에 추가
for mixin in [
    AccountBalanceMixin, 
    AccountDepositMixin,
    StockInfoMixin,
    MarketPriceMixin,
    ChartDataMixin,
    StockOrderMixin,
    OrderStatusMixin,
    SectorIndexMixin,
    SectorInfoMixin
]:
    for name, method in mixin.__dict__.items():
        if not name.startswith('_') and callable(method):
            setattr(KoreaInvestmentApiClient, name, method) 