"""
한국 주식 자동매매 - 기본 API 클라이언트 모듈
한국투자증권 API 호출 기능
"""

import json
import requests
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta

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

# 클래스 정의를 먼저 하고 나중에 Mixin 클래스들을 적용
class KoreaInvestmentApiClient:
    """한국투자증권 API 클라이언트"""
    
    def __init__(self):
        """API 클라이언트 초기화"""
        self.access_token = None
        self.token_expired_at = None
        self.last_request_time = 0
        self.request_interval = 0.1  # 최소 API 호출 간격(초)
        
        # 액세스 토큰 발급
        self.issue_access_token()
    
    def issue_access_token(self) -> bool:
        """
        액세스 토큰 발급
        
        Returns:
            bool: 토큰 발급 성공 여부
        """
        try:
            result = get_access_token()
            if not result:
                return False
            
            self.access_token = result["access_token"]
            self.token_expired_at = datetime.now() + timedelta(seconds=result["expires_in"] - 60)
            
            logger.info("액세스 토큰 발급 성공")
            return True
        except Exception as e:
            logger.error(f"액세스 토큰 발급 실패: {e}", exc_info=True)
            return False
    
    def _get_headers(self, tr_id: str, hashkey_val: str = None) -> Dict[str, str]:
        """
        API 요청 헤더 생성
        
        Args:
            tr_id (str): 트랜잭션 ID
            hashkey_val (str, optional): 해시키 값
            
        Returns:
            dict: 헤더 정보
        """
        # 토큰 만료 여부 확인 및 갱신
        if not refresh_token_if_needed(self):
            self.issue_access_token()
            
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
        """API 호출 속도 제한 적용"""
        rate_limit_wait(self)
        self.last_request_time = time.time()
    
    def _handle_response(self, response: requests.Response, error_msg: str = "API 요청 실패") -> Optional[Dict[str, Any]]:
        """
        API 응답 처리
        
        Args:
            response (Response): HTTP 응답 객체
            error_msg (str): 에러 발생시 로그 메시지
            
        Returns:
            dict or None: 응답 JSON 또는 None
        """
        try:
            if response.status_code == 200:
                return response.json()
            else:
                handle_http_error(response, error_msg)
                return None
        except Exception as e:
            logger.error(f"{error_msg}: {e}", exc_info=True)
            send_message(f"[오류] {error_msg}: {e}")
            return None

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