"""
한국 주식 자동매매 - 기본 API 클라이언트 모듈
한국투자증권 API 호출 기능
"""

import json
import requests
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, TypeVar, cast, TYPE_CHECKING
from datetime import datetime, timedelta
import os

if TYPE_CHECKING:
    from korea_stock_auto.config import AppConfig

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

# 컴포지션 패턴으로 변경된 API 클라이언트
class KoreaInvestmentApiClient:
    """
    한국투자증권 API 클라이언트
    
    한국투자증권 Open API를 호출하기 위한 기본 클래스입니다.
    인증, 요청 생성, 응답 처리 등의 공통 기능을 제공합니다.
    """
    
    def __init__(self, config: 'AppConfig') -> None:
        """
        API 클라이언트 초기화
        
        Args:
            config: 애플리케이션 설정
        """
        self.config = config
        self.access_token: Optional[str] = None
        self.token_expired_at: Optional[datetime] = None
        self.last_request_time: float = 0
        self.request_interval: float = 0.1  # 최소 API 호출 간격(초)
        
        # API 속성들
        self.base_url = config.current_api.base_url
        self.app_key = config.current_api.app_key
        self.app_secret = config.current_api.app_secret
        
        # 액세스 토큰 발급
        self.issue_access_token()
        
        # 컴포지션 패턴: 각 도메인별 서비스 초기화
        from korea_stock_auto.api.patterns import MarketService, OrderService, AccountService
        self.market_service = MarketService(self)
        self.order_service = OrderService(self)
        self.account_service = AccountService(self)
    
    def _get_cache_path(self, filename: str) -> str:
        """
        캐시 파일 경로 반환
        
        Args:
            filename: 캐시 파일명
            
        Returns:
            str: 캐시 파일의 전체 경로
        """
        return os.path.join(os.path.dirname(__file__), '../../../data/cache', filename)
    
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
            send_message(f"[오류] 액세스 토큰 발급 실패: {e}", self.config.notification.discord_webhook_url)
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
            "appkey": self.app_key,
            "appsecret": self.app_secret,
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
                    send_message(f"[오류] {error_msg} - {error_detail}", self.config.notification.discord_webhook_url)
                    return None
                
                return result
            else:
                handle_http_error(response, error_msg)
                return None
        except json.JSONDecodeError:
            logger.error(f"{error_msg}: 응답을 JSON으로 파싱할 수 없습니다", exc_info=True)
            send_message(f"[오류] {error_msg}: 응답 형식 오류", self.config.notification.discord_webhook_url)
            return None
        except Exception as e:
            logger.error(f"{error_msg}: {e}", exc_info=True)
            send_message(f"[오류] {error_msg}: {e}", self.config.notification.discord_webhook_url)
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

    # 기존 인터페이스 호환성을 위한 프록시 메서드들
    def get_current_price(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        특정 종목의 현재가 정보를 조회합니다.
        
        Args:
            stock_code: 종목 코드
            
        Returns:
            dict or None: 현재가 정보 또는 조회 실패 시 None
        """
        return self.market_service.get_current_price(stock_code)
    
    def get_real_time_price_by_api(self, code: str) -> Optional[Dict[str, Any]]:
        """
        실시간 시세 조회 API (MarketService로 위임)
        
        Args:
            code: 종목 코드
            
        Returns:
            dict or None: 실시간 시세 정보
        """
        return self.market_service.get_real_time_price_by_api(code)
    
    def fetch_stock_current_price(self, code: str) -> Optional[Dict[str, Any]]:
        """
        주식 현재가 조회 (MarketService로 위임)
        
        Args:
            code: 종목 코드
            
        Returns:
            dict or None: 주식 현재가 정보
        """
        return self.market_service.fetch_stock_current_price(code)
    
    def buy_stock(self, code: str, qty: int, price: Optional[int] = None) -> bool:
        """
        주식 매수 (OrderService로 위임)
        
        Args:
            code: 종목 코드
            qty: 주문 수량
            price: 주문 가격
            
        Returns:
            bool: 매수 성공 여부
        """
        return self.order_service.buy_stock(code, qty, price)
    
    def sell_stock(self, code: str, qty: int, price: Optional[int] = None) -> bool:
        """
        주식 매도 (OrderService로 위임)
        
        Args:
            code: 종목 코드
            qty: 주문 수량
            price: 주문 가격
            
        Returns:
            bool: 매도 성공 여부
        """
        return self.order_service.sell_stock(code, qty, price)
    
    def fetch_buyable_amount(self, code: str, price: int = 0) -> Optional[Dict[str, Any]]:
        """
        매수 가능 금액 조회 (OrderService로 위임)
        
        Args:
            code: 종목 코드
            price: 주문 가격
            
        Returns:
            dict or None: 매수 가능 정보
        """
        return self.order_service.fetch_buyable_amount(code, price)
    
    def get_account_balance(self) -> Optional[Dict[str, Any]]:
        """
        계좌 잔고 조회 (AccountService로 위임)
        
        Returns:
            dict or None: 계좌 정보
        """
        return self.account_service.get_account_balance()
    
    def get_deposit_info(self) -> Optional[Dict[str, Any]]:
        """
        예수금 정보 조회 (AccountService로 위임)
        
        Returns:
            dict or None: 예수금 정보
        """
        return self.account_service.get_deposit_info()
    
    def get_top_traded_stocks(self, market_type: str = "0", top_n: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        거래량 상위 종목 조회
        
        Args:
            market_type: 시장 구분 ("0": 전체, "1": 코스피, "2": 코스닥)
            top_n: 조회할 상위 종목 수
            
        Returns:
            list or None: 거래량 상위 종목 목록
        """
        return self.market_service.get_top_traded_stocks(market_type, top_n)
    
    def get_stock_balance(self) -> Optional[Dict[str, Any]]:
        """
        주식 보유 종목 조회
        
        Returns:
            dict or None: 보유 종목 정보
        """
        return self.account_service.get_stock_balance()
    
    def is_etf_stock(self, code: str) -> bool:
        """
        ETF 종목 여부 확인
        
        Args:
            code: 종목코드
            
        Returns:
            bool: ETF 종목 여부
        """
        return self.market_service.is_etf_stock(code)
    
    def fetch_balance(self) -> Optional[Dict[str, Any]]:
        """
        계좌 잔고 원시 데이터 조회
        
        Returns:
            dict or None: 원시 계좌 잔고 데이터
        """
        return self.account_service.fetch_balance() 