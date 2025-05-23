"""
한국 주식 자동매매 - 시장 가격 조회 모듈
시장 가격 조회 및 실시간 가격 정보 관련 기능 제공
"""

import logging
import os
import json
import datetime
import requests
import time
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING, cast
from pathlib import Path

from korea_stock_auto.config import URL_BASE
from korea_stock_auto.utils.utils import send_message

# 타입 힌트만을 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

# 로깅 설정
logger = logging.getLogger("stock_auto")

class MarketPriceMixin:
    """
    시장 가격 조회 관련 기능 Mixin
    
    주식 현재가, 호가, 체결 정보 등 시장 가격 관련 조회 기능을 제공합니다.
    """
    
    def is_etf_stock(self, code: str) -> bool:
        """
        종목 코드가 ETF인지 확인합니다.
        
        Args:
            code (str): 6자리 종목 코드
            
        Returns:
            bool: ETF 여부
            
        Notes:
            대표적인 ETF 종목 코드 패턴:
            - 일반적으로 ETF는 특정 패턴을 가집니다:
              * KODEX 계열: 069500, 114800, 152100, 229200, 305720 등
              * TIGER 계열: 228790, 251350, 360750, 364980 등
              * KBSTAR, SOL 등: 270300, 466920 등
              * 하지만 특정 패턴만으로는 정확한 구분이 어렵고
              * 개별 주식(462870-시프트업, 272210-한화시스템 등)도 혼재
            
            더 정확한 방법은 API를 통해 상품구분코드를 확인하는 것이지만,
            현재는 알려진 ETF 종목코드 리스트를 사용합니다.
        """
        try:
            # 알려진 ETF 종목 코드 리스트 (일부)
            # 실제 운영 시에는 더 포괄적인 리스트나 API 조회로 대체 필요
            known_etf_codes = {
                # KODEX 시리즈
                '069500',  # KODEX 200
                '114800',  # KODEX 인버스
                '152100',  # KODEX 레버리지
                '229200',  # KODEX 코스닥150
                '305720',  # KODEX 2차전지산업
                '379800',  # KODEX 미국S&P500
                '487230',  # KODEX 미국AI전력핵심인프라
                
                # TIGER 시리즈
                '228790',  # TIGER 화장품
                '251350',  # TIGER 코스닥150
                '360750',  # TIGER 미국S&P500
                '364980',  # TIGER 2차전지TOP10
                '462010',  # TIGER 2차전지소재Fn
                
                # 기타 ETF
                '270300',  # KBSTAR 코스닥150
                '449450',  # PLUS K방산
                '466920',  # SOL 조선TOP3플러스
            }
            
            if code in known_etf_codes:
                logger.info(f"{code} 종목은 ETF로 식별됨")
                return True
            
            # 추가로 ETF일 가능성이 높은 패턴 확인 (보수적으로)
            # 3으로 시작하는 6자리 중 특정 범위만 ETF로 간주
            if code.startswith('3') and len(code) == 6:
                # 30xxxx, 31xxxx 대역 중 일부만 ETF (더 보수적)
                if code.startswith(('305', '306', '360', '364', '379')):
                    logger.info(f"{code} 종목은 ETF 패턴으로 식별됨")
                    return True
            
            # 4로 시작하는 일부 ETF 패턴 (매우 보수적)
            if code.startswith('4') and len(code) == 6:
                if code.startswith(('449', '462', '466', '487')):
                    # 하지만 462870(시프트업)처럼 일반 주식도 있으므로 개별 확인 필요
                    if code not in {'462870'}:  # 시프트업 제외
                        logger.info(f"{code} 종목은 ETF 패턴으로 식별됨")
                        return True
            
            return False
        except (IndexError, TypeError):
            return False
    
    def fetch_stock_current_price(self, code: str) -> Optional[Dict[str, Any]]:
        """
        주식 현재가 조회
        
        한국투자증권 API를 통해 특정 종목의 현재가 정보를 조회합니다.
        
        Args:
            code: 종목 코드
            
        Returns:
            dict or None: 주식 현재가 정보 또는 조회 실패 시 None
            
        Notes:
            모의투자와 실전투자 모두 지원하는 함수입니다.
            내부적으로 get_real_time_price_by_api() 함수를 호출하여 중복 코드를 방지합니다.
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # get_real_time_price_by_api() 함수를 호출하여 데이터 획득
        real_time_info = self.get_real_time_price_by_api(code)
        
        if not real_time_info:
            logger.error(f"{code} 현재가 조회 실패")
            return None
        
        # 현재가 정보 형식으로 변환하여 반환
        price_info = {
            "stock_code": real_time_info.get("stock_code", ""),
            "stock_name": real_time_info.get("stock_name", ""),
            "market": real_time_info.get("market", ""),
            "time": real_time_info.get("time", ""),
            "current_price": real_time_info.get("current_price", 0),
            "open_price": real_time_info.get("open_price", 0),
            "high_price": real_time_info.get("high_price", 0),
            "low_price": real_time_info.get("low_price", 0),
            "prev_close_price": real_time_info.get("prev_close_price", 0),
            "price_change": real_time_info.get("price_change", 0),
            "change_rate": real_time_info.get("change_rate", 0),
            "volume": real_time_info.get("volume", 0),
            "volume_value": real_time_info.get("volume_value", 0),
            "market_cap": real_time_info.get("market_cap", 0),
            "listed_shares": real_time_info.get("listed_shares", 0),
            "highest_52_week": real_time_info.get("highest_52_week", 0),
            "lowest_52_week": real_time_info.get("lowest_52_week", 0),
            "per": real_time_info.get("per", 0),
            "eps": real_time_info.get("eps", 0),
            "pbr": real_time_info.get("pbr", 0),
            "div_yield": real_time_info.get("div_yield", 0),
            "foreign_rate": real_time_info.get("foreign_rate", 0),
            "day_range_rate": real_time_info.get("day_range_rate", 0),
            "current_to_open_rate": real_time_info.get("current_to_open_rate", 0),
            "is_52week_high": real_time_info.get("is_52week_high", False),
            "is_52week_low": real_time_info.get("is_52week_low", False),
            "gap_from_52week_high": real_time_info.get("gap_from_52week_high", 0),
            "gap_from_52week_low": real_time_info.get("gap_from_52week_low", 0)
        }
        
        logger.info(f"{code} 현재가 조회 성공: {price_info['current_price']}원 ({price_info['change_rate']}%)")
        return price_info
    
    def get_real_time_price_by_api(self, code: str) -> Optional[Dict[str, Any]]:
        """
        실시간 시세 조회 API (국내주식 실시간호가 통합 API 사용)
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict or None: 실시간 시세 정보
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # 이 함수는 WebSocket 연결을 통해 실시간 데이터를 수신하는 것이 이상적입니다.
        # 여기서는 REST API로 현재가만 가져와 실시간에 가까운 정보를 반환합니다.
        
        # 현재가 조회 - 직접 API 호출
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            # ETF 여부에 따라 시장 구분 코드 설정
            is_etf = self.is_etf_stock(code)
            market_code = "E" if is_etf else "J"  # E: ETF, J: 주식
            
            path = "uapi/domestic-stock/v1/quotations/inquire-price"
            url = f"{URL_BASE}/{path}"
            
            params = {
                "fid_cond_mrkt_div_code": market_code,
                "fid_input_iscd": code,
            }
            
            headers = self._get_headers("FHKST01010100")
            
            logger.info(f"{code} 현재가 조회 요청 (시장구분: {market_code})")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            price_result = self._handle_response(res, f"{code} 현재가 조회 실패")
            
            # API 호출 실패 시 캐시된 데이터 사용
            if not price_result or price_result.get("rt_cd") != "0":
                # API 호출 실패 시 None 반환 (캐시 데이터 사용하지 않음)
                logger.warning(f"API를 통한 현재가 조회 실패, 매매를 건너뜁니다.")
                return None
            else:
                output = price_result.get("output", {})
                
                # 주요 정보 추출 및 가공
                price_info = {
                    "stock_code": code,
                    "stock_name": output.get("prdt_abrv_name", ""),
                    "market": output.get("rprs_mrkt_kor_name", ""),
                    "time": output.get("stck_basc_hour", ""),
                    "current_price": int(output.get("stck_prpr", "0").replace(',', '')),
                    "open_price": int(output.get("stck_oprc", "0").replace(',', '')),
                    "high_price": int(output.get("stck_hgpr", "0").replace(',', '')),
                    "low_price": int(output.get("stck_lwpr", "0").replace(',', '')),
                    "prev_close_price": int(output.get("stck_sdpr", "0").replace(',', '')),
                    "price_change": int(output.get("prdy_vrss", "0").replace(',', '')),
                    "change_rate": float(output.get("prdy_ctrt", "0")),
                    "volume": int(output.get("acml_vol", "0").replace(',', '')),
                    "volume_value": int(output.get("acml_tr_pbmn", "0").replace(',', '')),
                    "market_cap": int(output.get("hts_avls", "0").replace(',', '')),
                    "listed_shares": int(output.get("lstn_stcn", "0").replace(',', '')),
                    "highest_52_week": int(output.get("w52_hgpr", "0").replace(',', '')),
                    "lowest_52_week": int(output.get("w52_lwpr", "0").replace(',', '')),
                    "per": float(output.get("per", "0").replace(',', '')),
                    "eps": float(output.get("eps", "0").replace(',', '')),
                    "pbr": float(output.get("pbr", "0").replace(',', '')),
                    "div_yield": float(output.get("dvr", "0").replace(',', '')),
                    "foreign_rate": float(output.get("frgn_hldn_qty_rt", "0").replace(',', '')),
                    "day_range_rate": ((int(output.get("stck_hgpr", "0").replace(',', '')) - 
                                        int(output.get("stck_lwpr", "0").replace(',', ''))) / 
                                       int(output.get("stck_lwpr", "0").replace(',', '')) * 100) 
                                       if int(output.get("stck_lwpr", "0").replace(',', '')) > 0 else 0,
                    "current_to_open_rate": ((int(output.get("stck_prpr", "0").replace(',', '')) - 
                                             int(output.get("stck_oprc", "0").replace(',', ''))) / 
                                            int(output.get("stck_oprc", "0").replace(',', '')) * 100) 
                                            if int(output.get("stck_oprc", "0").replace(',', '')) > 0 else 0
                }
                
                # 추가 분석 정보
                price_info["is_52week_high"] = price_info["current_price"] >= price_info["highest_52_week"]
                price_info["is_52week_low"] = price_info["current_price"] <= price_info["lowest_52_week"]
                price_info["gap_from_52week_high"] = ((price_info["highest_52_week"] - price_info["current_price"]) / 
                                                     price_info["current_price"] * 100) if price_info["current_price"] > 0 else 0
                price_info["gap_from_52week_low"] = ((price_info["current_price"] - price_info["lowest_52_week"]) / 
                                                   price_info["lowest_52_week"] * 100) if price_info["lowest_52_week"] > 0 else 0
                
                # 데이터 캐싱 제거 - 실시간 데이터는 항상 최신 정보 사용
                logger.info(f"{code} 현재가 조회 성공: {price_info['current_price']}원 ({price_info['change_rate']}%)")
            
            # 현재가 정보 기반으로 실시간 정보 구성 (호가 정보 없이)
            real_time_info = {
                "stock_code": code,
                "stock_name": price_info.get("stock_name", ""),
                "current_price": price_info.get("current_price", 0),
                "change_rate": price_info.get("change_rate", 0),
                "volume": price_info.get("volume", 0),
                "time": price_info.get("time", ""),
                "bid_ask_ratio": 1.0,  # 기본값
                "highest_ask": 0,
                "lowest_bid": 0,
                "spread": 0,
                "total_ask_qty": 0,
                "total_bid_qty": 0,
                "asks": [],
                "bids": [],
                # 추가 정보도 포함
                "open_price": price_info.get("open_price", 0),
                "high_price": price_info.get("high_price", 0),
                "low_price": price_info.get("low_price", 0),
                "prev_close_price": price_info.get("prev_close_price", 0),
                "price_change": price_info.get("price_change", 0),
                "market_cap": price_info.get("market_cap", 0),
                "per": price_info.get("per", 0),
                "eps": price_info.get("eps", 0),
                "pbr": price_info.get("pbr", 0),
                "div_yield": price_info.get("div_yield", 0),
                "foreign_rate": price_info.get("foreign_rate", 0),
                "day_range_rate": price_info.get("day_range_rate", 0),
                "is_52week_high": price_info.get("is_52week_high", False),
                "is_52week_low": price_info.get("is_52week_low", False),
                "highest_52_week": price_info.get("highest_52_week", 0),
                "lowest_52_week": price_info.get("lowest_52_week", 0),
                "gap_from_52week_high": price_info.get("gap_from_52week_high", 0),
                "gap_from_52week_low": price_info.get("gap_from_52week_low", 0),
                "has_asking_price": False,  # 호가 정보 없음을 표시
                "has_valid_data": True,    # 현재가 정보만으로도 매매에 적합한 데이터로 간주
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 강화학습을 위한 추가 지표
            real_time_info["market_pressure"] = 1.0  # 호가 정보 없이 기본값 설정
            real_time_info["price_volatility"] = price_info.get("day_range_rate", 0)
            
            # 실시간 데이터 캐싱하지 않음 - 항상 최신 데이터 사용
            logger.info(f"{code} 실시간 시세 조회 성공: {real_time_info['current_price']}원")
            return real_time_info
        
        except Exception as e:
            logger.error(f"{code} 실시간 시세 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 실시간 시세 조회 실패: {e}")
            
            # 예외 상황에서는 항상 None 반환 - 캐시된 데이터 사용하지 않음
            return None 