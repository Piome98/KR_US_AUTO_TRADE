"""
한국 주식 자동매매 - 시장 가격 조회 모듈
시장 가격 조회 및 실시간 가격 정보 관련 기능 제공
"""

import logging
import os
import json
import datetime
import requests
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
    
    def get_stock_asking_price(self, code: str) -> Optional[Dict[str, Any]]:
        """
        주식 현재가 호가 조회
        
        한국투자증권 API를 통해 특정 종목의 호가 정보를 조회합니다.
        
        Args:
            code: 종목 코드
            
        Returns:
            dict or None: 주식 호가 정보 또는 조회 실패 시 None
            
        Notes:
            모의투자와 실전투자 모두 지원하는 함수입니다.
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/quotations/inquire-asking-price"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "fid_cond_mrkt_div_code": "J",  # 시장 구분 코드: J-주식
            "fid_input_iscd": code,         # 종목 코드
        }
        
        try:
            headers = self._get_headers("FHKST01010200")
            
            logger.info(f"{code} 호가 조회 요청")
            result = self._request_get(url, headers, params, f"{code} 호가 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                # API 호출 실패 시 캐시된 데이터 사용
                logger.warning(f"API를 통한 호가 조회 실패, 캐시된 데이터 확인")
                cache_file = self._get_cache_path(f'asking_price_{code}.json')
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                            if cached_data and isinstance(cached_data, dict):
                                cache_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
                                current_time = datetime.datetime.now()
                                # 호가 데이터는 빠르게 변하므로 30분 이내인 경우에만 사용
                                if (current_time - cache_time).total_seconds() < 1800:
                                    logger.info(f"캐시된 호가 데이터 사용 (캐시 시간: {cache_time})")
                                    return cached_data
                                else:
                                    logger.warning(f"캐시된 데이터가 오래됨 (캐시 시간: {cache_time})")
                    except Exception as e:
                        logger.error(f"캐시 파일 읽기 실패: {e}")
                return None
                
            # 매도/매수 호가 정보 추출
            output = result.get("output", {})
            
            # 10단계 호가 정보 정리
            asking_prices = {
                "stock_code": code,
                "time": output.get("stck_bsop_hour", ""),
                "total_ask_qty": int(output.get("total_askp_rsqn", "0").replace(',', '')),
                "total_bid_qty": int(output.get("total_bidp_rsqn", "0").replace(',', '')),
                "asks": [],  # 매도호가
                "bids": [],  # 매수호가
                "ask_prices": [],  # 매도호가 가격만
                "bid_prices": [],  # 매수호가 가격만
                "ask_quantities": [],  # 매도호가 수량만
                "bid_quantities": []   # 매수호가 수량만
            }
            
            # 매도호가 정보 추출 (1~10)
            for i in range(1, 11):
                ask_price = int(output.get(f"askp{i}", "0").replace(',', ''))
                ask_qty = int(output.get(f"askp_rsqn{i}", "0").replace(',', ''))
                
                asking_prices["asks"].append({"price": ask_price, "quantity": ask_qty})
                asking_prices["ask_prices"].append(ask_price)
                asking_prices["ask_quantities"].append(ask_qty)
            
            # 매수호가 정보 추출 (1~10)
            for i in range(1, 11):
                bid_price = int(output.get(f"bidp{i}", "0").replace(',', ''))
                bid_qty = int(output.get(f"bidp_rsqn{i}", "0").replace(',', ''))
                
                asking_prices["bids"].append({"price": bid_price, "quantity": bid_qty})
                asking_prices["bid_prices"].append(bid_price)
                asking_prices["bid_quantities"].append(bid_qty)
            
            # 최고/최저 호가
            asking_prices["highest_ask"] = asking_prices["ask_prices"][0]
            asking_prices["lowest_bid"] = asking_prices["bid_prices"][0]
            
            # 호가 스프레드
            asking_prices["spread"] = asking_prices["highest_ask"] - asking_prices["lowest_bid"]
            
            # 시장 압력 지표 계산 (매수세/매도세 지표)
            asking_prices["bid_ask_ratio"] = (sum(asking_prices["bid_quantities"]) / sum(asking_prices["ask_quantities"])) if sum(asking_prices["ask_quantities"]) > 0 else 0
            
            # 데이터 캐싱
            try:
                cache_file = self._get_cache_path(f'asking_price_{code}.json')
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(asking_prices, f)
                logger.info(f"{code} 호가 데이터 캐싱 완료")
            except Exception as e:
                logger.error(f"{code} 호가 데이터 캐싱 실패: {e}")
            
            logger.info(f"{code} 호가 조회 성공")
            return asking_prices
            
        except Exception as e:
            logger.error(f"{code} 호가 조회 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] {code} 호가 조회 실패: {e}")
            
            # 예외 발생 시 캐시된 데이터 사용
            logger.warning(f"API 조회 중 예외 발생, 캐시된 데이터 확인")
            cache_file = self._get_cache_path(f'asking_price_{code}.json')
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        if cached_data and isinstance(cached_data, dict):
                            cache_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
                            current_time = datetime.datetime.now()
                            if (current_time - cache_time).total_seconds() < 3600:  # 1시간 이내
                                logger.info(f"캐시된 호가 데이터 사용 (캐시 시간: {cache_time})")
                                return cached_data
                except Exception as e:
                    logger.error(f"캐시 파일 읽기 실패: {e}")
            return None
    
    def _get_cache_path(self, filename: str) -> str:
        """
        캐시 파일 경로 반환
        
        Args:
            filename: 캐시 파일명
            
        Returns:
            str: 캐시 파일의 전체 경로
        """
        return os.path.join(os.path.dirname(__file__), '../../../../data/cache', filename)
    
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
        # 여기서는 REST API로 현재가와 호가 정보를 함께 가져와 실시간에 가까운 정보를 반환합니다.
        
        # 현재가 조회 - 직접 API 호출
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            path = "uapi/domestic-stock/v1/quotations/inquire-price"
            url = f"{URL_BASE}/{path}"
            
            params = {
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": code,
            }
            
            headers = self._get_headers("FHKST01010100")
            
            logger.info(f"{code} 현재가 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            price_result = self._handle_response(res, f"{code} 현재가 조회 실패")
            
            if not price_result or price_result.get("rt_cd") != "0":
                # API 호출 실패 시 캐시된 데이터 사용
                logger.warning(f"API를 통한 현재가 조회 실패, 캐시된 데이터 확인")
                cache_file = os.path.join(os.path.dirname(__file__), f'../../../../data/cache/price_{code}.json')
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            price_info = json.load(f)
                            cache_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
                            current_time = datetime.datetime.now()
                            if (current_time - cache_time).total_seconds() < 3600:
                                logger.info(f"캐시된 현재가 데이터 사용 (캐시 시간: {cache_time})")
                            else:
                                logger.warning(f"캐시된 데이터가 오래됨 (캐시 시간: {cache_time})")
                    except Exception as e:
                        logger.error(f"캐시 파일 읽기 실패: {e}")
                        price_info = None
                else:
                    price_info = None
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
                
                # 데이터 캐싱
                try:
                    cache_dir = os.path.join(os.path.dirname(__file__), '../../../../data/cache')
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_file = os.path.join(cache_dir, f'price_{code}.json')
                    with open(cache_file, 'w') as f:
                        json.dump(price_info, f)
                    logger.info(f"{code} 현재가 데이터 캐싱 완료: {cache_file}")
                except Exception as e:
                    logger.error(f"{code} 현재가 데이터 캐싱 실패: {e}")
                
                logger.info(f"{code} 현재가 조회 성공: {price_info['current_price']}원 ({price_info['change_rate']}%)")
            
            if price_info is None:
                logger.warning(f"{code} 실시간 시세 조회 실패: 현재가 조회 실패")
                return None
            
            # 호가 조회
            asking_price = self.get_stock_asking_price(code)
            if asking_price is None:
                logger.warning(f"{code} 실시간 시세 조회 실패: 호가 조회 실패")
                # 현재가라도 있으면 반환
                return price_info
            
            # 현재가와 호가 정보 통합
            real_time_info = {
                "stock_code": code,
                "stock_name": price_info.get("stock_name", ""),
                "current_price": price_info.get("current_price", 0),
                "change_rate": price_info.get("change_rate", 0),
                "volume": price_info.get("volume", 0),
                "time": asking_price.get("time", ""),
                "bid_ask_ratio": asking_price.get("bid_ask_ratio", 0),
                "highest_ask": asking_price.get("highest_ask", 0),
                "lowest_bid": asking_price.get("lowest_bid", 0),
                "spread": asking_price.get("spread", 0),
                "total_ask_qty": asking_price.get("total_ask_qty", 0),
                "total_bid_qty": asking_price.get("total_bid_qty", 0),
                "asks": asking_price.get("asks", []),
                "bids": asking_price.get("bids", []),
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
                "gap_from_52week_low": price_info.get("gap_from_52week_low", 0)
            }
            
            # 강화학습을 위한 추가 지표
            real_time_info["market_pressure"] = real_time_info["bid_ask_ratio"] if real_time_info["bid_ask_ratio"] > 0 else 0
            real_time_info["price_volatility"] = price_info.get("day_range_rate", 0)
            
            # 데이터 캐싱
            try:
                cache_dir = os.path.join(os.path.dirname(__file__), '../../../../data/cache')
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, f'realtime_{code}.json')
                with open(cache_file, 'w') as f:
                    json.dump(real_time_info, f)
                logger.info(f"{code} 실시간 시세 데이터 캐싱 완료: {cache_file}")
            except Exception as e:
                logger.error(f"{code} 실시간 시세 데이터 캐싱 실패: {e}")
            
            logger.info(f"{code} 실시간 시세 조회 성공: {real_time_info['current_price']}원, 매수/매도비율: {real_time_info['bid_ask_ratio']:.2f}")
            return real_time_info
        
        except Exception as e:
            logger.error(f"{code} 실시간 시세 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 실시간 시세 조회 실패: {e}")
            
            # 예외 발생 시 캐시된 데이터 사용
            logger.warning(f"API 조회 중 예외 발생, 캐시된 데이터 확인")
            cache_file = os.path.join(os.path.dirname(__file__), f'../../../../data/cache/realtime_{code}.json')
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        cache_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
                        current_time = datetime.datetime.now()
                        # 1시간 이내인 경우에만 사용
                        if (current_time - cache_time).total_seconds() < 3600:
                            logger.info(f"캐시된 실시간 데이터 사용 (캐시 시간: {cache_time})")
                            return cached_data
                except Exception as e:
                    logger.error(f"캐시 파일 읽기 실패: {e}")
            
            return None 