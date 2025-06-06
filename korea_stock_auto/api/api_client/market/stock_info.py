"""
한국 주식 자동매매 - 주식 기본 정보 조회 모듈
종목 기본 정보 및 거래량 상위 종목 조회 기능 제공
"""

import logging
import os
import json
import datetime
import requests
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING, cast
from pathlib import Path

from korea_stock_auto.config import get_config
from korea_stock_auto.utils.utils import send_message

# 타입 힌트만을 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

# 로깅 설정
logger = logging.getLogger("stock_auto")

class StockInfoMixin:
    """
    주식 기본 정보 조회 관련 기능 Mixin
    
    종목 기본 정보, 거래량 상위 종목 등 주식 정보 조회 기능을 제공합니다.
    """
    
    def get_stock_info(self, code: str) -> Optional[Dict[str, Any]]:
        """
        종목 기본 정보 조회
        
        한국투자증권 API를 통해 특정 종목의 기본 정보를 조회합니다.
        
        Args:
            code: 종목 코드
            
        Returns:
            dict or None: 종목 기본 정보 또는 조회 실패 시 None
            
        Notes:
            모의투자와 실전투자 모두 지원하는 함수입니다.
        """

        # 설정 로드
        config = get_config()
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/quotations/search-info"
        url = f"{config.current_api.base_url}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # 시장 구분 코드: J-주식
            "FID_INPUT_ISCD": code,         # 종목 코드
            "PDNO": code,                   # 종목 코드 (필수 파라미터)
            "PRDT_TYPE_CD": "300"           # 상품 유형 코드 (필수 파라미터) - 주식:300
        }
        
        try:
            headers = self._get_headers("CTPF1002R")
            
            logger.info(f"{code} 종목 기본 정보 조회 요청")
            result = self._request_get(url, headers, params, f"{code} 종목 정보 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                logger.error(f"{code} 종목 정보 조회 결과가 유효하지 않습니다.")
                return None
                
            output = result.get("output", {})
            if not output:
                logger.warning(f"{code} 종목 정보가 없습니다.")
                return None
            
            # 결과 가공
            stock_info = {
                "code": code,
                "name": output.get("hts_kor_isnm", ""),
                "market": output.get("bstp_kor_isnm", ""),
                "sector": output.get("bstp_larg_div_name", ""),
                "industry": output.get("bstp_med_div_name", ""),
                "listed_shares": int(output.get("lstn_stcn", "0").replace(',', '')),
                "capital": int(output.get("cpfn", "0").replace(',', '')),
                "par_value": int(output.get("stck_fcam", "0").replace(',', '')),
                "foreign_rate": float(output.get("frgn_hldn_qty_rt", "0")),
                "per": float(output.get("per", "0")),
                "eps": float(output.get("eps", "0").replace(',', '')),
                "pbr": float(output.get("pbr", "0")),
                "bps": float(output.get("bps", "0").replace(',', ''))
            }
            
            logger.info(f"{code} 종목 기본 정보 조회 성공")
            return stock_info
            
        except Exception as e:
            logger.error(f"{code} 종목 정보 조회 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] {code} 종목 정보 조회 실패: {e}", config.notification.discord_webhook_url)
            return None
    
    def get_top_traded_stocks(self, market_type: str = "0", top_n: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        거래량 상위 종목 조회 (국내주식-047 API 활용)
        
        Args:
            market_type (str): 시장 구분 (0:전체, 1:코스피, 2:코스닥)
            top_n (int): 조회할 종목 수 (최대 100)
            
        Returns:
            list or None: 거래량 상위 종목 목록
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # 설정 가져오기 (self.config가 있으면 사용, 없으면 get_config() 사용)
        config = getattr(self, 'config', None) or get_config()
        
        path = "uapi/domestic-stock/v1/quotations/volume-rank"
        url = f"{config.current_api.base_url}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # 시장 구분 코드 "J"(주식)으로 고정
            "FID_COND_SCR_DIV_CODE": "20171",  # 거래량 순위 조회 화면번호
            "FID_INPUT_ISCD": "0000", 
            "FID_DIV_CLS_CODE": "0",  
            "FID_BLNG_CLS_CODE": "0",
            "FID_TRGT_CLS_CODE": "1", 
            "FID_TRGT_EXLS_CLS_CODE": "0000000000",
            "FID_INPUT_PRICE_1": "0",
            "FID_INPUT_PRICE_2": "999999999",
            "FID_VOL_CNT": "1000000",
            "FID_INPUT_DATE_1": ""
        }
        
        # API 요청 헤더 생성
        headers = self._get_headers("FHPST01710000")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            market_name = {
                "0": "전체", 
                "1": "코스피", 
                "2": "코스닥"
            }.get(market_type, "전체")
            
            logger.debug(f"{market_name} 거래량 상위 {top_n}개 종목 조회 요청")
            # API 오류를 더 자세히 확인하기 위해 직접 요청 처리
            res = requests.get(url, headers=headers, params=params, timeout=10)
            if res.status_code != 200:
                logger.error(f"거래량 상위 종목 조회 API 오류: HTTP {res.status_code}")
                logger.error(f"응답: {res.text[:200]}...")
                return None
                
            result = res.json()
            if result.get("rt_cd") != "0":
                error_code = result.get("msg_cd", "")
                error_msg = result.get("msg1", "")
                logger.error(f"거래량 상위 종목 조회 API 오류: {error_code} - {error_msg}")
                send_message(f"[오류] 거래량 상위 종목 조회 실패: {error_msg}", config.notification.discord_webhook_url)
                return None
                
            # API 응답 구조 확인 및 처리
            output = result.get("output")
            logger.debug(f"거래량 상위 종목 API 응답 output 타입: {type(output)}")
            
            # 응답 구조도 로깅 (문제 해결을 위해)
            if isinstance(output, list) and len(output) > 0:
                logger.debug(f"첫 번째 항목 키: {list(output[0].keys())}")
            elif isinstance(output, dict):
                logger.debug(f"output 키: {list(output.keys())}")
                
            # output이 없거나 빈 리스트인 경우
            if not output:
                logger.warning("거래량 상위 종목이 없습니다.")
                return []
            
            # 결과 가공
            ranked_stocks = []
            
            # 리스트 형태의 응답 처리 (일반적인 경우)
            if isinstance(output, list):
                for stock in output:
                    try:
                        if not isinstance(stock, dict):
                            logger.debug(f"종목 데이터가 딕셔너리가 아닙니다: {type(stock)}")
                            continue
                            
                        # 키 이름이 다를 수 있으므로 확인하고 추출
                        # 종목코드 필드 찾기 (mksc_shrn_iscd 또는 유사한 필드)
                        code_key = next((k for k in stock.keys() if 'iscd' in k.lower() or 'code' in k.lower()), None)
                        name_key = next((k for k in stock.keys() if 'isnm' in k.lower() or 'name' in k.lower()), None)
                        price_key = next((k for k in stock.keys() if 'prpr' in k.lower() or 'price' in k.lower()), None)
                        change_key = next((k for k in stock.keys() if 'ctrt' in k.lower() or 'rate' in k.lower()), None)
                        volume_key = next((k for k in stock.keys() if 'vol' in k.lower() or 'volume' in k.lower()), None)
                        market_cap_key = next((k for k in stock.keys() if 'avls' in k.lower() or 'cap' in k.lower()), None)
                        
                        # 키를 찾지 못한 경우 기본값 설정
                        if not code_key:
                            logger.debug(f"종목코드 필드를 찾을 수 없습니다: {list(stock.keys())}")
                            code_key = "mksc_shrn_iscd"
                        
                        stock_info = {
                            "rank": stock.get("no", "0"),
                            "code": stock.get(code_key, ""),
                            "name": stock.get(name_key, "N/A") if name_key else "N/A",
                            "price": int(stock.get(price_key, "0").replace(',', '')) if price_key else 0,
                            "change_rate": float(stock.get(change_key, "0").replace('%', '')) if change_key else 0.0,
                            "volume": int(stock.get(volume_key, "0").replace(',', '')) if volume_key else 0,
                            "market_cap": int(stock.get(market_cap_key, "0").replace(',', '')) if market_cap_key else 0
                        }
                        
                        # 코드가 있는 경우만 추가
                        if stock_info["code"]:
                            ranked_stocks.append(stock_info)
                    except (ValueError, TypeError) as e:
                        logger.debug(f"종목 데이터 파싱 오류: {e} - {stock}")
                        continue
            # 딕셔너리 형태의 응답 처리 (일부 상황)
            elif isinstance(output, dict):
                try:
                    # 응답 데이터가 리스트가 아닌 딕셔너리 형태로 제공될 수 있음
                    # 1. "data" 등의 키 안에 리스트가 있는 경우
                    for key, value in output.items():
                        if isinstance(value, list) and len(value) > 0:
                            # 리스트를 발견하면 재귀적으로 처리
                            for item in value:
                                if isinstance(item, dict):
                                    code_key = next((k for k in item.keys() if 'iscd' in k.lower() or 'code' in k.lower()), None)
                                    if code_key:
                                        ranked_stocks.append({
                                            "rank": item.get("no", "0"),
                                            "code": item.get(code_key, ""),
                                            "name": item.get("hts_kor_isnm", "N/A"),
                                            "price": int(item.get("stck_prpr", "0").replace(',', '')),
                                            "change_rate": float(item.get("prdy_ctrt", "0").replace('%', '')),
                                            "volume": int(item.get("acml_vol", "0").replace(',', '')),
                                            "market_cap": int(item.get("hts_avls", "0").replace(',', ''))
                                        })
                            
                    # 2. 단일 종목 정보가 담긴 딕셔너리인 경우
                    code_key = next((k for k in output.keys() if 'iscd' in k.lower() or 'code' in k.lower()), None)
                    if code_key:
                        stock_info = {
                            "rank": output.get("no", "0"),
                            "code": output.get(code_key, ""),
                            "name": output.get("hts_kor_isnm", "N/A"),
                            "price": int(output.get("stck_prpr", "0").replace(',', '')),
                            "change_rate": float(output.get("prdy_ctrt", "0").replace('%', '')),
                            "volume": int(output.get("acml_vol", "0").replace(',', '')),
                            "market_cap": int(output.get("hts_avls", "0").replace(',', ''))
                        }
                        ranked_stocks.append(stock_info)
                except (ValueError, TypeError) as e:
                    logger.debug(f"종목 데이터 파싱 오류: {e} - {output}")
            else:
                logger.warning(f"예상치 못한 output 형식: {type(output)}")
            
            # 결과가 비어있는 경우 백업 방법 시도
            if not ranked_stocks and output:
                logger.info("기본 파싱 실패, 대체 방식 시도")
                try:
                    if isinstance(output, list):
                        for item in output:
                            if isinstance(item, dict):
                                # 모든 가능한 키와 값 로깅
                                for k, v in item.items():
                                    logger.debug(f"키: {k}, 값: {v}")
                                
                                # 첫 번째 항목의 모든 필드를 이름과 함께 저장
                                stock_info = {
                                    "rank": "0",
                                    "code": "",
                                    "name": "N/A",
                                    "price": 0,
                                    "change_rate": 0.0,
                                    "volume": 0,
                                    "market_cap": 0
                                }
                                
                                # 모든 필드 중에서 종목코드 찾기
                                for k, v in item.items():
                                    if isinstance(v, str) and len(v) >= 6 and v.isalnum():
                                        stock_info["code"] = v
                                        break
                                
                                if stock_info["code"]:
                                    ranked_stocks.append(stock_info)
                except Exception as e:
                    logger.debug(f"대체 파싱 방식 오류: {e}")
            
            # 결과 캐싱
            try:
                cache_file = self._get_cache_path(f'volume_rank_{market_type}.json')
                cache_dir = os.path.dirname(cache_file)
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(ranked_stocks, f)
                logger.debug(f"거래량 상위 종목 데이터 캐싱 완료")
            except Exception as e:
                logger.debug(f"거래량 상위 종목 데이터 캐싱 실패: {e}")
            
            logger.info(f"{market_name} 거래량 상위 {len(ranked_stocks)}개 종목 조회 성공")
            return ranked_stocks
            
        except Exception as e:
            logger.error(f"거래량 상위 종목 조회 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] 거래량 상위 종목 조회 실패: {e}", self.config.notification.discord_webhook_url)
            
            # 예외 발생 시 캐시된 데이터 사용
            logger.warning(f"API 조회 중 예외 발생, 캐시된 데이터 확인")
            cache_file = self._get_cache_path(f'volume_rank_{market_type}.json')
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        if cached_data and isinstance(cached_data, list):
                            cache_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
                            current_time = datetime.datetime.now()
                            if (current_time - cache_time).total_seconds() < 86400:  # 24시간 이내
                                logger.info(f"캐시된 거래량 상위 종목 데이터 사용 (캐시 시간: {cache_time})")
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
        cache_path = os.path.join(os.path.dirname(__file__), '../../../../data/cache', filename)
        # 캐시 디렉토리가 없으면 생성
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_path
    
    def get_volume_increasing_stocks(self, market_type: str = "0", top_n: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        거래량 급증 종목 조회 (국내주식-047 API 활용)
        
        Args:
            market_type (str): 시장 구분 (0:전체, 1:코스피, 2:코스닥)
            top_n (int): 조회할 종목 수 (최대 100)
            
        Returns:
            list or None: 선정된 거래량 급증 종목 목록
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # 설정 가져오기 (self.config가 있으면 사용, 없으면 get_config() 사용)
        config = getattr(self, 'config', None) or get_config()
        
        path = "uapi/domestic-stock/v1/quotations/volume-rank"
        url = f"{config.current_api.base_url}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # 시장 구분 코드 "J"(주식)으로 고정
            "FID_COND_SCR_DIV_CODE": "20171",  # 거래량 순위 조회 화면번호
            "FID_INPUT_ISCD": "0000", 
            "FID_DIV_CLS_CODE": "2",  # 0: 거래량, 1: 거래대금, 2: 거래량 급증, 3: 거래대금 급증
            "FID_BLNG_CLS_CODE": "0",
            "FID_TRGT_CLS_CODE": "1",
            "FID_TRGT_EXLS_CLS_CODE": "0000000000",
            "FID_INPUT_PRICE_1": "0",
            "FID_INPUT_PRICE_2": "999999999",
            "FID_VOL_CNT": "1000000",
            "FID_INPUT_DATE_1": ""
        }
        
        # API 요청 헤더 생성
        headers = self._get_headers("FHPST01710000")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            market_name = {
                "0": "전체", 
                "1": "코스피", 
                "2": "코스닥"
            }.get(market_type, "전체")
            
            logger.debug(f"{market_name} 거래량 급증 {top_n}개 종목 조회 요청")
            # API 오류를 더 자세히 확인하기 위해 직접 요청 처리
            res = requests.get(url, headers=headers, params=params, timeout=10)
            if res.status_code != 200:
                logger.error(f"거래량 급증 종목 조회 API 오류: HTTP {res.status_code}")
                logger.error(f"응답: {res.text[:200]}...")
                return None
                
            result = res.json()
            if result.get("rt_cd") != "0":
                error_code = result.get("msg_cd", "")
                error_msg = result.get("msg1", "")
                logger.error(f"거래량 급증 종목 조회 API 오류: {error_code} - {error_msg}")
                send_message(f"[오류] 거래량 급증 종목 조회 실패: {error_msg}", config.notification.discord_webhook_url)
                return None
                
            # API 응답 구조 확인 및 처리
            output = result.get("output")
            logger.debug(f"거래량 급증 종목 API 응답 output 타입: {type(output)}")
            
            if not output:
                logger.warning("거래량 급증 종목이 없습니다.")
                return []
            
            # 결과 가공
            ranked_stocks = []
            
            # 리스트 형태의 응답 처리 (일반적인 경우)
            if isinstance(output, list):
                for stock in output:
                    try:
                        if not isinstance(stock, dict):
                            logger.warning(f"종목 데이터가 딕셔너리가 아닙니다: {type(stock)}")
                            continue
                            
                        stock_info = {
                            "rank": stock.get("no", "0"),
                            "code": stock.get("mksc_shrn_iscd", ""),
                            "name": stock.get("hts_kor_isnm", ""),
                            "price": int(stock.get("stck_prpr", "0").replace(',', '')),
                            "change_rate": float(stock.get("prdy_ctrt", "0").replace('%', '')),
                            "volume": int(stock.get("acml_vol", "0").replace(',', '')),
                            "volume_ratio": float(stock.get("vol_inrt", "0").replace('%', '')),  # 거래량 증가율
                            "market_cap": int(stock.get("hts_avls", "0").replace(',', ''))
                        }
                        ranked_stocks.append(stock_info)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"종목 데이터 파싱 오류: {e} - {stock}")
                        continue
            # 딕셔너리 형태의 응답 처리 (일부 상황)
            elif isinstance(output, dict):
                try:
                    # 단일 종목 정보가 담긴 딕셔너리인 경우
                    if "mksc_shrn_iscd" in output:
                        stock_info = {
                            "rank": output.get("no", "0"),
                            "code": output.get("mksc_shrn_iscd", ""),
                            "name": output.get("hts_kor_isnm", ""),
                            "price": int(output.get("stck_prpr", "0").replace(',', '')),
                            "change_rate": float(output.get("prdy_ctrt", "0").replace('%', '')),
                            "volume": int(output.get("acml_vol", "0").replace(',', '')),
                            "volume_ratio": float(output.get("vol_inrt", "0").replace('%', '')),  # 거래량 증가율
                            "market_cap": int(output.get("hts_avls", "0").replace(',', ''))
                        }
                        ranked_stocks.append(stock_info)
                    else:
                        logger.warning(f"거래량 급증 종목 데이터 구조 예상과 다름: {output.keys()}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"종목 데이터 파싱 오류: {e} - {output}")
            else:
                logger.warning(f"예상치 못한 output 형식: {type(output)}")
            
            # 결과 캐싱
            try:
                cache_file = self._get_cache_path(f'volume_increasing_{market_type}.json')
                cache_dir = os.path.dirname(cache_file)
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(ranked_stocks, f)
                logger.debug(f"거래량 급증 종목 데이터 캐싱 완료")
            except Exception as e:
                logger.debug(f"거래량 급증 종목 데이터 캐싱 실패: {e}")
            
            logger.info(f"{market_name} 거래량 급증 {len(ranked_stocks)}개 종목 조회 성공")
            return ranked_stocks
            
        except Exception as e:
            logger.error(f"거래량 급증 종목 조회 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] 거래량 급증 종목 조회 실패: {e}", self.config.notification.discord_webhook_url)
            return None
    
    def get_investor_trends(self, code: str) -> Optional[Dict[str, Any]]:
        """
        주식 현재가 투자자별 매매현황 조회
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict or None: 투자자별 매매현황
            
        Notes:
            모의투자 지원 함수입니다.
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/quotations/inquire-investor"
        url = f"{self.config.current_api.base_url}/{path}"
        
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": code,
        }
        
        headers = self._get_headers("FHKST01010900")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} 투자자별 매매현황 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} 투자자별 매매현황 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            # 투자자별 매매현황 추출
            output1 = result.get("output1")
            output2 = result.get("output2")
            
            # 응답 구조 로깅
            logger.info(f"{code} 투자자별 매매현황 output1 타입: {type(output1)}, output2 타입: {type(output2)}")
            
            # output1 처리 (종목 기본 정보)
            stock_basic_info = {}
            if isinstance(output1, dict):
                stock_basic_info = output1
            elif isinstance(output1, list) and len(output1) > 0:
                stock_basic_info = output1[0]
            else:
                logger.warning(f"{code} 투자자별 매매현황 - 종목 기본 정보가 없습니다.")
                return None
                
            # output2 처리 (투자자별 매매현황)
            if not output2:
                logger.warning(f"{code} 투자자별 매매현황 데이터가 없습니다.")
                return None
                
            investor_data_list = []
            if isinstance(output2, list):
                investor_data_list = output2
            elif isinstance(output2, dict):
                # 단일 투자자 정보일 경우 리스트로 변환
                if "bying_sell_invst_tp_nm" in output2:
                    investor_data_list = [output2]
                else:
                    logger.warning(f"{code} 투자자별 매매현황 - 예상과 다른 응답 구조: {output2.keys()}")
                    return None
            else:
                logger.warning(f"{code} 투자자별 매매현황 - 예상과 다른 데이터 형식: {type(output2)}")
                return None
            
            # 종목 기본 정보
            investor_info = {
                "stock_code": code,
                "stock_name": stock_basic_info.get("hts_kor_isnm", ""),
                "current_price": int(stock_basic_info.get("stck_prpr", "0").replace(',', '')),
                "change_rate": float(stock_basic_info.get("prdy_ctrt", "0").replace(',', '')),
                "volume": int(stock_basic_info.get("acml_vol", "0").replace(',', '')),
                "investors": {}
            }
            
            # 투자자별 매매현황 - 데이터 정의
            investor_types = {
                "korea_institution": "금융투자", 
                "insurance": "보험", 
                "investment": "투신",
                "private_equity": "사모펀드", 
                "bank": "은행", 
                "pension": "연기금",
                "korea_general": "기타법인", 
                "individual": "개인", 
                "foreign": "외국인",
                "national": "국가", 
                "etc": "기타외국인"
            }
            
            # 투자자별 매매현황 데이터 추출
            for item in investor_data_list:
                investor_type = item.get("bying_sell_invst_tp_nm", "")
                if investor_type in investor_types.values():
                    # 영문 키로 변환
                    key = [k for k, v in investor_types.items() if v == investor_type][0]
                    
                    investor_data = {
                        "name": investor_type,
                        "today_volume": int(item.get("tddy_cprs_icdc_qty", "0").replace(',', '')),
                        "yesterday_volume": int(item.get("yndy_cmpr_icdc_qty", "0").replace(',', '')),
                        "today_amount": int(item.get("tddy_acrq_icdc_amt", "0").replace(',', '')),
                        "yesterday_amount": int(item.get("yndy_cmpr_icdc_amt", "0").replace(',', ''))
                    }
                    
                    # 순매수(+) 또는 순매도(-) 여부
                    investor_data["is_net_buying"] = investor_data["today_volume"] > 0
                    
                    investor_info["investors"][key] = investor_data
            
            # 주요 매매주체 분석
            # 개인, 외국인, 기관(금융투자+보험+투신+...) 순매수 종합
            foreign_data = investor_info["investors"].get("foreign", {"today_volume": 0})
            individual_data = investor_info["investors"].get("individual", {"today_volume": 0})
            
            # 기관 순매수 합산 
            institution_volume = sum(
                investor_info["investors"].get(key, {"today_volume": 0})["today_volume"]
                for key in ["korea_institution", "insurance", "investment", 
                           "private_equity", "bank", "pension"]
            )
            
            # 주요 매매주체 요약 정보
            investor_info["summary"] = {
                "foreign_volume": foreign_data.get("today_volume", 0),
                "individual_volume": individual_data.get("today_volume", 0),
                "institution_volume": institution_volume,
                "main_player": "foreign" if abs(foreign_data.get("today_volume", 0)) > max(abs(individual_data.get("today_volume", 0)), abs(institution_volume)) else
                               "individual" if abs(individual_data.get("today_volume", 0)) > abs(institution_volume) else
                               "institution"
            }
            
            logger.info(f"{code} 투자자별 매매현황 조회 성공")
            return investor_info
            
        except Exception as e:
            logger.error(f"{code} 투자자별 매매현황 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 투자자별 매매현황 조회 실패: {e}", self.config.notification.discord_webhook_url)
            return None 