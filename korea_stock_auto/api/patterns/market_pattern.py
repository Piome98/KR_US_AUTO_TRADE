"""
한국 주식 자동매매 - 시장 데이터 서비스

시장 가격, 차트, 종목 정보 조회 등 시장 데이터 관련 기능을 제공합니다.
"""

import logging
import json
import requests
import time
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Union
from datetime import datetime, timedelta

from korea_stock_auto.config import get_config
from korea_stock_auto.utils.utils import send_message

# 도메인 엔터티 import
from korea_stock_auto.domain import Stock, Price, Money

# API 매퍼 통합
from korea_stock_auto.api.mappers import StockMapper, MappingError

if TYPE_CHECKING:
    from korea_stock_auto.api.client import KoreaInvestmentApiClient

logger = logging.getLogger("stock_auto")

class MarketService:
    """
    시장 데이터 조회 서비스 (API 매퍼 통합)
    
    주식 현재가, 호가, 차트 데이터, 종목 정보 등 시장 데이터 관련 조회 기능을 제공합니다.
    """
    
    def __init__(self, api_client: 'KoreaInvestmentApiClient'):
        """
        MarketService 초기화
        
        Args:
            api_client: KoreaInvestmentApiClient 인스턴스
        """
        self.api_client = api_client
        self.config = get_config()
        
        # StockMapper 초기화
        self.stock_mapper = StockMapper(
            enable_cache=True,
            cache_ttl_seconds=30  # 시세 데이터는 30초 캐시
        )
        
        logger.debug("MarketService 초기화 완료 (StockMapper 통합)")
    
    def is_etf_stock(self, code: str) -> bool:
        """
        ETF 종목 여부 확인
        
        Args:
            code: 종목코드
            
        Returns:
            bool: ETF 종목 여부
        """
        if not code:
            return False
            
        # ETF 종목코드 특성에 따른 판별
        # 한국 ETF는 주로 특정 패턴을 가짐
        etf_patterns = [
            # KODEX 시리즈 (대부분 1로 시작)
            "122", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
            "140", "141", "142", "143", "144", "145", "146", "147", "148", "149",
            "150", "151", "152", "153", "154", "155", "156", "157", "158", "159",
            # TIGER 시리즈
            "102", "103", "104", "105", "106", "107", "108", "109",
            "210", "211", "212", "213", "214", "215", "216", "217", "218", "219",
            "220", "221", "222", "223", "224", "225", "226", "227", "228", "229",
            # KBSTAR 시리즈
            "091", "092", "093", "094", "095", "096", "097", "098", "099",
            # HANARO 시리즈  
            "069", "070", "071", "072", "073", "074", "075", "076", "077", "078",
            # PLUS 시리즈
            "252", "253", "254", "255", "256", "257", "258", "259",
            # ARIRANG 시리즈
            "161", "162", "163", "164", "165", "166", "167", "168", "169"
        ]
        
        # 종목코드 앞 3자리로 ETF 여부 판별
        if len(code) >= 3:
            prefix = code[:3]
            if prefix in etf_patterns:
                return True
        
        # 추가적인 ETF 판별 로직 (API 호출 없이)
        # ETF는 보통 특정 숫자 범위에 있음
        try:
            code_num = int(code)
            # 대부분의 한국 ETF는 이 범위에 있음
            if (69000 <= code_num <= 79999) or \
               (91000 <= code_num <= 99999) or \
               (102000 <= code_num <= 169999) or \
               (210000 <= code_num <= 259999):
                return True
        except ValueError:
            # 숫자가 아닌 경우
            pass
        
        return False

    def get_current_price(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        특정 종목의 현재가 정보를 조회합니다.
        
        Args:
            stock_code: 종목 코드
            
        Returns:
            dict or None: 현재가 정보 또는 조회 실패 시 None
        """
        return self.get_real_time_price_by_api(stock_code)
    
    def get_current_price_as_entity(self, stock_code: str) -> Optional[Stock]:
        """
        특정 종목의 현재가 정보를 Stock 엔터티로 조회합니다 (NEW)
        
        Args:
            stock_code: 종목 코드
            
        Returns:
            Stock: Stock 엔터티 또는 조회 실패 시 None
        """
        try:
            # API 응답 조회
            price_data = self._get_real_time_price_raw(stock_code)
            if not price_data:
                return None
            
            # 🔧 이미 가공된 데이터로 직접 Stock 엔터티 생성 (StockMapper 우회)
            from korea_stock_auto.domain.entities import Stock
            from korea_stock_auto.domain.value_objects import Price, Money, Quantity
            from datetime import datetime
            
            try:
                stock = Stock(
                    code=stock_code,
                    name=price_data.get('stock_name', ''),
                    current_price=Price(price_data.get('current_price', 0)),
                    previous_close=Price(price_data.get('prev_close_price', 0)),
                    market_cap=Money(price_data.get('market_cap', 0)),
                    volume=Quantity(price_data.get('volume', 0)),
                    updated_at=datetime.now()
                )
                
                logger.debug(f"{stock_code} Stock 엔터티 생성 완료: {stock.current_price}")
                return stock
                
            except MappingError as e:
                logger.warning(f"{stock_code} Stock 매핑 실패: {e}")
                return None
                
        except Exception as e:
            logger.error(f"{stock_code} Stock 엔터티 조회 실패: {e}")
            return None

    def _get_real_time_price_raw(self, code: str) -> Optional[Dict[str, Any]]:
        """
        실시간 시세 조회 API (원시 응답 반환)
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict or None: 원시 API 응답
        """
        try:
            # API 호출 속도 제한 적용
            self.api_client._rate_limit()
            
            # ETF 여부에 따라 시장 구분 코드 설정
            is_etf = self.is_etf_stock(code)
            market_code = "E" if is_etf else "J"  # E: ETF, J: 주식
            
            path = "uapi/domestic-stock/v1/quotations/inquire-price"
            url = f"{self.config.current_api.base_url}/{path}"
            
            params = {
                "fid_cond_mrkt_div_code": market_code,
                "fid_input_iscd": code,
            }
            
            # 트랜잭션 ID는 실전/모의 환경에 따라 다름
            tr_id = "FHKST01010100" if self.config.use_realtime_api else "FHKST01010100"
            
            headers = self.api_client._get_headers(tr_id)
            
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self.api_client._handle_response(res, f"{code} 실시간 시세 조회 실패")
            
            # 🔍 API 응답 구조 로깅 (디버깅용)
            logger.debug(f"[{code}] API 응답 전체: {result}")
            if result:
                logger.debug(f"[{code}] 응답 키: {list(result.keys())}")
            
            if not result or result.get("rt_cd") != "0":
                logger.warning(f"[{code}] API 응답 오류: rt_cd={result.get('rt_cd') if result else 'None'}")
                return None
            
            output = result.get("output", {})
            if not output:
                logger.warning(f"[{code}] output 필드가 없음. 응답 키: {list(result.keys()) if result else []}")
                # output 대신 다른 필드명이 사용될 수 있으므로 확인
                for key in result.keys():
                    if isinstance(result[key], dict) and 'stck_prpr' in result[key]:
                        logger.info(f"[{code}] output 대신 '{key}' 필드 사용")
                        output = result[key]
                        break
            
            return output
            
        except Exception as e:
            logger.error(f"{code} 실시간 시세 원시 조회 실패: {e}")
            return None

    def get_real_time_price_by_api(self, code: str) -> Optional[Dict[str, Any]]:
        """
        실시간 시세 조회 API (백워드 호환성 유지)
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict or None: 실시간 시세 정보
        """
        try:
            output = self._get_real_time_price_raw(code)
            if not output:
                return None
            
            # 결과 가공 (기존 방식)
            try:
                # 숫자 변환 헬퍼 함수
                def safe_int(value, default=0):
                    """소수점이 포함된 문자열도 안전하게 정수로 변환"""
                    try:
                        if not value or value in ["", "-"]:
                            return default
                        # 소수점이 포함된 경우 float으로 먼저 변환 후 int로 변환
                        return int(float(value))
                    except (ValueError, TypeError):
                        return default

                def safe_float(value, default=0.0):
                    """문자열을 안전하게 실수로 변환"""
                    try:
                        if not value or value in ["", "-"]:
                            return default
                        return float(value)
                    except (ValueError, TypeError):
                        return default

                # 🔍 주식명 필드 디버깅
                stock_name = output.get("hts_kor_isnm", "")
                if not stock_name:
                    # 다른 가능한 주식명 필드들 확인
                    for name_field in ["hts_kor_isnm", "prdt_name", "kor_isnm", "iscd_stat_cls_code", "marg_rate", "stck_prdt_name"]:
                        if name_field in output and output[name_field]:
                            stock_name = output[name_field]
                            logger.debug(f"[{code}] 주식명을 '{name_field}' 필드에서 발견: {stock_name}")
                            break
                    
                    # 여전히 주식명이 없으면 API 응답 전체 로깅
                    if not stock_name:
                        logger.warning(f"[{code}] 주식명을 찾을 수 없습니다. API 응답: {output}")
                        # 코드를 주식명으로 사용 (임시)
                        stock_name = f"주식_{code}"

                stock_info = {
                    "stock_code": code,
                    "stock_name": stock_name,
                    "market": "코스피" if output.get("bstp_kor_isnm") == "코스피" else "코스닥",
                    "time": output.get("stck_prpr_time", ""),
                    "current_price": safe_int(output.get("stck_prpr", "0")),
                    "open_price": safe_int(output.get("stck_oprc", "0")),
                    "high_price": safe_int(output.get("stck_hgpr", "0")),
                    "low_price": safe_int(output.get("stck_lwpr", "0")),
                    "prev_close_price": safe_int(output.get("stck_sdpr", "0")),
                    "price_change": safe_int(output.get("prdy_vrss", "0")),
                    "change_rate": safe_float(output.get("prdy_vrss_rate", "0")),
                    "volume": safe_int(output.get("acml_vol", "0")),
                    "volume_value": safe_int(output.get("acml_tr_pbmn", "0")),
                    "market_cap": safe_int(output.get("hts_avls", "0")) * 100000000,  # 억원 단위를 원 단위로 변환
                    "listed_shares": safe_int(output.get("lstn_stcn", "0")),
                    "highest_52_week": safe_int(output.get("w52_hgpr", "0")),
                    "lowest_52_week": safe_int(output.get("w52_lwpr", "0")),
                    "per": safe_float(output.get("per", "0")),
                    "eps": safe_int(output.get("eps", "0")),
                    "pbr": safe_float(output.get("pbr", "0")),
                    "div_yield": safe_float(output.get("dvyn", "0")),
                    "foreign_rate": safe_float(output.get("hts_frgn_ehrt", "0")),
                }
                
                # 추가 계산 필드
                current_price = stock_info["current_price"]
                open_price = stock_info["open_price"]
                high_price = stock_info["high_price"]
                low_price = stock_info["low_price"]
                highest_52_week = stock_info["highest_52_week"]
                lowest_52_week = stock_info["lowest_52_week"]
                
                # 일중 변동폭 비율
                if high_price > 0 and low_price > 0:
                    stock_info["day_range_rate"] = round(((high_price - low_price) / low_price) * 100, 2)
                else:
                    stock_info["day_range_rate"] = 0
                
                # 시가 대비 현재가 비율
                if open_price > 0:
                    stock_info["current_to_open_rate"] = round(((current_price - open_price) / open_price) * 100, 2)
                else:
                    stock_info["current_to_open_rate"] = 0
                
                # 52주 신고가/신저가 여부
                stock_info["is_52week_high"] = (current_price == highest_52_week) if highest_52_week > 0 else False
                stock_info["is_52week_low"] = (current_price == lowest_52_week) if lowest_52_week > 0 else False
                
                # 52주 고점/저점 대비 거리
                if highest_52_week > 0:
                    stock_info["gap_from_52week_high"] = round(((current_price - highest_52_week) / highest_52_week) * 100, 2)
                else:
                    stock_info["gap_from_52week_high"] = 0
                
                if lowest_52_week > 0:
                    stock_info["gap_from_52week_low"] = round(((current_price - lowest_52_week) / lowest_52_week) * 100, 2)
                else:
                    stock_info["gap_from_52week_low"] = 0
                
                return stock_info
                
            except (ValueError, TypeError) as e:
                logger.error(f"{code} 실시간 시세 데이터 변환 오류: {e}")
                return None
            
        except Exception as e:
            logger.error(f"{code} 실시간 시세 조회 실패: {e}")
            return None

    def fetch_stock_current_price(self, code: str) -> Optional[Dict[str, Any]]:
        """
        주식 현재가 조회 (get_real_time_price_by_api의 별칭)
        
        Args:
            code: 종목 코드
            
        Returns:
            dict or None: 주식 현재가 정보 또는 조회 실패 시 None
        """
        return self.get_real_time_price_by_api(code)

    def clear_mapper_cache(self) -> None:
        """매퍼 캐시 전체 삭제"""
        self.stock_mapper.clear_cache()
        logger.debug("StockMapper 캐시 삭제 완료")
    
    def get_mapper_cache_stats(self) -> Dict[str, Any]:
        """매퍼 캐시 통계 조회"""
        return {
            'stock_mapper': self.stock_mapper.get_cache_stats()
        }
    
    def get_stocks_as_entities(self, stock_codes: List[str]) -> Dict[str, Optional[Stock]]:
        """
        여러 종목의 현재가를 Stock 엔터티로 조회 (NEW)
        
        Args:
            stock_codes: 종목 코드 리스트
            
        Returns:
            Dict: 종목코드 -> Stock 엔터티 (실패한 경우 None)
        """
        results = {}
        for code in stock_codes:
            try:
                stock = self.get_current_price_as_entity(code)
                results[code] = stock
                time.sleep(0.1)  # API 호출 간격 조절
            except Exception as e:
                logger.error(f"{code} Stock 엔터티 조회 실패: {e}")
                results[code] = None
        
        return results
    
    def get_top_traded_stocks(self, market_type: str = "0", top_n: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        거래량 상위 종목 조회 (국내주식-047 API 활용)
        
        Args:
            market_type (str): 시장 구분 (0:전체, 1:코스피, 2:코스닥)
            top_n (int): 조회할 종목 수 (최대 100)
            
        Returns:
            list or None: 거래량 상위 종목 목록
        """
        # 설정 가져오기
        config = get_config()
        
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
        headers = self.api_client._get_headers("FHPST01710000")
        
        try:
            # API 호출 속도 제한 적용
            self.api_client._rate_limit()
            
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
                        
                        # 유효한 종목코드가 있는 경우만 추가
                        if stock_info["code"]:
                            ranked_stocks.append(stock_info)
                            
                        # 요청한 개수만큼 수집하면 중단
                        if len(ranked_stocks) >= top_n:
                            break
                            
                    except (ValueError, KeyError, TypeError) as e:
                        logger.warning(f"종목 데이터 처리 중 오류: {e}, 건너뜀")
                        continue
            
            # 딕셔너리 형태의 응답 처리 (단일 종목)
            elif isinstance(output, dict):
                try:
                    code_key = next((k for k in output.keys() if 'iscd' in k.lower() or 'code' in k.lower()), None)
                    if code_key and output.get(code_key):
                        stock_info = {
                            "rank": output.get("no", "1"),
                            "code": output.get(code_key, ""),
                            "name": output.get(next((k for k in output.keys() if 'isnm' in k.lower() or 'name' in k.lower()), "name"), "N/A"),
                            "price": int(output.get(next((k for k in output.keys() if 'prpr' in k.lower() or 'price' in k.lower()), "price"), "0").replace(',', '')),
                            "change_rate": float(output.get(next((k for k in output.keys() if 'ctrt' in k.lower() or 'rate' in k.lower()), "rate"), "0").replace('%', '')),
                            "volume": int(output.get(next((k for k in output.keys() if 'vol' in k.lower() or 'volume' in k.lower()), "volume"), "0").replace(',', '')),
                            "market_cap": int(output.get(next((k for k in output.keys() if 'avls' in k.lower() or 'cap' in k.lower()), "market_cap"), "0").replace(',', ''))
                        }
                        ranked_stocks.append(stock_info)
                        
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"단일 종목 데이터 처리 중 오류: {e}")
            
            # 결과 로깅
            if ranked_stocks:
                logger.info(f"{market_name} 거래량 상위 {len(ranked_stocks)}개 종목 조회 성공")
                if logger.isEnabledFor(logging.DEBUG):
                    for i, stock in enumerate(ranked_stocks[:5]):  # 상위 5개만 로깅
                        logger.debug(f"{i+1}. {stock['name']}({stock['code']}): 거래량 {stock['volume']:,}")
            else:
                logger.warning(f"{market_name} 거래량 상위 종목이 없습니다.")
            
            return ranked_stocks[:top_n]  # 요청한 개수만큼만 반환
            
        except requests.exceptions.RequestException as e:
            logger.error(f"거래량 상위 종목 조회 네트워크 오류: {e}")
            send_message(f"[오류] 거래량 상위 종목 조회 네트워크 실패: {e}", config.notification.discord_webhook_url)
            return None
        except Exception as e:
            logger.error(f"거래량 상위 종목 조회 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] 거래량 상위 종목 조회 실패: {e}", config.notification.discord_webhook_url)
            return None 