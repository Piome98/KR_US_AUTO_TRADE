"""
한국 주식 자동매매 - 주식 기본 정보 조회 모듈
"""

import requests
import logging
import os
import pandas as pd
import json
import datetime
from typing import Dict, List, Optional, Any, Union

from korea_stock_auto.config import URL_BASE
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

logger = logging.getLogger("stock_auto")

class StockInfoMixin:
    """주식 기본 정보 조회 관련 기능 Mixin"""
    
    def get_stock_info(self, code: str) -> Optional[Dict[str, Any]]:
        """
        종목 기본 정보 조회
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict or None: 종목 기본 정보
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/search-info"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code
        }
        
        headers = self._get_headers("CTPF1002R")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} 종목 기본 정보 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} 종목 정보 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
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
            logger.error(f"{code} 종목 정보 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 종목 정보 조회 실패: {e}")
            return None
    
    def get_top_traded_stocks(self, market_type: str = "0", top_n: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        거래량 상위 종목 조회
        
        Args:
            market_type (str): 시장 구분 (0:전체, 1:코스피, 2:코스닥)
            top_n (int): 조회할 종목 수 (최대 100)
            
        Returns:
            list or None: 거래량 상위 종목 목록
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/volume-rank"
        url = f"{URL_BASE}/{path}"
        
        if top_n > 100:
            top_n = 100  # 최대 100개까지만 조회 가능
            
        params = {
            "FID_COND_MRKT_DIV_CODE": market_type,
            "FID_COND_SCR_DIV_CODE": "20171",
            "FID_INPUT_ISCD": "0000",
            "FID_DIV_CLS_CODE": "0",  # 0: 거래량, 1: 거래대금, 2: 거래량 급증, 3: 거래대금 급증
            "FID_BLNG_CLS_CODE": "0",
            "FID_TRGT_CLS_CODE": "111111111",
            "FID_TRGT_EXLS_CLS_CODE": "000000",
            "FID_INPUT_PRICE_1": "",
            "FID_INPUT_PRICE_2": "",
            "FID_VOL_CNT": str(top_n),
            "FID_INPUT_DATE_1": ""
        }
        
        headers = self._get_headers("FHPST01710000")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            market_name = {
                "0": "전체", 
                "1": "코스피", 
                "2": "코스닥"
            }.get(market_type, "전체")
            
            logger.info(f"{market_name} 거래량 상위 {top_n}개 종목 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"거래량 상위 종목 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                # API 호출 실패 시 캐시된 데이터 사용
                logger.warning(f"API를 통한 거래량 상위 종목 조회 실패, 캐시된 데이터 확인")
                cache_file = os.path.join(os.path.dirname(__file__), f'../../../../data/cache/volume_rank_{market_type}.json')
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                            if cached_data and isinstance(cached_data, list):
                                cache_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
                                current_time = datetime.datetime.now()
                                # 캐시가 24시간 이내인 경우에만 사용
                                if (current_time - cache_time).total_seconds() < 86400:
                                    logger.info(f"캐시된 거래량 상위 종목 데이터 사용 (캐시 시간: {cache_time})")
                                    return cached_data
                                else:
                                    logger.warning(f"캐시된 데이터가 오래됨 (캐시 시간: {cache_time})")
                    except Exception as e:
                        logger.error(f"캐시 파일 읽기 실패: {e}")
                
                return None
                
            stocks = result.get("output", [])
            if not stocks:
                logger.warning("거래량 상위 종목이 없습니다.")
                return []
            
            # 결과 가공
            ranked_stocks = []
            for stock in stocks:
                stock_info = {
                    "rank": stock.get("no", "0"),
                    "code": stock.get("mksc_shrn_iscd", ""),
                    "name": stock.get("hts_kor_isnm", ""),
                    "price": int(stock.get("stck_prpr", "0").replace(',', '')),
                    "change_rate": float(stock.get("prdy_ctrt", "0")),
                    "volume": int(stock.get("acml_vol", "0").replace(',', '')),
                    "market_cap": int(stock.get("hts_avls", "0").replace(',', ''))
                }
                ranked_stocks.append(stock_info)
            
            # 결과 캐싱
            try:
                cache_dir = os.path.join(os.path.dirname(__file__), '../../../../data/cache')
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, f'volume_rank_{market_type}.json')
                with open(cache_file, 'w') as f:
                    json.dump(ranked_stocks, f)
                logger.info(f"거래량 상위 종목 데이터 캐싱 완료: {cache_file}")
            except Exception as e:
                logger.error(f"거래량 상위 종목 데이터 캐싱 실패: {e}")
            
            logger.info(f"{market_name} 거래량 상위 {len(ranked_stocks)}개 종목 조회 성공")
            return ranked_stocks
            
        except Exception as e:
            logger.error(f"거래량 상위 종목 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 거래량 상위 종목 조회 실패: {e}")
            
            # 예외 발생 시 캐시된 데이터 사용
            logger.warning(f"API 조회 중 예외 발생, 캐시된 데이터 확인")
            cache_file = os.path.join(os.path.dirname(__file__), f'../../../../data/cache/volume_rank_{market_type}.json')
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        if cached_data and isinstance(cached_data, list):
                            cache_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
                            current_time = datetime.datetime.now()
                            # 캐시가 24시간 이내인 경우에만 사용
                            if (current_time - cache_time).total_seconds() < 86400:
                                logger.info(f"캐시된 거래량 상위 종목 데이터 사용 (캐시 시간: {cache_time})")
                                return cached_data
                except Exception as e:
                    logger.error(f"캐시 파일 읽기 실패: {e}")
            
            return None
    
    def get_volume_increasing_stocks(self, market_type: str = "0", top_n: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        거래량 급증 종목 조회 (국내주식-047 API 활용)
        
        Args:
            market_type (str): 시장 구분 (0:전체, 1:코스피, 2:코스닥)
            top_n (int): 조회할 종목 수 (최대 100)
            
        Returns:
            list or None: 거래량 급증 종목 목록
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/volume-rank"
        url = f"{URL_BASE}/{path}"
        
        if top_n > 100:
            top_n = 100  # 최대 100개까지만 조회 가능
            
        params = {
            "FID_COND_MRKT_DIV_CODE": market_type,
            "FID_COND_SCR_DIV_CODE": "20171",
            "FID_INPUT_ISCD": "0000",
            "FID_DIV_CLS_CODE": "2",  # 0: 거래량, 1: 거래대금, 2: 거래량 급증, 3: 거래대금 급증
            "FID_BLNG_CLS_CODE": "0",
            "FID_TRGT_CLS_CODE": "111111111",
            "FID_TRGT_EXLS_CLS_CODE": "000000",
            "FID_INPUT_PRICE_1": "",
            "FID_INPUT_PRICE_2": "",
            "FID_VOL_CNT": str(top_n),
            "FID_INPUT_DATE_1": ""
        }
        
        headers = self._get_headers("FHPST01710000")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            market_name = {
                "0": "전체", 
                "1": "코스피", 
                "2": "코스닥"
            }.get(market_type, "전체")
            
            logger.info(f"{market_name} 거래량 급증 {top_n}개 종목 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"거래량 급증 종목 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                logger.warning(f"API를 통한 거래량 급증 종목 조회 실패")
                return None
                
            stocks = result.get("output", [])
            if not stocks:
                logger.warning("거래량 급증 종목이 없습니다.")
                return []
            
            # 결과 가공
            ranked_stocks = []
            for stock in stocks:
                stock_info = {
                    "rank": stock.get("no", "0"),
                    "code": stock.get("mksc_shrn_iscd", ""),
                    "name": stock.get("hts_kor_isnm", ""),
                    "price": int(stock.get("stck_prpr", "0").replace(',', '')),
                    "change_rate": float(stock.get("prdy_ctrt", "0")),
                    "volume": int(stock.get("acml_vol", "0").replace(',', '')),
                    "volume_ratio": float(stock.get("vol_inrt", "0")),  # 거래량 증가율
                    "market_cap": int(stock.get("hts_avls", "0").replace(',', ''))
                }
                ranked_stocks.append(stock_info)
            
            # 결과 캐싱
            try:
                cache_dir = os.path.join(os.path.dirname(__file__), '../../../../data/cache')
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, f'volume_increasing_{market_type}.json')
                with open(cache_file, 'w') as f:
                    json.dump(ranked_stocks, f)
                logger.info(f"거래량 급증 종목 데이터 캐싱 완료: {cache_file}")
            except Exception as e:
                logger.error(f"거래량 급증 종목 데이터 캐싱 실패: {e}")
            
            logger.info(f"{market_name} 거래량 급증 {len(ranked_stocks)}개 종목 조회 성공")
            return ranked_stocks
            
        except Exception as e:
            logger.error(f"거래량 급증 종목 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 거래량 급증 종목 조회 실패: {e}")
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
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/inquire-investor"
        url = f"{URL_BASE}/{path}"
        
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
            output1 = result.get("output1", {})
            output2 = result.get("output2", [])
            
            if not output2:
                logger.warning(f"{code} 투자자별 매매현황 데이터가 없습니다.")
                return None
            
            # 종목 기본 정보
            investor_info = {
                "stock_code": code,
                "stock_name": output1.get("hts_kor_isnm", ""),
                "current_price": int(output1.get("stck_prpr", "0").replace(',', '')),
                "change_rate": float(output1.get("prdy_ctrt", "0").replace(',', '')),
                "volume": int(output1.get("acml_vol", "0").replace(',', '')),
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
            for item in output2:
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
            send_message(f"[오류] {code} 투자자별 매매현황 조회 실패: {e}")
            return None 