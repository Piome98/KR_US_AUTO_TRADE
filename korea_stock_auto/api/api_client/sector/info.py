"""
한국 주식 자동매매 - 업종 정보 모듈
"""

import requests
import logging
from typing import Dict, List, Optional, Any, Union

from korea_stock_auto.config import URL_BASE
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

logger = logging.getLogger("stock_auto")

class SectorInfoMixin:
    """업종 정보 관련 기능 Mixin"""
    
    def get_sector_list(self, market_type: str = "01") -> Optional[List[Dict[str, Any]]]:
        """
        업종 리스트 조회
        
        Args:
            market_type (str): 시장 구분 (01:코스피, 02:코스닥)
            
        Returns:
            list or None: 업종 리스트
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/inquire-theme-list"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": market_type,
        }
        
        tr_id = "FHKST03910000"
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            market_name = "코스피" if market_type == "01" else "코스닥"
            logger.info(f"{market_name} 업종 리스트 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{market_name} 업종 리스트 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", [])
            if not output:
                logger.warning(f"{market_name} 업종 리스트가 없습니다.")
                return []
            
            # 결과 가공
            sectors = []
            for item in output:
                sector_info = {
                    "code": item.get("rprs_mrkt_n", ""),
                    "name": item.get("bstp_larg_div_name", ""),
                    "market": market_name
                }
                sectors.append(sector_info)
            
            logger.info(f"{market_name} 업종 {len(sectors)}개 조회 성공")
            return sectors
            
        except Exception as e:
            logger.error(f"업종 리스트 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 업종 리스트 조회 실패: {e}")
            return None
    
    def get_sector_stocks(self, sector_code: str, market_type: str = "01") -> Optional[List[Dict[str, Any]]]:
        """
        업종 내 종목 리스트 조회
        
        Args:
            sector_code (str): 업종 코드
            market_type (str): 시장 구분 (01:코스피, 02:코스닥)
            
        Returns:
            list or None: 업종 내 종목 리스트
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": market_type,
            "FID_INPUT_ISCD": sector_code,
            "FID_PERIOD_DIV_CODE": "D",  # 일봉
            "FID_INPUT_DATE_1": "",  # 빈 값으로 설정하면 최신 데이터 조회
            "FID_DIR_CLS_CODE": "1",  # 정순
            "FID_INPUT_DATE_2": "",
            "FID_ORG_ADJ_PRC": "1",  # 원주가
            "FID_SECT": "",
            "FID_VOL_CNT": "1"  # 최신 1개만 조회
        }
        
        tr_id = "FHPUP03900000"
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            market_name = "코스피" if market_type == "01" else "코스닥"
            logger.info(f"{market_name} 업종코드 {sector_code} 내 종목 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"업종 내 종목 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", [])
            if not output:
                logger.warning(f"업종코드 {sector_code} 내 종목이 없습니다.")
                return []
            
            # 결과 가공
            stocks = []
            for item in output:
                stock_info = {
                    "code": item.get("stck_shrn_iscd", ""),
                    "name": item.get("hts_kor_isnm", ""),
                    "price": int(item.get("stck_prpr", "0").replace(',', '')),
                    "change_rate": float(item.get("prdy_ctrt", "0")),
                    "volume": int(item.get("acml_vol", "0").replace(',', '')),
                    "market_cap": int(item.get("hts_avls", "0").replace(',', '')),
                    "sector": item.get("bstp_larg_div_name", ""),
                    "market": market_name
                }
                stocks.append(stock_info)
            
            logger.info(f"업종코드 {sector_code} 내 종목 {len(stocks)}개 조회 성공")
            return stocks
            
        except Exception as e:
            logger.error(f"업종 내 종목 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 업종 내 종목 조회 실패: {e}")
            return None
    
    def get_hot_sectors(self, market_type: str = "01", count: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        상승률 상위 업종 조회
        
        Args:
            market_type (str): 시장 구분 (01:코스피, 02:코스닥)
            count (int): 조회할 업종 수
            
        Returns:
            list or None: 상승률 상위 업종 목록
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/inquire-updown-sector"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": market_type,
            "FID_COND_SCR_DIV_CODE": "20400",
            "FID_INPUT_ISCD": "0002",  # 업종코드는 의미없음
            "FID_DIV_CLS_CODE": "0",   # 등락률 정렬기준 0:상승률순 1:하락률순
            "FID_BLNG_CLS_CODE": "0",  # 시장분류
            "FID_TRGT_CLS_CODE": "0",  # 매매주체
            "FID_TRGT_EXLS_CLS_CODE": "0",  # 제외할 매매주체
            "FID_INPUT_PRICE_1": "",
            "FID_INPUT_PRICE_2": "",
            "FID_VOL_CNT": str(count),
            "FID_INPUT_DATE_1": ""
        }
        
        tr_id = "FHPUP03700100"
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            market_name = "코스피" if market_type == "01" else "코스닥"
            logger.info(f"{market_name} 상승률 상위 업종 {count}개 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"상승률 상위 업종 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", [])
            if not output:
                logger.warning(f"{market_name} 상승률 상위 업종 데이터가 없습니다.")
                return []
            
            # 결과 가공
            hot_sectors = []
            for item in output:
                sector_info = {
                    "rank": int(item.get("no", "0")),
                    "code": item.get("mksc_shrn_iscd", ""),
                    "name": item.get("bstp_larg_div_name", ""),
                    "current": float(item.get("bstp_nmix_prpr", "0")),
                    "change": float(item.get("prdy_vrss", "0")),
                    "change_rate": float(item.get("prdy_ctrt", "0")),
                    "volume": int(item.get("acml_vol", "0").replace(',', '')) if item.get("acml_vol") else 0,
                    "trade_value": int(item.get("acml_tr_pbmn", "0").replace(',', '')) if item.get("acml_tr_pbmn") else 0,
                    "market": market_name
                }
                hot_sectors.append(sector_info)
            
            logger.info(f"{market_name} 상승률 상위 업종 {len(hot_sectors)}개 조회 성공")
            return hot_sectors
            
        except Exception as e:
            logger.error(f"상승률 상위 업종 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 상승률 상위 업종 조회 실패: {e}")
            return None
    
    def get_theme_list(self) -> Optional[List[Dict[str, Any]]]:
        """
        테마 리스트 조회
        
        Returns:
            list or None: 테마 리스트
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/inquire-theme-list"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "T",  # 테마
        }
        
        tr_id = "FHKST03910000"
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info("테마 리스트 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, "테마 리스트 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", [])
            if not output:
                logger.warning("테마 리스트가 없습니다.")
                return []
            
            # 결과 가공
            themes = []
            for item in output:
                theme_info = {
                    "code": item.get("tmname", ""),
                    "name": item.get("tmname", ""),
                }
                themes.append(theme_info)
            
            logger.info(f"테마 {len(themes)}개 조회 성공")
            return themes
            
        except Exception as e:
            logger.error(f"테마 리스트 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 테마 리스트 조회 실패: {e}")
            return None
    
    def get_theme_stocks(self, theme_code: str) -> Optional[List[Dict[str, Any]]]:
        """
        테마 내 종목 리스트 조회
        
        Args:
            theme_code (str): 테마 코드(테마명)
            
        Returns:
            list or None: 테마 내 종목 리스트
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/inquire-theme-member"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "T",  # 테마
            "FID_INPUT_ISCD": theme_code
        }
        
        tr_id = "FHKST03910200"
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"테마 '{theme_code}' 내 종목 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"테마 내 종목 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", [])
            if not output:
                logger.warning(f"테마 '{theme_code}' 내 종목이 없습니다.")
                return []
            
            # 결과 가공
            stocks = []
            for item in output:
                stock_info = {
                    "code": item.get("shcode", ""),
                    "name": item.get("hname", ""),
                    "price": int(item.get("price", "0").replace(',', '')),
                    "change_rate": float(item.get("rate", "0").replace('%', '')),
                    "volume": int(item.get("vol", "0").replace(',', '')),
                    "market_cap": int(item.get("marcap", "0").replace(',', '')),
                    "theme": theme_code
                }
                stocks.append(stock_info)
            
            logger.info(f"테마 '{theme_code}' 내 종목 {len(stocks)}개 조회 성공")
            return stocks
            
        except Exception as e:
            logger.error(f"테마 내 종목 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 테마 내 종목 조회 실패: {e}")
            return None
    
    def get_hot_themes(self, count: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        상승률 상위 테마 조회
        
        Args:
            count (int): 조회할 테마 수
            
        Returns:
            list or None: 상승률 상위 테마 목록
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/inquire-theme-trend"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "T",  # 테마
            "FID_TDAY_RTUR_RANK_SORT_CLS_CODE": "1",  # 일간 상승률순 정렬
            "FID_INPUT_ISCD": "0001"  # 의미 없음
        }
        
        tr_id = "FHKUP03800000"
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"상승률 상위 테마 {count}개 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"상승률 상위 테마 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", [])
            if not output:
                logger.warning("상승률 상위 테마 데이터가 없습니다.")
                return []
            
            # 결과 가공 - 상위 count개만 추출
            hot_themes = []
            for i, item in enumerate(output):
                if i >= count:
                    break
                    
                theme_info = {
                    "rank": i + 1,
                    "code": item.get("tmname", ""),
                    "name": item.get("tmname", ""),
                    "change_rate": float(item.get("tday_rtur", "0")),
                    "market_cap": int(item.get("tmcap", "0").replace(',', '')),
                    "stocks_count": int(item.get("totcnt", "0"))
                }
                hot_themes.append(theme_info)
            
            logger.info(f"상승률 상위 테마 {len(hot_themes)}개 조회 성공")
            return hot_themes
            
        except Exception as e:
            logger.error(f"상승률 상위 테마 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 상승률 상위 테마 조회 실패: {e}")
            return None 