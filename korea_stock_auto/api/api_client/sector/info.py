"""
한국 주식 자동매매 - 업종 정보 모듈

업종 및 테마 관련 정보 조회 기능을 제공합니다.
"""

import requests
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING, cast

from korea_stock_auto.config import get_config
from korea_stock_auto.utils.utils import send_message

# 타입 힌트만을 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

# 로깅 설정
logger = logging.getLogger("stock_auto")

class SectorInfoMixin:
    """
    업종 정보 관련 기능 Mixin
    
    업종 리스트 조회, 업종 내 종목 조회, 테마 정보 조회 등 업종 관련 기능을 제공합니다.
    """
    
    def get_sector_list(self, market_type: str = "01") -> Optional[List[Dict[str, Any]]]:
        """
        업종 리스트 조회
        
        Args:
            market_type: 시장 구분 (01:코스피, 02:코스닥)
            
        Returns:
            list or None: 업종 리스트 (실패 시 None)
            
        Examples:
            >>> api_client.get_sector_list("01")  # 코스피 업종 리스트 조회
            >>> api_client.get_sector_list("02")  # 코스닥 업종 리스트 조회
        """

        # 설정 로드
        config = get_config()
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/quotations/inquire-theme-list"
        url = f"{config.current_api.base_url}/{path}"
        
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
            send_message(f"[오류] 업종 리스트 조회 실패: {e}", config.notification.discord_webhook_url)
            return None
    
    def get_sector_stocks(self, sector_code: str, market_type: str = "01") -> Optional[List[Dict[str, Any]]]:
        """
        업종 내 종목 리스트 조회
        
        Args:
            sector_code: 업종 코드
            market_type: 시장 구분 (01:코스피, 02:코스닥)
            
        Returns:
            list or None: 업종 내 종목 리스트 (실패 시 None)
            
        Examples:
            >>> api_client.get_sector_stocks("001", "01")  # 코스피 특정 업종 내 종목 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        url = f"{config.current_api.base_url}/{path}"
        
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
            send_message(f"[오류] 업종 내 종목 조회 실패: {e}", config.notification.discord_webhook_url)
            return None
    
    def get_hot_sectors(self, market_type: str = "01", count: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        상승률 상위 업종 조회
        
        Args:
            market_type: 시장 구분 (01:코스피, 02:코스닥)
            count: 조회할 업종 수
            
        Returns:
            list or None: 상승률 상위 업종 목록 (실패 시 None)
            
        Examples:
            >>> api_client.get_hot_sectors("01", 5)  # 코스피 상승률 상위 5개 업종 조회
        """
        # 설정 로드
        config = get_config()
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/quotations/inquire-updown-sector"
        url = f"{config.current_api.base_url}/{path}"
        
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
                    "code": item.get("sec_code", ""),
                    "name": item.get("sec_name", ""),
                    "current": float(item.get("prsnt_prc", "0")),
                    "change": float(item.get("prdy_vrss", "0")),
                    "change_rate": float(item.get("prdy_ctrt", "0")),
                    "volume": int(item.get("vol", "0").replace(',', '')),
                    "market": market_name
                }
                hot_sectors.append(sector_info)
            
            logger.info(f"{market_name} 상승률 상위 업종 {len(hot_sectors)}개 조회 성공")
            return hot_sectors
            
        except Exception as e:
            logger.error(f"상승률 상위 업종 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 상승률 상위 업종 조회 실패: {e}", config.notification.discord_webhook_url)
            return None
    
    def get_theme_list(self) -> Optional[List[Dict[str, Any]]]:
        """
        테마 리스트 조회
        
        Returns:
            list or None: 테마 리스트 (실패 시 None)
            
        Examples:
            >>> api_client.get_theme_list()  # 전체 테마 리스트 조회
        """
        # 설정 로드
        config = get_config()
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/quotations/theme-list"
        url = f"{config.current_api.base_url}/{path}"
        
        tr_id = "FHKST03030100"
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info("테마 리스트 조회 요청")
            res = requests.get(url, headers=headers, timeout=10)
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
                    "code": item.get("tmcode", ""),
                    "name": item.get("tmname", "")
                }
                themes.append(theme_info)
            
            logger.info(f"테마 {len(themes)}개 조회 성공")
            return themes
            
        except Exception as e:
            logger.error(f"테마 리스트 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 테마 리스트 조회 실패: {e}", config.notification.discord_webhook_url)
            return None
    
    def get_theme_stocks(self, theme_code: str) -> Optional[List[Dict[str, Any]]]:
        """
        테마 내 종목 리스트 조회
        
        Args:
            theme_code: 테마 코드
            
        Returns:
            list or None: 테마 내 종목 리스트 (실패 시 None)
            
        Examples:
            >>> api_client.get_theme_stocks("T0001")  # 특정 테마 내 종목 조회
        """
        # 설정 로드
        config = get_config()
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/quotations/theme-items"
        url = f"{config.current_api.base_url}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "0",  # 전체
            "FID_INPUT_ISCD": theme_code
        }
        
        tr_id = "FHKST01702000"
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"테마코드 {theme_code} 내 종목 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"테마 내 종목 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", [])
            if not output:
                logger.warning(f"테마코드 {theme_code} 내 종목이 없습니다.")
                return []
            
            # 결과 가공
            stocks = []
            for item in output:
                stock_info = {
                    "code": item.get("mksc_shrn_iscd", ""),
                    "name": item.get("hts_kor_isnm", ""),
                    "price": int(item.get("stck_prpr", "0").replace(',', '')),
                    "change_rate": float(item.get("prdy_ctrt", "0")),
                    "volume": int(item.get("acml_vol", "0").replace(',', '')),
                    "market_cap": int(item.get("hts_avls", "0").replace(',', '')),
                    "market": "코스피" if item.get("bstp_larg_div_name", "").startswith("코스피") else "코스닥"
                }
                stocks.append(stock_info)
            
            logger.info(f"테마코드 {theme_code} 내 종목 {len(stocks)}개 조회 성공")
            return stocks
            
        except Exception as e:
            logger.error(f"테마 내 종목 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 테마 내 종목 조회 실패: {e}", config.notification.discord_webhook_url)
            return None
    
    def get_hot_themes(self, count: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        상승률 상위 테마 조회
        
        Args:
            count: 조회할 테마 수
            
        Returns:
            list or None: 상승률 상위 테마 목록 (실패 시 None)
            
        Examples:
            >>> api_client.get_hot_themes(5)  # 상승률 상위 5개 테마 조회
        """
        # 설정 로드
        config = get_config()
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/quotations/theme-updown-rank"
        url = f"{config.current_api.base_url}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "0",  # 전체
            "FID_COND_SCR_DIV_CODE": "20171",
            "FID_INPUT_ISCD": "0002",  # 테마코드는 의미없음
            "FID_DIV_CLS_CODE": "0",   # 등락률 정렬기준 0:상승률순 1:하락률순
            "FID_BLNG_CLS_CODE": "0",  # 시장분류
            "FID_TRGT_CLS_CODE": "111111111",  # 매매주체
            "FID_TRGT_EXLS_CLS_CODE": "000000000",  # 제외할 매매주체
            "FID_INPUT_PRICE_1": "0",
            "FID_INPUT_PRICE_2": "0",
            "FID_VOL_CNT": str(count),
            "FID_INPUT_DATE_1": ""
        }
        
        tr_id = "FHKUP03035001"
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
            
            # 결과 가공
            hot_themes = []
            for item in output:
                theme_info = {
                    "rank": int(item.get("no", "0")),
                    "code": item.get("tmcode", ""),
                    "name": item.get("tmname", ""),
                    "change_rate": float(item.get("prdy_ctrt", "0")),
                    "stock_count": int(item.get("tmstk_cnt", "0")),
                    "up_count": int(item.get("up_tmstk_cnt", "0")),
                    "down_count": int(item.get("down_tmstk_cnt", "0")),
                    "unchanged_count": int(item.get("pyod_tmstk_cnt", "0")),
                    "top_stock_code": item.get("top1_tmstk_shrn_iscd", ""),
                    "top_stock_name": item.get("top1_tmstk_hts_kor_isnm", ""),
                    "top_stock_change_rate": float(item.get("top1_tmstk_prdy_ctrt", "0"))
                }
                hot_themes.append(theme_info)
            
            logger.info(f"상승률 상위 테마 {len(hot_themes)}개 조회 성공")
            return hot_themes
            
        except Exception as e:
            logger.error(f"상승률 상위 테마 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 상승률 상위 테마 조회 실패: {e}", config.notification.discord_webhook_url)
            return None 