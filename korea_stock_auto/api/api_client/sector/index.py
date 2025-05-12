"""
한국 주식 자동매매 - 업종 지수 모듈

코스피, 코스닥 등 주요 지수 및 업종별 지수 조회 기능을 제공합니다.
"""

import requests
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING, cast
from datetime import datetime

from korea_stock_auto.config import URL_BASE
from korea_stock_auto.utils.utils import send_message

# 타입 힌트만을 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

# 로깅 설정
logger = logging.getLogger("stock_auto")

class SectorIndexMixin:
    """
    업종 지수 관련 기능 Mixin
    
    주요 지수 조회 및 업종별 지수 조회 기능을 제공합니다.
    """
    
    def get_market_index(self, index_code: str = "0001") -> Optional[Dict[str, Any]]:
        """
        주요 지수 조회 (코스피, 코스닥 등)
        
        Args:
            index_code: 지수 코드
                - 0001: 코스피
                - 1001: 코스닥
                - 2001: 코스피200
                - 0028: 코스피 건설업
                - 0034: 코스피 금융업
                - 0035: 코스피 보험업
                - 0150: 코스피 운수창고업
                - 0037: 코스피 전기전자
                - 0009: 코스피 의약품
                - 그 외 각종 섹터 인덱스 코드 사용 가능
                
        Returns:
            dict or None: 지수 정보 (실패 시 None)
            
        Examples:
            >>> api_client.get_market_index("0001")  # 코스피 지수 조회
            >>> api_client.get_market_index("1001")  # 코스닥 지수 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/quotations/inquire-index"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "U",
            "FID_INPUT_ISCD": index_code
        }
        
        tr_id = "FHKUP03500000"
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            # 지수 이름 매핑
            index_names = {
                "0001": "코스피",
                "1001": "코스닥",
                "2001": "코스피200",
                "0028": "코스피 건설업",
                "0034": "코스피 금융업",
                "0035": "코스피 보험업",
                "0150": "코스피 운수창고업",
                "0037": "코스피 전기전자",
                "0009": "코스피 의약품"
            }
            name = index_names.get(index_code, f"지수({index_code})")
            
            logger.info(f"{name} 지수 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{name} 지수 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", {})
            
            # 결과 가공
            index_info = {
                "code": index_code,
                "name": name,
                "current": float(output.get("stck_prpr", "0")),
                "change": float(output.get("prdy_vrss", "0")),
                "change_rate": float(output.get("prdy_ctrt", "0")),
                "volume": int(output.get("acml_vol", "0").replace(',', '')),
                "trade_value": int(output.get("acml_tr_pbmn", "0").replace(',', '')),
                "market_cap": int(output.get("hts_avls", "0").replace(',', '')),
                "highest": float(output.get("stck_mxpr", "0")),
                "lowest": float(output.get("stck_llam", "0")),
                "opening": float(output.get("stck_oprc", "0")),
                "prev_close": float(output.get("stck_sdpr", "0")),
                "timestamp": output.get("stck_bsop_date", "")
            }
            
            # 상승/하락/보합 여부
            if index_info["change"] > 0:
                index_info["status"] = "상승"
            elif index_info["change"] < 0:
                index_info["status"] = "하락"
            else:
                index_info["status"] = "보합"
            
            logger.info(f"{name} 지수 조회 성공: {index_info['current']} ({index_info['status']} {abs(index_info['change_rate'])}%)")
            return index_info
            
        except Exception as e:
            logger.error(f"지수 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 지수 조회 실패: {e}")
            return None
    
    def get_index_chart_daily(self, index_code: str = "0001", period: int = 30) -> Optional[List[Dict[str, Any]]]:
        """
        주요 지수 일별 시세 조회
        
        Args:
            index_code: 지수 코드
                - 0001: 코스피
                - 1001: 코스닥
                - 2001: 코스피200
            period: 조회 기간(일)
                
        Returns:
            list or None: 일별 지수 정보 목록 (실패 시 None)
            
        Examples:
            >>> api_client.get_index_chart_daily("0001", 10)  # 코스피 지수 10일치 일별 시세 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/quotations/inquire-index-daily-price"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "U",
            "FID_INPUT_ISCD": index_code,
            "FID_PERIOD_DIV_CODE": "D",
            "FID_INPUT_DATE_1": "",  # 빈 값으로 설정하면 최신 데이터부터 조회
            "FID_INPUT_DATE_2": "",
            "FID_VOL_CNT": str(period)
        }
        
        tr_id = "FHKUP03500001"
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            # 지수 이름 매핑
            index_names = {
                "0001": "코스피",
                "1001": "코스닥",
                "2001": "코스피200"
            }
            name = index_names.get(index_code, f"지수({index_code})")
            
            logger.info(f"{name} 지수 일별 시세 {period}일치 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{name} 지수 일별 시세 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output = result.get("output", [])
            if not output:
                logger.warning(f"{name} 지수 일별 시세 데이터가 없습니다.")
                return []
            
            # 결과 가공
            daily_prices = []
            for item in output:
                daily_info = {
                    "date": item.get("stck_bsop_date", ""),
                    "opening": float(item.get("bstp_nmix_oprc", "0")),
                    "high": float(item.get("bstp_nmix_hgpr", "0")),
                    "low": float(item.get("bstp_nmix_lwpr", "0")),
                    "close": float(item.get("bstp_nmix_prpr", "0")),
                    "change": float(item.get("prdy_vrss", "0")),
                    "change_rate": float(item.get("prdy_ctrt", "0")),
                    "volume": int(item.get("acml_vol", "0").replace(',', '')) if item.get("acml_vol") else 0,
                    "trade_value": int(item.get("acml_tr_pbmn", "0").replace(',', '')) if item.get("acml_tr_pbmn") else 0
                }
                daily_prices.append(daily_info)
            
            logger.info(f"{name} 지수 일별 시세 {len(daily_prices)}일치 조회 성공")
            return daily_prices
            
        except Exception as e:
            logger.error(f"지수 일별 시세 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 지수 일별 시세 조회 실패: {e}")
            return None
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        주요 지수 시장 상황 종합 조회
        
        코스피, 코스닥, 코스피200 지수를 한 번에 조회하여 시장 상황을 종합적으로 파악합니다.
        
        Returns:
            dict: 시장 상황 종합 정보
            
        Examples:
            >>> api_client.get_market_status()  # 주요 지수 시장 상황 종합 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # 주요 지수 조회
        kospi = self.get_market_index("0001")
        kosdaq = self.get_market_index("1001")
        kospi200 = self.get_market_index("2001")
        
        # 결과 가공
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        market_status = {
            "timestamp": now,
            "kospi": kospi,
            "kosdaq": kosdaq,
            "kospi200": kospi200
        }
        
        return market_status 