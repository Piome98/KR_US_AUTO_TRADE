"""
한국 주식 자동매매 - 차트 데이터 조회 모듈
일별/주별/월별 주가 데이터 및 시간대별 체결 내역 조회 기능 제공
"""

import requests
import logging
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING, cast
from datetime import datetime

from korea_stock_auto.config import get_config
from korea_stock_auto.utils.utils import send_message

# 타입 힌트만을 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

# 로깅 설정
logger = logging.getLogger("stock_auto")

class ChartDataMixin:
    """
    차트 데이터 조회 관련 기능 Mixin
    
    일별/주별/월별 주가 데이터 및 시간대별 체결 내역 조회 기능을 제공합니다.
    """
    
    def fetch_daily_price(self, code: str, date_from: str, date_to: str) -> Optional[Dict[str, Any]]:
        """
        일별 주가 조회
        
        한국투자증권 API를 통해 특정 종목의 일별 주가 데이터를 조회합니다.
        
        Args:
            code: 종목 코드
            date_from: 시작일(YYYYMMDD)
            date_to: 종료일(YYYYMMDD)
            
        Returns:
            dict or None: 일별 주가 정보 (실패 시 None)
            
        Notes:
            모의투자와 실전투자 모두 지원하는 함수입니다.
        """

        # 설정 로드
        config = get_config()
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        path = "uapi/domestic-stock/v1/quotations/inquire-daily-price"
        url = f"{config.current_api.base_url}/{path}"
        
        params = {
            "fid_cond_mrkt_div_code": "J",  # 시장 구분 코드: J-주식
            "fid_input_iscd": code,         # 종목 코드
            "fid_period_div_code": "D",     # 기간 구분: D-일
            "fid_org_adj_prc": "1",         # 수정주가 여부: 1-수정주가
            "FID_INPUT_DATE_1": date_from,  # 시작일
            "FID_INPUT_DATE_2": date_to,    # 종료일
        }
        
        try:
            headers = self._get_headers("FHKST01010400")
            
            logger.info(f"{code} 일별 주가 조회 요청 ({date_from} ~ {date_to})")
            return self._request_get(url, headers, params, f"{code} 일별 주가 조회 실패")
            
        except Exception as e:
            logger.error(f"{code} 일별 주가 조회 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] {code} 일별 주가 조회 실패: {e}", config.notification.discord_webhook_url)
            return None
    
    def get_stock_data(self, code: str, period_div_code: str = "D") -> Optional[List[Dict[str, Any]]]:
        """
        주식 시세 데이터 조회 (일/주/월 선택)
        
        한국투자증권 API를 통해 특정 종목의 일/주/월 시세 데이터를 조회합니다.
        
        Args:
            code: 종목 코드
            period_div_code: 조회 주기 (D:일, W:주, M:월)
            
        Returns:
            list or None: 시세 데이터 목록 (최신 순) (실패 시 None)
            
        Notes:
            모의투자와 실전투자 모두 지원하는 함수입니다.
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # 설정 로드
        config = get_config()
        
        path = "uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        url = f"{config.current_api.base_url}/{path}"
        
        period_name = {"D": "일봉", "W": "주봉", "M": "월봉"}.get(period_div_code, "일봉")
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # 시장 구분 코드: J-주식
            "FID_INPUT_ISCD": code,         # 종목 코드
            "FID_INPUT_DATE_1": "",         # 빈 값으로 두면 최근 데이터부터 조회
            "FID_INPUT_DATE_2": "",         # 빈 값으로 두면 최근 데이터부터 조회
            "FID_PERIOD_DIV_CODE": period_div_code,  # 기간 구분: D-일, W-주, M-월
            "FID_ORG_ADJ_PRC": "0"          # 수정주가 여부: 0-원주가
        }
        
        try:
            headers = self._get_headers("FHKST03010100")
            
            logger.info(f"{code} {period_name} 데이터 조회 요청")
            result = self._request_get(url, headers, params, f"{code} {period_name} 데이터 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                logger.error(f"{code} {period_name} 데이터 조회 결과가 유효하지 않습니다.")
                return None
                
            output1 = result.get("output1", {})
            output2 = result.get("output2", [])
            
            if not output2:
                logger.warning(f"{code} {period_name} 데이터가 없습니다.")
                return []
            
            # 응답 구조 확인을 위한 로깅 추가
            if output2 and len(output2) > 0:
                logger.info(f"{code} {period_name} 첫 번째 데이터 항목 키: {list(output2[0].keys())}")
                logger.info(f"{code} {period_name} 첫 번째 데이터 항목 값: {output2[0]}")
            
            # 결과 가공
            chart_data = []
            for item in output2:
                data_item = {
                    "date": item.get("stck_bsop_date", ""),
                    "open": int(item.get("stck_oprc", "0").replace(',', '')),
                    "high": int(item.get("stck_hgpr", "0").replace(',', '')),
                    "low": int(item.get("stck_lwpr", "0").replace(',', '')),
                    "close": int(item.get("stck_clpr", "0").replace(',', '')),
                    "volume": int(item.get("acml_vol", "0").replace(',', '')),
                    "value": int(item.get("acml_tr_pbmn", "0").replace(',', ''))
                }
                chart_data.append(data_item)
            
            logger.info(f"{code} {period_name} 데이터 {len(chart_data)}건 조회 성공")
            return chart_data
            
        except Exception as e:
            logger.error(f"{code} {period_name} 데이터 조회 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] {code} {period_name} 데이터 조회 실패: {e}", config.notification.discord_webhook_url)
            return None
    
    def get_daily_data(self, code: str, days: int = 30) -> Optional[List[Dict[str, Any]]]:
        """
        일별 시세 데이터 조회
        
        특정 종목의 일별 시세 데이터를 조회합니다.
        
        Args:
            code: 종목 코드
            days: 조회할 일수
            
        Returns:
            list or None: 일별 시세 데이터 (최신 순) 또는 조회 실패 시 None
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        data = self.get_stock_data(code, period_div_code="D")
        if data and days > 0 and days < len(data):
            return data[:days]
        return data
        
    def get_weekly_data(self, code: str, weeks: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        주별 시세 데이터 조회
        
        특정 종목의 주별 시세 데이터를 조회합니다.
        
        Args:
            code: 종목 코드
            weeks: 조회할 주 수
            
        Returns:
            list or None: 주별 시세 데이터 (최신 순) 또는 조회 실패 시 None
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        data = self.get_stock_data(code, period_div_code="W")
        if data and weeks > 0 and weeks < len(data):
            return data[:weeks]
        return data
    
    def get_monthly_data(self, code: str, months: int = 12) -> Optional[List[Dict[str, Any]]]:
        """
        월별 시세 데이터 조회
        
        특정 종목의 월별 시세 데이터를 조회합니다.
        
        Args:
            code: 종목 코드
            months: 조회할 월 수
            
        Returns:
            list or None: 월별 시세 데이터 (최신 순) 또는 조회 실패 시 None
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        data = self.get_stock_data(code, period_div_code="M")
        if data and months > 0 and months < len(data):
            return data[:months]
        return data
        
    def get_stock_time_conclusion(self, code: str, time_interval: str = "1", count: int = 100) -> Optional[Dict[str, Any]]:
        """
        주식 당일 시간대별 체결 내역 조회
        
        한국투자증권 API를 통해 특정 종목의 당일 시간대별 체결 내역을 조회합니다.
        
        Args:
            code: 종목 코드
            time_interval: 시간 간격 (1: 1분, 3: 3분, 5: 5분, 10: 10분, 30: 30분, 60: 60분)
            count: 조회할 데이터 개수 (최대 100)
            
        Returns:
            dict or None: 시간대별 체결 내역 또는 조회 실패 시 None
            
        Notes:
            모의투자와 실전투자 모두 지원하는 함수입니다.
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # 설정 로드
        config = get_config()
        
        path = "uapi/domestic-stock/v1/quotations/inquire-time-conc"
        url = f"{config.current_api.base_url}/{path}"
        
        # 최대 100개로 제한
        if count > 100:
            count = 100
        
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": code,
            "fid_input_hour_1": time_interval,
            "fid_etc_cls_code": "",
            "fid_ord_div_code": "",
            "fid_input_price_1": "0",
            "fid_input_price_2": "0",
            "fid_vol_cnt": str(count),
            "fid_input_date_1": ""
        }
        
        headers = self._get_headers("FHPST01060000")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} {time_interval}분 시간대별 체결 내역 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} 시간대별 체결 내역 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            # 시간대별 체결 정보 추출
            output1 = result.get("output1", {})
            output2 = result.get("output2", [])
            
            if not output2:
                logger.warning(f"{code} 시간대별 체결 내역이 없습니다.")
                return {
                    "stock_code": code,
                    "stock_name": output1.get("hts_kor_isnm", ""),
                    "time_interval": time_interval,
                    "time_conclusions": []
                }
            
            # 현재가 및 기본 정보
            time_conclusion_info = {
                "stock_code": code,
                "stock_name": output1.get("hts_kor_isnm", ""),
                "current_price": int(output1.get("stck_prpr", "0").replace(',', '')),
                "time_interval": time_interval,
                "time_conclusions": []
            }
            
            # 시간대별 체결 내역 정보 추출
            for item in output2:
                time_data = {
                    "time": item.get("stck_cntg_hour", ""),
                    "close_price": int(item.get("stck_prpr", "0").replace(',', '')),
                    "open_price": int(item.get("stck_oprc", "0").replace(',', '')),
                    "high_price": int(item.get("stck_hgpr", "0").replace(',', '')),
                    "low_price": int(item.get("stck_lwpr", "0").replace(',', '')),
                    "volume": int(item.get("cntg_vol", "0").replace(',', '')),
                    "cumulative_volume": int(item.get("acml_vol", "0").replace(',', '')),
                    "change_price": int(item.get("prdy_vrss", "0").replace(',', '')),
                    "change_rate": float(item.get("prdy_ctrt", "0").replace(',', ''))
                }
                time_conclusion_info["time_conclusions"].append(time_data)
            
            logger.info(f"{code} {time_interval}분 시간대별 체결 내역 조회 성공: {len(time_conclusion_info['time_conclusions'])}건")
            return time_conclusion_info
            
        except Exception as e:
            logger.error(f"{code} 시간대별 체결 내역 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 시간대별 체결 내역 조회 실패: {e}", config.notification.discord_webhook_url)
            return None
    
    def get_stock_minute_data(self, code: str, minute_unit: str = "1", count: int = 100) -> Optional[Dict[str, Any]]:
        """
        주식 당일 분봉 데이터 조회
        
        Args:
            code (str): 종목 코드
            minute_unit (str): 분봉 단위 (1: 1분, 3: 3분, 5: 5분, 10: 10분, 30: 30분, 60: 60분)
            count (int): 조회할 데이터 개수 (최대 100)
            
        Returns:
            dict or None: 당일 분봉 데이터
            
        Notes:
            모의투자 지원 함수입니다.
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # 설정 로드
        config = get_config()
        
        path = "uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        url = f"{config.current_api.base_url}/{path}"
        
        # 최대 100개로 제한
        if count > 100:
            count = 100
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": "",  # 빈 값으로 두면 당일 데이터
            "FID_INPUT_DATE_2": "",
            "FID_PERIOD_DIV_CODE": minute_unit + "T",  # 1T, 3T, 5T, 10T, 30T, 60T
            "FID_ORG_ADJ_PRC": "0",
            "FID_CYCLE_DIV_CODE": "D", # D: 일봉, W: 주봉, M: 월봉
            "FID_DIV_CODE": "0",
            "FID_COMP_VALL": str(count),  # 요청 개수
        }
        
        headers = self._get_headers("FHKST03010200")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} {minute_unit}분봉 데이터 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} 분봉 데이터 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            # 분봉 데이터 추출
            output1 = result.get("output1", {})
            output2 = result.get("output2", [])
            
            if not output2:
                logger.warning(f"{code} 분봉 데이터가 없습니다.")
                return {
                    "stock_code": code,
                    "stock_name": output1.get("hts_kor_isnm", ""),
                    "minute_unit": minute_unit,
                    "minute_data": []
                }
            
            # 기본 정보
            minute_data_info = {
                "stock_code": code,
                "stock_name": output1.get("hts_kor_isnm", ""),
                "minute_unit": minute_unit,
                "minute_data": []
            }
            
            # 분봉 데이터 추출
            for item in output2:
                candle_data = {
                    "date": item.get("stck_bsop_date", ""),
                    "time": item.get("stck_cntg_hour", ""),
                    "open": int(item.get("stck_oprc", "0").replace(',', '')),
                    "high": int(item.get("stck_hgpr", "0").replace(',', '')),
                    "low": int(item.get("stck_lwpr", "0").replace(',', '')),
                    "close": int(item.get("stck_prpr", "0").replace(',', '')),
                    "volume": int(item.get("cntg_vol", "0").replace(',', '')),
                    "change_price": int(item.get("prdy_vrss", "0").replace(',', '')),
                    "change_rate": float(item.get("prdy_ctrt", "0").replace(',', ''))
                }
                minute_data_info["minute_data"].append(candle_data)
            
            logger.info(f"{code} {minute_unit}분봉 데이터 조회 성공: {len(minute_data_info['minute_data'])}건")
            return minute_data_info
            
        except Exception as e:
            logger.error(f"{code} 분봉 데이터 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 분봉 데이터 조회 실패: {e}", config.notification.discord_webhook_url)
            return None 