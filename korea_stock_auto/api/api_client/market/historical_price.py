"""
Historical Price API Mixin for Korea Investment Securities
한국투자증권 과거 시세 조회 API
"""

import logging
from typing import Dict, List, Optional, Any, cast, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

logger = logging.getLogger("stock_auto")

class HistoricalPriceMixin:
    """한국투자증권 과거 시세 조회 기능을 제공하는 Mixin 클래스"""
    
    def get_daily_stock_chart_data(self, symbol: str, start_date: str = None, end_date: str = None, 
                                   period_div_code: str = "D", adjust_price: str = "1", limit: int = 30) -> Optional[List[Dict[str, Any]]]:
        """
        주식 일자별/주별/월별 시세 조회 (한국투자증권 API: 주식현재가 일자별)
        TR_ID: FHKST01010400
        Method: GET
        Path: /uapi/domestic-stock/v1/quotations/inquire-daily-price
        
        Args:
            symbol (str): 종목 코드 (6자리)
            start_date (str, optional): 조회 시작일자 (YYYYMMDD). 미지정시 30일 전
            end_date (str, optional): 조회 종료일자 (YYYYMMDD). 미지정시 오늘
            period_div_code (str): 기간 분류 코드 
                - "D": 일봉 
                - "W": 주봉 
                - "M": 월봉
            adjust_price (str): 수정주가 반영 여부
                - "0": 수정주가 미반영
                - "1": 수정주가 반영 (기본값)
            limit (int): 조회 건수 제한 (기본: 30, 최대: 100)
        
        Returns:
            List[Dict[str, Any]] or None: 일자별 OHLCV 데이터 리스트 또는 오류 시 None
            
        Example:
            [
                {
                    "stck_bsop_date": "20240315",  # 영업일자
                    "stck_oprc": "75000",          # 시가
                    "stck_hgpr": "76000",          # 고가  
                    "stck_lwpr": "74500",          # 저가
                    "stck_clpr": "75500",          # 종가
                    "acml_vol": "1234567",         # 누적거래량
                    "prdy_vrss": "500",            # 전일대비
                    "prdy_ctrt": "0.67"            # 전일대비율
                },
                ...
            ]
        """

        # 설정 로드
        config = get_config()
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        try:
            # 날짜 설정 (미지정 시 기본값)
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            
            # API 요청 URL (config에서 실제 config.current_api.base_url 사용)
            from korea_stock_auto.config import get_config
            url = f"{config.current_api.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
            
            # Query Parameters (한국투자증권 API 문서 기준)
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",      # 조건 시장 분류 코드 (J: 주식)
                "FID_INPUT_ISCD": symbol,            # 입력 종목코드
                "FID_INPUT_DATE_1": start_date,      # 조회 시작일자
                "FID_INPUT_DATE_2": end_date,        # 조회 종료일자
                "FID_PERIOD_DIV_CODE": period_div_code,  # 기간 분류 코드 (D/W/M)
                "FID_ORG_ADJ_PRC": adjust_price,     # 수정가 원주가 구분 (0:원주가, 1:수정주가)
            }
            
            # 헤더 생성 (TR_ID: FHKST01010400)
            headers = self._get_headers("FHKST01010400")
            
            logger.info(f"일자별 시세 조회 요청: {symbol} ({start_date}~{end_date})")
            
            # GET 요청 수행
            response = self._request_get(url, headers, params, f"일자별 시세 조회 실패: {symbol}")
            
            if response and response.get("rt_cd") == "0":  # 성공
                output_data = response.get("output", [])  # output2 -> output으로 변경
                if output_data:
                    logger.info(f"일자별 시세 조회 성공: {symbol}, 데이터 건수: {len(output_data)}")
                    return output_data[:limit]  # limit 적용
                else:
                    logger.warning(f"일자별 시세 데이터 없음: {symbol}")
                    return []
            else:
                error_msg = response.get("msg1", "알 수 없는 오류") if response else "응답 없음"
                logger.error(f"일자별 시세 조회 실패: {symbol} - {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"일자별 시세 조회 중 오류 발생: {symbol} - {e}", exc_info=True)
            return None

    def get_intraday_minute_chart_data(self, symbol: str, time_interval: str = "5", 
                                       limit_count: Optional[int] = 120) -> Optional[List[Dict[str, Any]]]:
        """
        주식 분봉 데이터 조회 (한국투자증권 API: 주식 현재가 시간대별체결)
        TR_ID: FHKST01010400 (일자별 시세와 동일한 API 사용, 기간을 분으로 설정)
        Method: GET
        Path: /uapi/domestic-stock/v1/quotations/inquire-daily-price
        
        Args:
            symbol (str): 종목 코드 (6자리)
            time_interval (str): 분봉 간격 ("1", "5", "10", "30", "60")
            limit_count (Optional[int]): 조회할 분봉 개수 (기본: 120개, 하루 약 8시간)
        
        Returns:
            List[Dict[str, Any]] or None: 분봉 데이터 리스트 또는 오류 시 None
            
        Example:
            [
                {
                    "stck_bsop_date": "20240315",  # 영업일자
                    "stck_oprc": "75000",          # 시가
                    "stck_hgpr": "76000",          # 고가  
                    "stck_lwpr": "74500",          # 저가
                    "stck_clpr": "75500",          # 종가
                    "acml_vol": "1234567",         # 누적거래량
                },
                ...
            ]
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        try:
            # 현재 날짜만 조회 (분봉은 당일 데이터만 제공)
            today = datetime.now().strftime("%Y%m%d")
            
            # API 요청 URL (일자별 시세와 동일한 엔드포인트 사용)
            from korea_stock_auto.config import get_config
            config = get_config()
            url = f"{config.current_api.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
            
            # Query Parameters (분봉 조회용 파라미터)
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",      # 조건 시장 분류 코드 (J: 주식)
                "FID_INPUT_ISCD": symbol,            # 입력 종목코드
                "FID_INPUT_DATE_1": today,           # 조회 시작일자 (당일)
                "FID_INPUT_DATE_2": today,           # 조회 종료일자 (당일)
                "FID_PERIOD_DIV_CODE": time_interval,  # 기간 분류 코드 (분봉 간격)
                "FID_ORG_ADJ_PRC": "1",              # 수정가 원주가 구분 (1:수정주가)
            }
            
            # 헤더 생성 (TR_ID: FHKST01010400)
            headers = self._get_headers("FHKST01010400")
            
            logger.info(f"분봉 데이터 조회 요청: {symbol} ({time_interval}분봉)")
            
            # GET 요청 수행
            response = self._request_get(url, headers, params, f"분봉 데이터 조회 실패: {symbol}")
            
            if response and response.get("rt_cd") == "0":  # 성공
                output_data = response.get("output", [])  # output 사용
                if output_data:
                    logger.info(f"분봉 데이터 조회 성공: {symbol}, 데이터 건수: {len(output_data)}")
                    # limit_count 적용
                    return output_data[:limit_count] if limit_count else output_data
                else:
                    logger.warning(f"분봉 데이터 없음: {symbol}")
                    return []
            else:
                error_msg = response.get("msg1", "알 수 없는 오류") if response else "응답 없음"
                logger.error(f"분봉 데이터 조회 실패: {symbol} - {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"분봉 데이터 조회 중 오류 발생: {symbol} - {e}", exc_info=True)
            return None
