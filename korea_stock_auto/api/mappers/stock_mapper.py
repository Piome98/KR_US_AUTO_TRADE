"""
주식 API 응답 매퍼

주식 관련 API 응답을 Stock 도메인 엔터티로 변환합니다:
- 현재가 조회 응답
- 주식 검색 응답
- 호가 정보 응답
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .base_mapper import BaseMapper, MappingError
from korea_stock_auto.domain.entities import Stock
from korea_stock_auto.domain.value_objects import Price, Money, Quantity

logger = logging.getLogger(__name__)


class StockMapper(BaseMapper[Stock]):
    """주식 API 응답 → Stock 엔터티 변환 매퍼"""
    
    def __init__(self, enable_cache: bool = True, cache_ttl_seconds: int = 30):
        """
        주식 매퍼 초기화
        
        Args:
            enable_cache: 캐시 사용 여부
            cache_ttl_seconds: 캐시 유효 시간 (주식 데이터는 30초)
        """
        super().__init__(enable_cache, cache_ttl_seconds)
    
    def map_single(self, api_response: Dict[str, Any]) -> Stock:
        """
        단일 주식 API 응답을 Stock 엔터티로 변환
        
        Args:
            api_response: 주식 API 응답 데이터
            
        Returns:
            Stock: 변환된 Stock 엔터티
            
        Raises:
            MappingError: 매핑 실패 시
        """
        try:
            # 응답 구조 확인 및 데이터 추출
            stock_data = self._extract_stock_data(api_response)
            
            # 필수 필드 검증
            self.validate_api_response(stock_data, ['code', 'name'])
            
            # Stock 엔터티 생성
            stock = Stock(
                code=self._normalize_stock_code(stock_data['code']),
                name=self.safe_get_value(stock_data, 'name', '', str),
                current_price=self._create_price_from_response(stock_data, 'current_price'),
                previous_close=self._create_price_from_response(stock_data, 'previous_close'),
                market_cap=self._create_money_from_response(stock_data, 'market_cap'),
                volume=self._create_quantity_from_response(stock_data, 'volume'),
                updated_at=datetime.now()
            )
            
            logger.debug(f"Stock 엔터티 생성 성공: {stock.code} - {stock.name}")
            return stock
            
        except Exception as e:
            logger.error(f"Stock 매핑 실패: {e}")
            raise MappingError(f"Stock 매핑 실패: {e}", api_response)
    
    def map_from_current_price_response(self, api_response: Dict[str, Any]) -> Stock:
        """
        현재가 조회 API 응답에서 Stock 엔터티 생성
        
        API 응답 구조:
        {
            "rt_cd": "0",
            "output": {
                "iscd_stat_cls_code": "55",  # 종목상태구분코드
                "marg_rate": "40.00",        # 증거금비율
                "rprs_mrkt_kor_name": "코스피", # 대표시장한글명
                "new_hgpr_lwpr_cls_code": "1", # 신고가저가구분코드
                "bstp_kor_isnm": "삼성전자",     # 업종한글종목명
                "temp_stop_yn": "N",          # 임시정지여부
                "oprc_rang_cont_yn": "N",     # 가격범위제한연속여부
                "clpr_vs_oprc_sign": "2",     # 종가대비시가부호
                "clpr_vs_oprc": "-1500",      # 종가대비시가
                "cndc_diff_rmn_rate": "-1.99", # 전일대비등락율
                "oprc_rang_cont_yn": "N",      # 가격범위제한연속여부
                "stck_prpr": "74200",          # 주식현재가
                "stck_oprc": "75700",          # 주식시가
                "stck_hgpr": "75800",          # 주식최고가
                "stck_lwpr": "74100",          # 주식최저가
                "stck_prdy_clpr": "75700",     # 주식전일종가
                "acml_vol": "10039467",        # 누적거래량
                "acml_tr_pbmn": "750090086050", # 누적거래대금
                "ssts_yn": "N",                # 정지여부
                "stck_fcam": "0",              # 주식액면가
                "stck_sspr": "0",              # 주식대용가
                "hts_kor_isnm": "삼성전자",      # HTS한글종목명
                "stck_prdy_clpr": "75700"       # 주식전일종가
            }
        }
        """
        try:
            # 🔍 API 응답 구조 디버깅
            logger.debug(f"현재가 응답 매핑 - 전체 응답: {api_response}")
            logger.debug(f"현재가 응답 키: {list(api_response.keys()) if isinstance(api_response, dict) else 'dict가 아님'}")
            
            output = api_response.get('output', {})
            if not output:
                # output이 없을 때 대안 필드 확인
                logger.warning(f"output 필드가 없습니다. 응답 키: {list(api_response.keys()) if isinstance(api_response, dict) else 'N/A'}")
                
                # output1, output2 등 다른 필드명 확인
                for key in api_response.keys():
                    if isinstance(api_response[key], dict) and ('stck_prpr' in api_response[key] or 'prpr' in api_response[key]):
                        logger.info(f"output 대신 '{key}' 필드 사용")
                        output = api_response[key]
                        break
                
                if not output:
                    raise MappingError("output 필드가 없습니다", api_response)
            
            # 기본 정보 추출 - 종목코드 우선순위: pdno -> code -> iscd_stat_cls_code
            code = self.safe_get_value(output, 'pdno', '', str)
            if not code:
                code = self.safe_get_value(output, 'code', '', str)
            if not code:
                code = self.safe_get_value(output, 'iscd_stat_cls_code', '', str)
            if not code:
                # API 응답 최상위에서 종목코드 찾기 시도
                code = self.safe_get_value(api_response, 'code', '', str)
            
            name = self.safe_get_value(output, 'hts_kor_isnm', 
                                     self.safe_get_value(output, 'bstp_kor_isnm', ''), str)
            
            # 가격 정보 추출
            current_price = self.safe_get_value(output, 'stck_prpr', 0, int)
            previous_close = self.safe_get_value(output, 'stck_prdy_clpr', 0, int)
            
            # 거래량 정보
            volume = self.safe_get_value(output, 'acml_vol', 0, int)
            
            # 시가총액 계산 (현재가 × 상장주식수, 정확한 상장주식수가 없으므로 0으로 설정)
            market_cap = 0  # API에서 시가총액 직접 제공하지 않음
            
            stock_data = {
                'code': code,
                'name': name,
                'current_price': current_price,
                'previous_close': previous_close,
                'volume': volume,
                'market_cap': market_cap
            }
            
            # 캐시를 사용한 매핑 (종목코드를 캐시 키로 사용)
            cache_key = f"stock_{code}" if code else None
            return self.map_with_cache({'stock_data': stock_data}, cache_key)
            
        except Exception as e:
            logger.error(f"현재가 응답 매핑 실패: {e}")
            raise MappingError(f"현재가 응답 매핑 실패: {e}", api_response)
    
    def map_from_balance_response(self, balance_item: Dict[str, Any]) -> Stock:
        """
        계좌 잔고 응답의 보유 종목에서 Stock 엔터티 생성
        
        Args:
            balance_item: 계좌 잔고 API의 개별 종목 데이터
        """
        try:
            stock_data = {
                'code': self.safe_get_value(balance_item, 'pdno', '', str),  # 종목코드
                'name': self.safe_get_value(balance_item, 'prdt_name', '', str),  # 종목명
                'current_price': self.safe_get_value(balance_item, 'prpr', 0, int),  # 현재가
                'previous_close': self.safe_get_value(balance_item, 'prdy_pr', 0, int),  # 전일가
                'volume': 0,  # 보유종목 정보에서는 거래량 미제공
                'market_cap': 0  # 시가총액 미제공
            }
            
            # 캐시를 사용한 매핑 (종목코드를 캐시 키로 사용)
            cache_key = f"balance_stock_{stock_data['code']}" if stock_data['code'] else None
            return self.map_with_cache({'stock_data': stock_data}, cache_key)
            
        except Exception as e:
            logger.error(f"잔고 응답 매핑 실패: {e}")
            raise MappingError(f"잔고 응답 매핑 실패: {e}", balance_item)
    
    def _extract_stock_data(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """API 응답에서 주식 데이터 추출"""
        # 직접 주식 데이터가 있는 경우
        if 'stock_data' in api_response:
            return api_response['stock_data']
        
        # output 구조인 경우
        if 'output' in api_response:
            return self._extract_from_output_structure(api_response['output'])
        
        # 직접 필드들이 있는 경우
        return api_response
    
    def _extract_from_output_structure(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """output 구조에서 주식 데이터 추출"""
        return {
            'code': self.safe_get_value(output, 'pdno', 
                   self.safe_get_value(output, 'code', ''), str),
            'name': self.safe_get_value(output, 'prdt_name',
                   self.safe_get_value(output, 'hts_kor_isnm', ''), str),
            'current_price': self.safe_get_value(output, 'prpr',
                            self.safe_get_value(output, 'stck_prpr', 0), int),
            'previous_close': self.safe_get_value(output, 'prdy_pr',
                             self.safe_get_value(output, 'stck_prdy_clpr', 0), int),
            'volume': self.safe_get_value(output, 'acml_vol', 0, int),
            'market_cap': 0  # API에서 직접 제공하지 않음
        }
    
    def _normalize_stock_code(self, code: str) -> str:
        """종목코드 정규화 (6자리 숫자로)"""
        if not code:
            raise MappingError("종목코드가 비어있습니다")
        
        # 숫자만 추출
        numeric_code = ''.join(filter(str.isdigit, str(code)))
        
        if len(numeric_code) != 6:
            raise MappingError(f"유효하지 않은 종목코드: {code}")
        
        return numeric_code
    
    def _create_price_from_response(self, data: Dict[str, Any], field: str) -> Price:
        """응답 데이터에서 Price 객체 생성"""
        amount = self.safe_get_value(data, field, 0, int)
        return Price(Money.won(amount))
    
    def _create_money_from_response(self, data: Dict[str, Any], field: str) -> Money:
        """응답 데이터에서 Money 객체 생성"""
        amount = self.safe_get_value(data, field, 0, int)
        return Money.won(amount)
    
    def _create_quantity_from_response(self, data: Dict[str, Any], field: str) -> Quantity:
        """응답 데이터에서 Quantity 객체 생성"""
        value = self.safe_get_value(data, field, 0, int)
        return Quantity(value) 