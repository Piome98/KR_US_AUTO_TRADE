"""
한국 주식 자동매매 - 시장 데이터 서비스

시장 데이터 관리를 담당합니다:
- 현재가 정보 수집 및 캐싱
- 주기적 데이터 업데이트
- 관심 종목 관리
- 시장 데이터 제공
"""

import logging
import time
from typing import Dict, List, Any, Optional

from korea_stock_auto.config import AppConfig
from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.trading.technical_analyzer import TechnicalAnalyzer
from korea_stock_auto.trading.stock_selector import StockSelector

# 도메인 엔터티 import
from korea_stock_auto.domain import Stock, Price, Money

# API 매퍼 통합
from korea_stock_auto.api.mappers import StockMapper, MappingError

logger = logging.getLogger(__name__)


class MarketDataService:
    """시장 데이터 관리 서비스 (API 매퍼 통합)"""
    
    def __init__(self, 
                 api: KoreaInvestmentApiClient, 
                 config: AppConfig,
                 analyzer: TechnicalAnalyzer,
                 selector: StockSelector):
        """
        시장 데이터 서비스 초기화
        
        Args:
            api: 한국투자증권 API 클라이언트
            config: 애플리케이션 설정
            analyzer: 기술적 분석기
            selector: 종목 선택기
        """
        self.api = api
        self.config = config
        self.analyzer = analyzer
        self.selector = selector
        
        # API 매퍼 초기화
        self.stock_mapper = StockMapper(
            enable_cache=True,
            cache_ttl_seconds=30  # 주식 시세 데이터는 30초 캐시
        )
        
        # 시장 데이터 상태 (도메인 엔터티 사용)
        self.stocks: Dict[str, Stock] = {}  # 종목코드 -> Stock 엔터티
        self.symbol_list: List[str] = []
        
        # 백워드 호환성을 위한 기존 인터페이스 유지
        self.current_prices: Dict[str, Dict[str, Any]] = {}
        
        # 업데이트 관리
        self.last_market_data_fetch_time: float = 0.0
        self.market_data_fetch_interval: int = config.system.data_update_interval
        
        logger.debug("MarketDataService 초기화 완료 (API 매퍼 통합)")
    
    def initialize(self) -> bool:
        """
        시장 데이터 서비스 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            logger.info("시장 데이터 서비스 초기화 시작")
            # 초기화 로직이 필요하면 여기에 추가
            logger.info("시장 데이터 서비스 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"시장 데이터 서비스 초기화 실패: {e}")
            return False
    
    def select_interest_stocks(self, target_count: Optional[int] = None) -> List[str]:
        """
        관심 종목 선정
        
        Args:
            target_count: 목표 종목 수 (None이면 config 사용)
            
        Returns:
            List[str]: 선정된 관심 종목 코드 리스트
        """
        if target_count is None:
            target_count = self.config.trading.target_buy_count
        
        try:
            self.symbol_list = self.selector.select_interest_stocks(target_count)
            
            if self.symbol_list:
                logger.info(f"관심 종목 {len(self.symbol_list)}개 선정 완료: {self.symbol_list}")
            else:
                logger.warning("관심 종목 선정 실패")
                
            return self.symbol_list
            
        except Exception as e:
            logger.error(f"관심 종목 선정 중 오류: {e}")
            return []
    
    def update_market_data(self, target_symbols: List[str], force: bool = False) -> bool:
        """
        시장 데이터 업데이트 (API 매퍼 통합)
        
        Args:
            target_symbols: 업데이트할 종목 리스트
            force: 강제 업데이트 여부 (interval 무시)
            
        Returns:
            bool: 업데이트 성공 여부
        """
        current_time = time.time()
        
        if not force and current_time - self.last_market_data_fetch_time < self.market_data_fetch_interval:
            return True
        
        if not target_symbols:
            logger.debug("업데이트할 종목이 없습니다.")
            return True
        
        logger.info(f"시장 데이터 업데이트 시작: {len(target_symbols)}개 종목")
        
        success_count = 0
        for code in target_symbols:
            try:
                if self._update_symbol_data_with_mapper(code, current_time):
                    success_count += 1
            except Exception as e:
                logger.warning(f"{code} 데이터 업데이트 실패: {e}")
        
        self.last_market_data_fetch_time = current_time
        
        success_rate = success_count / len(target_symbols) if target_symbols else 0
        logger.info(f"시장 데이터 업데이트 완료: {success_count}/{len(target_symbols)} ({success_rate:.1%})")
        
        return success_rate > 0.5  # 50% 이상 성공하면 성공으로 간주
    
    def _update_symbol_data_with_mapper(self, code: str, timestamp: float) -> bool:
        """
        특정 종목의 데이터 업데이트 (API 매퍼 사용)
        
        Args:
            code: 종목 코드
            timestamp: 타임스탬프
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            # 현재가 정보 API 호출
            price_data = self.api.get_current_price(code)
            if not price_data or not isinstance(price_data, dict):
                logger.warning(f"{code} 현재가 정보 조회 실패")
                return self._fallback_to_legacy_update(code, timestamp)
            
            # 🔧 이미 가공된 데이터로 직접 Stock 엔터티 생성 (StockMapper 우회)
            from korea_stock_auto.domain.entities import Stock
            from korea_stock_auto.domain.value_objects import Price, Money, Quantity
            from datetime import datetime
            
            try:
                # 🔧 주식명 검증 및 기본값 설정
                stock_name = price_data.get('stock_name', '').strip()
                if not stock_name:
                    stock_name = f"종목_{code}"  # 주식명이 없으면 기본값 사용
                    logger.debug(f"{code} 주식명이 비어있어 기본값 사용: {stock_name}")
                
                stock = Stock(
                    code=code,
                    name=stock_name,
                    current_price=Price(price_data.get('current_price', 0)),
                    previous_close=Price(price_data.get('prev_close_price', 0)),
                    market_cap=Money(price_data.get('market_cap', 0)),
                    volume=Quantity(price_data.get('volume', 0)),
                    updated_at=datetime.now()
                )
                
                # Stock 엔터티 저장
                self.stocks[code] = stock
                
                # 백워드 호환성을 위한 데이터 동기화
                self._sync_legacy_data_from_stock(stock, timestamp)
                
                logger.debug(f"{code} Stock 엔터티 업데이트 완료: {stock.current_price}")
                
            except MappingError as e:
                logger.warning(f"{code} Stock 매핑 실패, 기존 방식 사용: {e}")
                return self._fallback_to_legacy_update(code, timestamp, price_data)
            
            # 캐시된 데이터 확인 - 충분한 데이터가 있으면 추가 크롤링 하지 않음
            cached_data = self.analyzer._get_cached_ohlcv(code, 'D')
            if cached_data is not None and len(cached_data) >= 200:
                logger.debug(f"{code} 충분한 캐시 데이터 존재 ({len(cached_data)}개), 추가 크롤링 생략")
                # 기술적 지표만 업데이트 (데이터 크롤링 없이)
                self.analyzer._update_indicators_only(code, 'D')
            else:
                # 캐시 데이터가 부족하면 전체 데이터 크롤링
                logger.debug(f"{code} 캐시 데이터 부족, 전체 데이터 크롤링 실행")
                # analyzer의 메서드를 통해 OHLCV 데이터 및 지표 업데이트
                # 실제 구현은 analyzer의 메서드를 호출
            
            return True
            
        except Exception as e:
            logger.error(f"{code} 데이터 업데이트 중 오류: {e}")
            return False
    
    def _sync_legacy_data_from_stock(self, stock: Stock, timestamp: float) -> None:
        """Stock 엔터티에서 백워드 호환성 데이터 동기화"""
        code = stock.code
        current_price = int(stock.current_price.value.to_float())
        
        self.current_prices[code] = {
            'price': current_price,
            'bid': current_price,  # 실제로는 호가 정보 필요
            'ask': current_price,  # 실제로는 호가 정보 필요
            'volume': stock.volume.value if stock.volume else 0,
            'timestamp': timestamp
        }
    
    def _fallback_to_legacy_update(self, code: str, timestamp: float, price_data: Optional[Dict[str, Any]] = None) -> bool:
        """기존 방식의 데이터 업데이트 (백워드 호환성)"""
        try:
            if price_data and 'current_price' in price_data:
                current_price = int(price_data['current_price'])
                self.current_prices[code] = {
                    'price': current_price,
                    'bid': int(price_data.get('lowest_bid', 0)),
                    'ask': int(price_data.get('highest_ask', 0)),
                    'volume': int(price_data.get('volume', 0)),
                    'timestamp': timestamp
                }
                logger.debug(f"{code} 현재가 업데이트 (legacy): {current_price:,}")
                return True
            else:
                # 기존 데이터가 있으면 타임스탬프만 업데이트
                if code in self.current_prices:
                    self.current_prices[code]['timestamp'] = timestamp
                else:
                    self.current_prices[code] = {'price': 0, 'timestamp': timestamp}
                return False
        except Exception as e:
            logger.error(f"{code} Legacy 업데이트 실패: {e}")
            return False
    
    def _update_symbol_data(self, code: str, timestamp: float) -> bool:
        """
        특정 종목의 데이터 업데이트 (기존 방식, 백워드 호환성)
        
        Args:
            code: 종목 코드
            timestamp: 타임스탬프
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            # 현재가 정보 업데이트
            price_data = self.api.get_current_price(code)
            if price_data and isinstance(price_data, dict) and 'current_price' in price_data:
                current_price = int(price_data['current_price'])
                self.current_prices[code] = {
                    'price': current_price,
                    'bid': int(price_data.get('lowest_bid', 0)),
                    'ask': int(price_data.get('highest_ask', 0)),
                    'volume': int(price_data.get('volume', 0)),
                    'timestamp': timestamp
                }
                logger.debug(f"{code} 현재가 업데이트: {current_price:,}")
            else:
                logger.warning(f"{code} 현재가 정보 조회 실패 또는 형식 오류")
                # 기존 데이터가 있으면 타임스탬프만 업데이트
                if code in self.current_prices:
                    self.current_prices[code]['timestamp'] = timestamp
                else:
                    self.current_prices[code] = {'price': 0, 'timestamp': timestamp}
                return False

            # 캐시된 데이터 확인 - 충분한 데이터가 있으면 추가 크롤링 하지 않음
            cached_data = self.analyzer._get_cached_ohlcv(code, 'D')
            if cached_data is not None and len(cached_data) >= 200:
                logger.debug(f"{code} 충분한 캐시 데이터 존재 ({len(cached_data)}개), 추가 크롤링 생략")
                # 기술적 지표만 업데이트 (데이터 크롤링 없이)
                self.analyzer._update_indicators_only(code, 'D')
            else:
                # 캐시 데이터가 부족하면 전체 데이터 크롤링
                logger.debug(f"{code} 캐시 데이터 부족, 전체 데이터 크롤링 실행")
                # analyzer의 메서드를 통해 OHLCV 데이터 및 지표 업데이트
                # 실제 구현은 analyzer의 메서드를 호출
            
            return True
            
        except Exception as e:
            logger.error(f"{code} 데이터 업데이트 중 오류: {e}")
            return False
    
    def get_stock(self, code: str) -> Optional[Stock]:
        """
        특정 종목의 Stock 엔터티 조회 (NEW)
        
        Args:
            code: 종목 코드
            
        Returns:
            Stock: Stock 엔터티 (없으면 None)
        """
        return self.stocks.get(code)
    
    def get_stocks(self, codes: List[str] = None) -> Dict[str, Stock]:
        """
        Stock 엔터티 조회 (여러 종목, NEW)
        
        Args:
            codes: 종목 코드 리스트 (None이면 전체)
            
        Returns:
            Dict: 종목코드 -> Stock 엔터티
        """
        if codes is None:
            return dict(self.stocks)
        
        return {code: stock for code, stock in self.stocks.items() if code in codes}
    
    def get_current_price(self, code: str) -> Optional[Dict[str, Any]]:
        """
        특정 종목의 현재가 정보 조회 (백워드 호환성)
        
        Args:
            code: 종목 코드
            
        Returns:
            Dict: 현재가 정보 (없으면 None)
        """
        return self.current_prices.get(code)
    
    def get_current_prices(self, codes: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        현재가 정보 조회 (여러 종목, 백워드 호환성)
        
        Args:
            codes: 종목 코드 리스트 (None이면 전체)
            
        Returns:
            Dict: 종목별 현재가 정보
        """
        if codes is None:
            return dict(self.current_prices)
        
        return {code: data for code, data in self.current_prices.items() if code in codes}
    
    def get_price_only(self, code: str) -> float:
        """
        특정 종목의 현재가만 반환 (백워드 호환성)
        
        Args:
            code: 종목 코드
            
        Returns:
            float: 현재가 (데이터 없으면 0)
        """
        # 도메인 엔터티에서 먼저 조회 시도
        stock = self.stocks.get(code)
        if stock:
            return stock.current_price.value.to_float()
        
        # 백워드 호환성: 기존 데이터에서 조회
        price_info = self.current_prices.get(code, {})
        return price_info.get('price', 0)
    
    def has_valid_price(self, code: str) -> bool:
        """
        특정 종목의 유효한 가격 데이터 보유 여부
        
        Args:
            code: 종목 코드
            
        Returns:
            bool: 유효한 가격 데이터 보유 여부
        """
        # 도메인 엔터티에서 확인
        stock = self.stocks.get(code)
        if stock and not stock.current_price.is_zero():
            return True
        
        # 백워드 호환성: 기존 데이터에서 확인
        price_info = self.current_prices.get(code, {})
        return price_info.get('price', 0) > 0
    
    def get_interest_stocks(self) -> List[str]:
        """
        현재 관심 종목 리스트 반환
        
        Returns:
            List[str]: 관심 종목 코드 리스트
        """
        return self.symbol_list.copy()
    
    def add_interest_stock(self, code: str) -> None:
        """
        관심 종목 추가
        
        Args:
            code: 추가할 종목 코드
        """
        if code not in self.symbol_list:
            self.symbol_list.append(code)
            logger.info(f"관심 종목 추가: {code}")
    
    def remove_interest_stock(self, code: str) -> bool:
        """
        관심 종목 제거
        
        Args:
            code: 제거할 종목 코드
            
        Returns:
            bool: 제거 성공 여부
        """
        try:
            if code in self.symbol_list:
                self.symbol_list.remove(code)
                logger.info(f"관심 종목 제거: {code}")
                return True
            return False
        except Exception as e:
            logger.error(f"관심 종목 제거 중 오류: {e}")
            return False
    
    def get_market_data_summary(self) -> Dict[str, Any]:
        """
        시장 데이터 요약 정보 반환
        
        Returns:
            Dict: 시장 데이터 요약
        """
        return {
            "interest_stocks_count": len(self.symbol_list),
            "interest_stocks": self.symbol_list.copy(),
            "cached_stocks_count": len(self.stocks),
            "cached_prices_count": len(self.current_prices),
            "last_update_time": self.last_market_data_fetch_time,
            "update_interval": self.market_data_fetch_interval,
            "stock_mapper_cache_stats": self.stock_mapper.get_cache_stats()
        }
    
    def force_refresh_symbol_data(self, code: str) -> bool:
        """
        특정 종목 데이터 강제 새로고침
        
        Args:
            code: 종목 코드
            
        Returns:
            bool: 새로고침 성공 여부
        """
        try:
            return self._update_symbol_data_with_mapper(code, time.time())
        except Exception as e:
            logger.error(f"{code} 강제 새로고침 실패: {e}")
            return False
    
    def clear_mapper_cache(self) -> None:
        """매퍼 캐시 전체 삭제"""
        self.stock_mapper.clear_cache()
        logger.debug("StockMapper 캐시 삭제 완료")
    
    def get_mapper_cache_stats(self) -> Dict[str, Any]:
        """매퍼 캐시 통계 조회"""
        return {
            'stock_mapper': self.stock_mapper.get_cache_stats()
        }
    
    def force_update_all_symbols(self) -> bool:
        """모든 관심 종목 강제 업데이트"""
        if not self.symbol_list:
            logger.warning("관심 종목이 없어 업데이트할 수 없습니다")
            return False
        
        return self.update_market_data(self.symbol_list, force=True)
    
    def get_top_traded_stocks(self, market_type: str = "0", top_n: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        거래량 상위 종목 조회
        
        Args:
            market_type: 시장 구분 ("0": 전체, "1": 코스피, "2": 코스닥)  
            top_n: 조회할 상위 종목 수
            
        Returns:
            list or None: 거래량 상위 종목 목록
        """
        try:
            # StockInfoMixin을 통해 거래량 상위 종목 조회
            from korea_stock_auto.api.api_client.market.stock_info import StockInfoMixin
            
            # Mixin 메서드를 직접 호출
            return StockInfoMixin.get_top_traded_stocks(self.api, market_type, top_n)
                
        except Exception as e:
            logger.error(f"거래량 상위 종목 조회 실패: {e}")
            return None 