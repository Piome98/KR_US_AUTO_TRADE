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

logger = logging.getLogger(__name__)


class MarketDataService:
    """시장 데이터 관리 서비스"""
    
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
        
        # 시장 데이터 상태
        self.current_prices: Dict[str, Dict[str, Any]] = {}
        self.symbol_list: List[str] = []
        
        # 업데이트 관리
        self.last_market_data_fetch_time: float = 0.0
        self.market_data_fetch_interval: int = config.system.data_update_interval
        
        logger.debug("MarketDataService 초기화 완료")
    
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
        시장 데이터 업데이트
        
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
                if self._update_symbol_data(code, current_time):
                    success_count += 1
            except Exception as e:
                logger.warning(f"{code} 데이터 업데이트 실패: {e}")
        
        self.last_market_data_fetch_time = current_time
        
        success_rate = success_count / len(target_symbols) if target_symbols else 0
        logger.info(f"시장 데이터 업데이트 완료: {success_count}/{len(target_symbols)} ({success_rate:.1%})")
        
        return success_rate > 0.5  # 50% 이상 성공하면 성공으로 간주
    
    def _update_symbol_data(self, code: str, timestamp: float) -> bool:
        """
        특정 종목의 데이터 업데이트
        
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
    
    def get_current_price(self, code: str) -> Optional[Dict[str, Any]]:
        """
        특정 종목의 현재가 정보 조회
        
        Args:
            code: 종목 코드
            
        Returns:
            Dict: 현재가 정보 (없으면 None)
        """
        return self.current_prices.get(code)
    
    def get_current_prices(self, codes: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        현재가 정보 조회 (여러 종목)
        
        Args:
            codes: 종목 코드 리스트 (None이면 전체)
            
        Returns:
            Dict: 현재가 정보 딕셔너리
        """
        if codes is None:
            return self.current_prices.copy()
        
        return {code: self.current_prices[code] for code in codes if code in self.current_prices}
    
    def get_price_only(self, code: str) -> float:
        """
        특정 종목의 현재가만 조회
        
        Args:
            code: 종목 코드
            
        Returns:
            float: 현재가 (없으면 0)
        """
        price_info = self.current_prices.get(code, {})
        return price_info.get('price', 0)
    
    def has_valid_price(self, code: str) -> bool:
        """
        특정 종목의 유효한 현재가 정보 보유 여부
        
        Args:
            code: 종목 코드
            
        Returns:
            bool: 유효한 가격 정보 보유 여부
        """
        price_info = self.current_prices.get(code)
        return price_info is not None and price_info.get('price', 0) > 0
    
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
            code: 종목 코드
        """
        if code not in self.symbol_list:
            self.symbol_list.append(code)
            logger.info(f"관심 종목 추가: {code}")
    
    def remove_interest_stock(self, code: str) -> bool:
        """
        관심 종목 제거
        
        Args:
            code: 종목 코드
            
        Returns:
            bool: 제거 성공 여부
        """
        if code in self.symbol_list:
            self.symbol_list.remove(code)
            logger.info(f"관심 종목 제거: {code}")
            return True
        return False
    
    def get_market_data_summary(self) -> Dict[str, Any]:
        """
        시장 데이터 요약 정보
        
        Returns:
            Dict: 요약 정보
        """
        valid_prices = sum(1 for info in self.current_prices.values() if info.get('price', 0) > 0)
        
        return {
            "interest_stocks_count": len(self.symbol_list),
            "tracked_symbols": len(self.current_prices),
            "valid_prices": valid_prices,
            "last_update": self.last_market_data_fetch_time,
            "update_interval": self.market_data_fetch_interval
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
            return self._update_symbol_data(code, time.time())
        except Exception as e:
            logger.error(f"{code} 강제 새로고침 실패: {e}")
            return False 