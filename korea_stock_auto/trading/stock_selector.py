"""
한국 주식 자동매매 - 종목 선정 모듈
관심 종목 선정 관련 클래스 및 함수 정의
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional

from korea_stock_auto.config import STOCK_FILTER, EXCLUDE_ETF
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.trading.technical_analyzer import TechnicalAnalyzer

logger = logging.getLogger("stock_auto")

class StockSelector:
    """국내 주식 관심 종목 선정 클래스"""
    
    def __init__(self, api_client: KoreaInvestmentApiClient):
        """
        종목 선정기 초기화
        
        Args:
            api_client: API 클라이언트 인스턴스
        """
        self.api = api_client
        self.score_threshold = STOCK_FILTER["score_threshold"]
        self.market_cap_threshold = STOCK_FILTER["market_cap_threshold"]
        self.price_threshold = STOCK_FILTER["price_threshold"]
        self.monthly_volatility_threshold = STOCK_FILTER["monthly_volatility_threshold"]
        self.trade_volume_increase_ratio = STOCK_FILTER["trade_volume_increase_ratio"]
        self.close_price_increase_ratio = STOCK_FILTER["close_price_increase_ratio"]
        # 기술적 분석기 초기화
        self.tech_analyzer = TechnicalAnalyzer(api_client)
    
    def select_interest_stocks(self, target_count: int) -> List[str]:
        """
        거래량 상위 종목 중에서 특정 조건을 만족하는 관심 종목을 선정하는 함수
        
        Args:
            target_count: 선정할 종목 수
            
        Returns:
            list: 선정된 관심 종목 코드 리스트
        """
        send_message("관심 종목 선정 시작")
        
        # API를 통해 거래량 상위 종목 조회
        logger.debug("거래량 상위 종목 조회 시도")
        top_stocks = self.api.get_top_traded_stocks(market_type="1", top_n=50)  # 코스피 시장 50개로 증가
        
        if not top_stocks:
            logger.error("거래량 상위 종목 조회 실패 - API 응답 없음")
            send_message("관심 종목 없음 (API 응답 실패)")
            return []
        
        logger.debug(f"거래량 상위 종목 {len(top_stocks)}개 조회 성공")
        send_message(f"거래량 상위 종목 {len(top_stocks)}개 조회 성공")
        
        # ETF 필터링
        filtered_stocks = []
        for stock in top_stocks:
            code = stock["code"]
            name = stock["name"]
            
            # ETF 종목 제외 (설정에 따라)
            if EXCLUDE_ETF and self.api.is_etf_stock(code):
                logger.info(f"ETF 종목 제외: {name} ({code})")
                continue
                
            filtered_stocks.append(stock)
        
        if not filtered_stocks:
            logger.warning("ETF를 제외한 거래량 상위 종목이 없습니다.")
            send_message("ETF를 제외한 관심 종목 없음")
            return []
            
        logger.info(f"ETF 제외 후 {len(filtered_stocks)}개 종목 선정됨")
        send_message(f"ETF 제외 후 {len(filtered_stocks)}개 종목 선정됨")
        
        # 점수 계산 로직 활성화
        scored_stocks = []
        
        for stock in filtered_stocks:
            code = stock["code"]
            name = stock["name"]
            current_price = stock.get("price", 0)
            market_cap = stock["market_cap"]
            volume = stock["volume"]
            
            # 기본 점수
            score = 0
            
            # 1. 시가총액/가격 조건
            if market_cap > 1000000000000 and current_price > 5000:  # 1조원 이상, 5천원 이상
                score += 1
            
            # 2. 거래량 조건
            if volume > 1000000:  # 백만주 이상 거래
                score += 1
            
            # 3. 이동평균선 조건
            ma20 = self.tech_analyzer.get_moving_average(code, window=20)
            if ma20 and current_price > ma20 * 1.01:  # 20일 이평선 1% 이상 상회
                score += 1
            
            # 4. 변동성 조건
            volatility = self.tech_analyzer.calculate_volatility(code)
            if volatility and 0.2 <= volatility <= 0.5:  # 적정 변동성 범위
                score += 1
            
            # 5. 골든 크로스 조건
            is_golden = self.tech_analyzer.is_golden_cross(code)
            if is_golden:
                score += 2
            
            # 6. 볼린저 밴드 조건
            bb = self.tech_analyzer.calculate_bollinger_bands(code)
            if bb and current_price > bb["middle"] and current_price < bb["upper"]:
                score += 1
            
            # 점수와 함께 저장
            scored_stocks.append({
                "code": code,
                "name": name,
                "score": score,
                "current_price": current_price
            })
        
        # 점수 기준으로 정렬
        scored_stocks.sort(key=lambda x: x["score"], reverse=True)
        
        # 상위 종목 선정
        interest_stocks = []
        for stock in scored_stocks[:target_count]:
            if stock["score"] >= 2:  # 최소 2점 이상인 종목만 선정
                interest_stocks.append(stock["code"])
                logger.debug(f"거래량 상위 종목으로 선정: {stock['name']} ({stock['code']}) - 점수: {stock['score']}")
        
        # 선정된 종목이 없으면 점수 상위 종목 선택
        if not interest_stocks and scored_stocks:
            for stock in scored_stocks[:min(target_count, len(scored_stocks))]:
                interest_stocks.append(stock["code"])
                logger.debug(f"거래량 상위 종목으로 선정: {stock['name']} ({stock['code']}) - 점수: {stock['score']}")
        
        send_message(f"관심 종목 {len(interest_stocks)}개 선정 완료")
        logger.info(f"관심 종목 {len(interest_stocks)}개 선정 완료")
        
        return interest_stocks
    
    def _calculate_stock_score(self, code: str, stock_name: str, current_price: float, market_cap: float) -> int:
        """
        종목의 점수를 계산하는 내부 함수
        
        Args:
            code: 종목 코드
            stock_name: 종목명
            current_price: 현재가
            market_cap: 시가총액
            
        Returns:
            int: 종목 점수
        """
        score = 0
        
        # 시가총액 및 가격 기준 점수
        if (market_cap >= self.market_cap_threshold and 
            current_price >= self.price_threshold):
            score += 1
            logger.debug(f"{stock_name}({code}) - 시가총액/가격 조건 만족 (+1점)")

        # 월간 변동성 점수
        monthly_data = self.api.get_monthly_data(code)
        daily_data = self.api.get_daily_data(code)
        if not monthly_data or not daily_data:
            logger.debug(f"{stock_name}({code}) 차트 데이터 조회 실패")
            return 0

        try:
            monthly_changes = [float(candle["prdy_ctrt"]) for candle in monthly_data]
            avg_monthly_change = np.mean(monthly_changes)
            if abs(avg_monthly_change) <= self.monthly_volatility_threshold:
                score += 1
                logger.debug(f"{stock_name}({code}) - 월간 변동성 조건 만족 (+1점), 변동성: {abs(avg_monthly_change):.2f}%")
        except Exception as e:
            logger.debug(f"{stock_name}({code}) 월간 차트 데이터 분석 오류: {e}")
            return 0

        # 거래량 증가 점수
        try:
            avg_30d_volume = np.mean([int(candle["acml_vol"]) for candle in daily_data])
            today_volume = int(daily_data[0]["acml_vol"])
            volume_ratio = today_volume / avg_30d_volume if avg_30d_volume > 0 else 0
            
            if today_volume >= avg_30d_volume * self.trade_volume_increase_ratio:
                score += 1
                logger.debug(f"{stock_name}({code}) - 거래량 증가 조건 만족 (+1점), 비율: {volume_ratio:.2f}배")
            else:
                logger.debug(f"{stock_name}({code}) - 거래량 증가 조건 미달, 비율: {volume_ratio:.2f}배")
        except Exception as e:
            logger.debug(f"{stock_name}({code}) 거래량 데이터 분석 오류: {e}")
            return 0

        # 종가 상승 점수
        try:
            today_close = int(daily_data[0]["stck_clpr"])
            prev_close = int(daily_data[1]["stck_clpr"])
            price_change_ratio = (today_close - prev_close) / prev_close if prev_close > 0 else 0
            
            if price_change_ratio >= self.close_price_increase_ratio:
                score += 1
                logger.debug(f"{stock_name}({code}) - 종가 상승 조건 만족 (+1점), 상승률: {price_change_ratio:.2f}%")
            else:
                logger.debug(f"{stock_name}({code}) - 종가 상승 조건 미달, 상승률: {price_change_ratio:.2f}%")
        except Exception as e:
            logger.debug(f"{stock_name}({code}) 종가 데이터 분석 오류: {e}")
            return 0
            
        logger.debug(f"{stock_name}({code}) - 최종 점수: {score}")
        return score
    
    def get_volume_increasing_stocks(self, target_count: int) -> List[str]:
        """
        거래량 급증 종목을 선정하는 함수
        
        Args:
            target_count: 선정할 종목 수
            
        Returns:
            list: 선정된 거래량 급증 종목 코드 리스트
        """
        send_message("거래량 급증 종목 선정 시작")
        
        # API를 통해 거래량 급증 종목 조회
        logger.info("거래량 급증 종목 조회 시도")
        increasing_stocks = self.api.get_volume_increasing_stocks(top_n=30)
        
        if not increasing_stocks:
            logger.error("거래량 급증 종목 조회 실패 - API 응답 없음")
            send_message("거래량 급증 종목 없음 (API 응답 실패)")
            return []
        
        logger.info(f"거래량 급증 종목 {len(increasing_stocks)}개 조회 성공")
            
        # 기본 필터링 (시가총액, 가격 등)
        filtered_stocks = []
        filtered_stock_names = []
        
        for stock in increasing_stocks:
            code = stock.get("code")
            name = stock.get("name", "N/A")
            
            if not code:
                continue
            
            # ETF 종목 제외
            if EXCLUDE_ETF and self.api.is_etf_stock(code):
                logger.info(f"ETF 종목 제외: {name} ({code})")
                continue
                
            # 주식 기본 정보 조회
            logger.info(f"{name}({code}) 정보 조회")
            info = self.api.get_stock_info(code)
            
            if not info:
                logger.warning(f"{code} 상세 정보 조회 실패")
                continue
                
            current_price = info.get("price", 0)
            market_cap = info.get("market_cap", 0)
            
            # 기본 조건 확인
            logger.info(f"{name}({code}) - 가격: {current_price:,}원, 시가총액: {market_cap:,}원")
            
            if (market_cap >= self.market_cap_threshold and 
                current_price >= self.price_threshold):
                filtered_stocks.append(code)
                filtered_stock_names.append(f"{name}({code})")
                logger.info(f"{name}({code}) - 조건 충족으로 선정")
            else:
                logger.info(f"{name}({code}) - 조건 미달로 제외")
                
            if len(filtered_stocks) >= target_count:
                break
        
        # 최종 선정 결과 알림
        if filtered_stocks:
            send_message(f"선정된 거래량 급증 종목: {', '.join(filtered_stock_names)}")
        else:
            send_message("조건에 맞는 거래량 급증 종목이 없습니다.")
                
        return filtered_stocks[:target_count] 