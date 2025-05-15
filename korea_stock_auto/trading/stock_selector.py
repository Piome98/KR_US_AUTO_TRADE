"""
한국 주식 자동매매 - 종목 선정 모듈
관심 종목 선정 관련 클래스 및 함수 정의
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional

from korea_stock_auto.config import STOCK_FILTER
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api import KoreaInvestmentApiClient

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
        
        # top_stocks에서 종목 코드와 이름의 매핑을 생성
        top_stock_names = {
            stock.get("code"): stock.get("name", "N/A") 
            for stock in top_stocks
        }
        
        candidates = []
        
        for stock in top_stocks:
            code = stock.get("code")
            if not code:
                continue

            # 주식 기본 정보 조회 
            info_data = self.api.get_stock_info(code)
            if not info_data:
                logger.debug(f"{code} 상세 정보 조회 실패")
                continue

            if isinstance(info_data, list):
                info = info_data[0]
            elif isinstance(info_data, dict):
                info = info_data
            else:
                logger.debug(f"{code} 상세 정보 형식 오류")
                continue

            stock_name = info.get("name", "N/A")
            if stock_name == "N/A":
                stock_name = top_stock_names.get(code, "N/A")

            try:
                current_price = float(info.get("price", 0))
                listed_shares = float(info.get("listed_shares", 0))
                market_cap = current_price * listed_shares
            except Exception as e:
                logger.debug(f"{stock_name} ({code}) 정보 계산 오류: {e}")
                continue

            # 종목 점수 계산
            score = self._calculate_stock_score(code, stock_name, current_price, market_cap)
            
            if score >= self.score_threshold:
                candidates.append((code, stock_name, score))
                logger.debug(f"후보 추가: {stock_name} ({code}) - 점수: {score}")
            else:
                logger.debug(f"후보 탈락: {stock_name} ({code}) - 점수: {score}")

        candidates.sort(key=lambda x: x[2], reverse=True)
        selected_stocks = [code for (code, name, score) in candidates[:target_count]]
        
        if selected_stocks:
            stock_names = [f"{name}({code})" for code, name, _ in candidates[:target_count]]
            send_message(f"선정된 관심 종목: {', '.join(stock_names)}")
        else:
            send_message("조건에 맞는 관심 종목이 없습니다.")
            
        return selected_stocks
    
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