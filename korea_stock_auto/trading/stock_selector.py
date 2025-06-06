"""
한국 주식 자동매매 - 종목 선정 모듈
관심 종목 선정 관련 클래스 및 함수 정의
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional

from korea_stock_auto.config import get_config, AppConfig
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.trading.technical_analyzer import TechnicalAnalyzer

logger = logging.getLogger("stock_auto")

class StockSelector:
    """국내 주식 관심 종목 선정 클래스"""
    
    def __init__(self, api_client: KoreaInvestmentApiClient, config: AppConfig):
        """
        종목 선정기 초기화
        
        Args:
            api_client: API 클라이언트 인스턴스
            config: 애플리케이션 설정
        """
        self.api = api_client
        self.config = config
        self.score_threshold = config.stock_filter.score_threshold
        self.market_cap_threshold = config.stock_filter.market_cap_threshold
        self.price_threshold = config.stock_filter.price_threshold
        self.monthly_volatility_threshold = config.stock_filter.monthly_volatility_threshold
        self.trade_volume_increase_ratio = config.stock_filter.trade_volume_increase_ratio
        self.close_price_increase_ratio = config.stock_filter.close_price_increase_ratio
        # 기술적 분석기 초기화
        self.tech_analyzer = TechnicalAnalyzer(api_client)
    
    def select_interest_stocks(self, target_count: int) -> List[str]:
        """
        개선된 관심 종목 선정 함수
        1. 거래량 상위 종목으로 일정 거래량 이상 종목 필터링
        2. 일정 거래량 이상 종목 중 ETF 종목 제거
        3. 2중 필터링 된 종목들 일별 시세 데이터 및 재무 데이터 호출 및 크롤링 (한번에 수집 및 캐싱)
        4. 기술적 지표 조건에 따라 조건에 부합하는 요건 갯수당 점수 평가
        5. 점수 가장 높은 4개 종목 관심종목에 추가 및 해당 종목 매매 진행
        
        Args:
            target_count: 선정할 종목 수
            
        Returns:
            list: 선정된 관심 종목 코드 리스트
        """
        send_message("관심 종목 선정 시작", self.config.notification.discord_webhook_url)
        
        # 1단계: 거래량 상위 종목으로 일정 거래량 이상 종목 필터링
        logger.info("1단계: 거래량 상위 종목 조회 및 거래량 필터링")
        top_stocks = self.api.get_top_traded_stocks(market_type="1", top_n=50)  # 코스피 시장 50개
        
        if not top_stocks:
            logger.error("거래량 상위 종목 조회 실패 - API 응답 없음")
            send_message("관심 종목 없음 (API 응답 실패)", self.config.notification.discord_webhook_url)
            return []
        
        logger.info(f"거래량 상위 종목 {len(top_stocks)}개 조회 성공")
        send_message(f"거래량 상위 종목 {len(top_stocks)}개 조회 성공", self.config.notification.discord_webhook_url)
        
        # 거래량 필터링 (일정 거래량 이상)
        volume_filtered_stocks = []
        min_volume = 500000  # 최소 50만주 거래량
        
        for stock in top_stocks:
            volume = stock.get("volume", 0)
            if volume >= min_volume:
                volume_filtered_stocks.append(stock)
            else:
                logger.debug(f"거래량 부족으로 제외: {stock['name']} ({stock['code']}) - 거래량: {volume:,}주")
        
        logger.info(f"거래량 필터링 후 {len(volume_filtered_stocks)}개 종목")
        
        # 2단계: ETF 종목 제거
        logger.info("2단계: ETF 종목 제거")
        non_etf_stocks = []
        
        for stock in volume_filtered_stocks:
            code = stock["code"]
            name = stock["name"]
            
            # ETF 종목 제외 (설정에 따라)
            if self.config.stock_filter.exclude_etf and self.api.is_etf_stock(code):
                logger.info(f"ETF 종목 제외: {name} ({code})")
                continue
                
            non_etf_stocks.append(stock)
        
        if not non_etf_stocks:
            logger.warning("ETF를 제외한 거래량 상위 종목이 없습니다.")
            send_message("ETF를 제외한 관심 종목 없음", self.config.notification.discord_webhook_url)
            return []
            
        logger.info(f"ETF 제외 후 {len(non_etf_stocks)}개 종목 선정됨")
        send_message(f"ETF 제외 후 {len(non_etf_stocks)}개 종목 선정됨", self.config.notification.discord_webhook_url)
        
        # 3단계: 일별 시세 데이터 및 재무 데이터 호출 및 크롤링 (충분한 데이터로 한번에 수집)
        logger.info("3단계: 종목별 데이터 수집 및 기술적 지표 계산")
        data_available_stocks = []
        
        for stock in non_etf_stocks:
            code = stock["code"]
            name = stock["name"]
            
            logger.debug(f"데이터 수집 시작: {name} ({code})")
            
            # 일별 시세 데이터 미리 수집 (충분한 데이터 확보 - 300개로 증가)
            try:
                # 기술적 분석기에 데이터 업데이트 (일봉 데이터, 충분한 데이터 요청)
                logger.info(f"기술적 지표 계산용 데이터 수집: {name} ({code})")
                self.tech_analyzer.update_symbol_data(code, interval='D', limit=300)  # 300개로 증가
                
                # 캐시된 데이터 확인
                ohlcv_data = self.tech_analyzer._get_cached_ohlcv(code, 'D')
                
                if ohlcv_data is None or len(ohlcv_data) < 30:
                    logger.warning(f"충분한 일봉 데이터 없음: {name} ({code}) - 데이터 수: {len(ohlcv_data) if ohlcv_data is not None else 0}")
                    continue
                
                # 기본 정보 추가
                stock["data_length"] = len(ohlcv_data)
                data_available_stocks.append(stock)
                logger.info(f"데이터 수집 완료: {name} ({code}) - 일봉 데이터: {len(ohlcv_data)}개")
                
                # API 호출 간 버퍼 (과도한 요청 방지)
                import time
                time.sleep(0.5)  # 500ms 지연
                
            except Exception as e:
                logger.warning(f"데이터 수집 실패: {name} ({code}) - 오류: {e}")
                continue
        
        logger.info(f"충분한 데이터를 가진 종목: {len(data_available_stocks)}개")
        
        if not data_available_stocks:
            logger.warning("충분한 데이터를 가진 종목이 없습니다.")
            send_message("충분한 데이터를 가진 관심 종목 없음", self.config.notification.discord_webhook_url)
            return []
        
        # 4단계: 기술적 지표 조건에 따라 점수 평가
        logger.info("4단계: 기술적 지표 기반 점수 평가")
        scored_stocks = []
        
        for stock in data_available_stocks:
            code = stock["code"]
            name = stock["name"]
            current_price = stock.get("price", 0)
            market_cap = stock.get("market_cap", 0)
            volume = stock.get("volume", 0)
            
            logger.debug(f"점수 계산 시작: {name} ({code})")
            
            # 기본 점수
            score = 0
            score_details = []
            
            try:
                # 1. 시가총액/가격 조건 (기본 조건)
                if market_cap > 1000000000000 and current_price > 5000:  # 1조원 이상, 5천원 이상
                    score += 1
                    score_details.append("시가총액/가격")
                
                # 2. 거래량 조건
                if volume > 1000000:  # 백만주 이상 거래
                    score += 1
                    score_details.append("거래량")
                
                # 3. 이동평균선 조건 (20일 이평선)
                ma20 = self.tech_analyzer.get_moving_average(code, interval='D', window=20)
                if ma20 is not None and isinstance(ma20, (int, float)) and current_price > ma20 * 1.01:  # 20일 이평선 1% 이상 상회
                    score += 1
                    score_details.append("20일이평선상회")
                
                # 4. 변동성 조건
                volatility = self.tech_analyzer.calculate_volatility(code, interval='D', window=20)
                if volatility is not None and isinstance(volatility, (int, float)) and 0.15 <= volatility <= 0.4:  # 적정 변동성 범위 (15%~40%)
                    score += 1
                    score_details.append("적정변동성")
                
                # 5. 골든 크로스 조건 (5일선이 20일선 상향돌파)
                is_golden = self.tech_analyzer.check_ma_golden_cross(code, interval='D', short_period=5, long_period=20)
                if is_golden is True:  # 명시적으로 True 체크
                    score += 2
                    score_details.append("골든크로스")
                
                # 6. RSI 조건 (과매도 구간에서 반등)
                rsi_data = self.tech_analyzer.get_rsi_values(code, interval='D', window=14)
                if rsi_data and isinstance(rsi_data, dict):
                    current_rsi = rsi_data.get('current', rsi_data.get('rsi', 50))
                    if isinstance(current_rsi, (int, float)) and 30 <= current_rsi <= 70:  # RSI 30~70 구간
                        score += 1
                        score_details.append("RSI적정")
                
                # 7. 볼린저 밴드 조건
                bb = self.tech_analyzer.calculate_bollinger_bands(code, interval='D', window=20)
                if bb and isinstance(bb, dict):
                    bb_middle = bb.get("middle")
                    bb_upper = bb.get("upper")
                    if (bb_middle is not None and bb_upper is not None and 
                        isinstance(bb_middle, (int, float)) and isinstance(bb_upper, (int, float)) and
                        bb_middle < current_price < bb_upper):  # 중간선과 상단선 사이
                        score += 1
                        score_details.append("볼린저밴드")
                
                # 8. MACD 조건
                macd_data = self.tech_analyzer.get_macd(code, interval='D')
                if macd_data and isinstance(macd_data, dict):
                    macd_val = macd_data.get('macd', 0)
                    signal_val = macd_data.get('signal', 0)
                    if (isinstance(macd_val, (int, float)) and isinstance(signal_val, (int, float)) and 
                        macd_val > signal_val):  # MACD > Signal
                        score += 1
                        score_details.append("MACD상승")
                
                logger.debug(f"{name} ({code}) - 점수: {score}, 조건: {', '.join(score_details)}")
                
            except Exception as e:
                logger.warning(f"점수 계산 중 오류: {name} ({code}) - {e}")
                score = 0
                score_details = ["오류발생"]
            
            # 점수와 함께 저장
            scored_stocks.append({
                "code": code,
                "name": name,
                "score": score,
                "score_details": score_details,
                "current_price": current_price,
                "market_cap": market_cap,
                "volume": volume,
                "data_length": stock.get("data_length", 0)
            })
        
        # 5단계: 점수 가장 높은 종목들 선정
        logger.info("5단계: 최종 관심종목 선정")
        
        # 점수 기준으로 정렬
        scored_stocks.sort(key=lambda x: (x["score"], x["volume"]), reverse=True)
        
        # 상위 종목 선정
        interest_stocks = []
        selected_details = []
        
        for stock in scored_stocks[:target_count * 2]:  # 여유있게 선택
            if stock["score"] >= 3:  # 최소 3점 이상인 종목만 선정
                interest_stocks.append(stock["code"])
                selected_details.append(f"{stock['name']}({stock['code']}) - {stock['score']}점({', '.join(stock['score_details'])})")
                logger.info(f"관심종목 선정: {stock['name']} ({stock['code']}) - 점수: {stock['score']}, 조건: {', '.join(stock['score_details'])}")
                
                if len(interest_stocks) >= target_count:
                    break
        
        # 선정된 종목이 부족하면 점수 상위 종목으로 보완
        if len(interest_stocks) < target_count and scored_stocks:
            remaining_count = target_count - len(interest_stocks)
            for stock in scored_stocks:
                if stock["code"] not in interest_stocks and stock["score"] >= 1:
                    interest_stocks.append(stock["code"])
                    selected_details.append(f"{stock['name']}({stock['code']}) - {stock['score']}점({', '.join(stock['score_details'])})")
                    logger.info(f"보완 선정: {stock['name']} ({stock['code']}) - 점수: {stock['score']}, 조건: {', '.join(stock['score_details'])}")
                    
                    remaining_count -= 1
                    if remaining_count <= 0:
                        break
        
        # 최종 결과 알림
        if interest_stocks:
            send_message(f"관심 종목 {len(interest_stocks)}개 선정 완료", self.config.notification.discord_webhook_url)
            for detail in selected_details:
                logger.info(f"선정 상세: {detail}")
        else:
            send_message("조건에 맞는 관심 종목이 없습니다.", self.config.notification.discord_webhook_url)
        
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
        send_message("거래량 급증 종목 선정 시작", self.config.notification.discord_webhook_url)
        
        # API를 통해 거래량 급증 종목 조회
        logger.info("거래량 급증 종목 조회 시도")
        increasing_stocks = self.api.get_volume_increasing_stocks(top_n=30)
        
        if not increasing_stocks:
            logger.error("거래량 급증 종목 조회 실패 - API 응답 없음")
            send_message("거래량 급증 종목 없음 (API 응답 실패)", self.config.notification.discord_webhook_url)
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
            if self.config.stock_filter.exclude_etf and self.api.is_etf_stock(code):
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
            send_message(f"선정된 거래량 급증 종목: {', '.join(filtered_stock_names)}", self.config.notification.discord_webhook_url)
        else:
            send_message("조건에 맞는 거래량 급증 종목이 없습니다.", self.config.notification.discord_webhook_url)
                
        return filtered_stocks[:target_count] 