"""
한국 주식 자동매매 - 매매 로직 모듈
자동 매매 관련 클래스 및 함수 정의
"""

import time
import numpy as np

from korea_stock_auto.config import (
    TRADE_CONFIG, STOCK_FILTER
)
from korea_stock_auto.utils import send_message, wait
from korea_stock_auto.api_client import APIClient

class StockTrader:
    """국내 주식 자동 매매 클래스"""
    
    def __init__(self):
        """트레이더 초기화"""
        self.api = APIClient()
        self.initialize_trading()
    
    def initialize_trading(self):
        """매매 초기화"""
        # 매매 관련 변수 초기화
        self.symbol_list = []       # 관심 종목 리스트
        self.bought_list = []       # 매수 완료된 종목 리스트
        self.entry_prices = {}      # 종목별 매수가
        self.total_cash = self.api.get_balance()  # 보유 현금 조회
        
        # 현재 보유 주식 조회
        self.stock_dict = self.api.get_stock_balance()
        for sym in self.stock_dict.keys():
            self.bought_list.append(sym)
        
        # 매매 설정
        self.target_buy_count = TRADE_CONFIG["target_buy_count"]  # 매수할 종목 수
        self.buy_percentage = TRADE_CONFIG["buy_percentage"]      # 종목당 매수 금액 비율
        self.macd_short = TRADE_CONFIG["macd_short"]              # MACD 단기 기간
        self.macd_long = TRADE_CONFIG["macd_long"]                # MACD 장기 기간
        self.moving_avg_period = TRADE_CONFIG["moving_avg_period"]  # 이동평균선 기간
        
        # 매수 금액 설정
        self.buy_amount = self.total_cash * self.buy_percentage   # 종목별 매수 금액
        self.soldout = False
    
    def select_interest_stocks(self):
        """
        거래량 상위 종목 중에서 특정 조건을 만족하는 관심 종목을 선정하는 함수
        
        Returns:
            list: 선정된 관심 종목 코드 리스트
        """
        send_message("관심 종목 선정 시작")
        top_stocks = self.api.get_top_traded_stocks()
        
        if not top_stocks:
            send_message("관심 종목 없음 (API 응답 실패 또는 데이터 없음)")
            return []
        
        # top_stocks에서 종목 코드와 이름의 매핑을 생성
        top_stock_names = {
            stock.get("mksc_shrn_iscd"): stock.get("hts_kor_isnm", "N/A") 
            for stock in top_stocks
        }
        
        candidates = []
        score_threshold = STOCK_FILTER["score_threshold"]

        for stock in top_stocks:
            code = stock.get("mksc_shrn_iscd")
            if not code:
                continue

            # 주식 기본 정보 조회 
            info_data = self.api.get_stock_info(code)
            if not info_data:
                send_message(f"{code} 상세 정보 없음")
                continue

            if isinstance(info_data, list):
                info = info_data[0]
            elif isinstance(info_data, dict):
                info = info_data
            else:
                send_message(f"{code} 상세 정보 형식 오류")
                continue

            stock_name = info.get("hts_kor_isnm", "N/A")
            if stock_name == "N/A":
                stock_name = top_stock_names.get(code, "N/A")

            try:
                current_price = float(info["stck_prpr"])
                listed_shares = float(info["lstn_stcn"])
                market_cap = current_price * listed_shares
            except Exception as e:
                send_message(f"{stock_name} ({code}) 정보 계산 오류: {e}")
                continue

            score = 0
            if (market_cap >= STOCK_FILTER["market_cap_threshold"] and 
                current_price >= STOCK_FILTER["price_threshold"]):
                score += 1

            monthly_data = self.api.get_monthly_data(code)
            daily_data = self.api.get_daily_data(code)
            if not monthly_data or not daily_data:
                send_message(f"{stock_name} ({code}) 차트 데이터 없음")
                continue

            try:
                monthly_changes = [float(candle["prdy_ctrt"]) for candle in monthly_data]
                avg_monthly_change = np.mean(monthly_changes)
                if abs(avg_monthly_change) <= STOCK_FILTER["monthly_volatility_threshold"]:
                    score += 1
            except Exception as e:
                send_message(f"{stock_name} ({code}) 차트 데이터 오류: {e}")
                continue

            try:
                avg_30d_volume = np.mean([int(candle["acml_vol"]) for candle in daily_data])
                today_volume = int(daily_data[0]["acml_vol"])
                if today_volume >= avg_30d_volume * STOCK_FILTER["trade_volume_increase_ratio"]:
                    score += 1
            except Exception as e:
                send_message(f"{stock_name} ({code}) 거래량 데이터 오류: {e}")
                continue

            try:
                today_close = int(daily_data[0]["stck_clpr"])
                prev_close = int(daily_data[1]["stck_clpr"])
                if (today_close - prev_close) / prev_close >= STOCK_FILTER["close_price_increase_ratio"]:
                    score += 1
            except Exception as e:
                send_message(f"{stock_name} ({code}) 종가 데이터 오류: {e}")
                continue

            if score >= score_threshold:
                candidates.append((code, stock_name, score))
                send_message(f"후보 추가: {stock_name} ({code}) - 점수: {score}")
            else:
                send_message(f"후보 탈락: {stock_name} ({code}) - 점수: {score}")

        candidates.sort(key=lambda x: x[2], reverse=True)
        self.symbol_list = [code for (code, name, score) in candidates[:self.target_buy_count]]
        return self.symbol_list
    
    def calculate_macd(self, prices):
        """
        MACD(Moving Average Convergence Divergence) 계산 함수
        
        Args:
            prices (list): 가격 데이터 리스트
            
        Returns:
            float: MACD 값
        """
        if len(prices) < self.macd_long:
            return 0
            
        short_ema = np.mean(prices[-self.macd_short:])
        long_ema = np.mean(prices[-self.macd_long:])
        return short_ema - long_ema
    
    def get_moving_average(self, code, period=None):
        """
        지정된 기간의 이동평균선 계산 함수
        
        Args:
            code (str): 종목 코드
            period (int, optional): 이동평균 기간
            
        Returns:
            float or None: 이동평균 값 또는 데이터가 없는 경우 None
        """
        if period is None:
            period = self.moving_avg_period
            
        # 코드 포맷: 12자리로 변환
        formatted_code = code.zfill(12)
        daily_data = self.api.get_daily_data(formatted_code)
        
        if not daily_data or len(daily_data) < period:
            return None
            
        typical_prices = [
            (int(candle["stck_hgpr"]) + int(candle["stck_lwpr"]) + int(candle["stck_clpr"])) / 3
            for candle in daily_data[:period]
        ]
        alpha = 2 / (period + 1)
        ema = typical_prices[0]
        for price in typical_prices[1:]:
            ema = (price * alpha) + (ema * (1 - alpha))
        return ema
    
    def try_buy(self, code):
        """
        매수 시도
        
        Args:
            code (str): 종목 코드
            
        Returns:
            bool: 매수 성공 여부
        """
        # 매수 종목 수 제한 확인
        if len(self.bought_list) >= self.target_buy_count:
            return False
                
        # 이미 매수한 종목 스킵
        if code in self.bought_list:
            return False
        
        # 현재가 조회
        price_data = self.api.get_stock_info(code)
        if not price_data:
            return False
            
        # 현재가, 이동평균선 비교
        try:
            current_price = int(price_data["stck_prpr"])
            ma_price = self.get_moving_average(code)
            
            if ma_price and current_price > ma_price:
                # 매수 금액 기준으로 수량 계산
                buy_qty = int(self.buy_amount // current_price)
                
                if buy_qty > 0:
                    send_message(f"{code} 매수 조건 충족(현재가: {current_price}, 이동평균: {ma_price}) 매수를 시도합니다.")
                    result = self.api.buy_stock(code, buy_qty)
                    
                    if result:
                        self.bought_list.append(code)
                        self.entry_prices[code] = current_price
                        self.api.get_stock_balance()
                        return True
        except Exception as e:
            send_message(f"매수 시도 중 오류 발생: {e}")
            
        return False
    
    def try_sell(self, code, qty):
        """
        매도 시도
        
        Args:
            code (str): 종목 코드
            qty (int): 매도 수량
            
        Returns:
            bool: 매도 성공 여부
        """
        if not code in self.stock_dict:
            return False
            
        # 현재가 조회
        price_data = self.api.get_stock_info(code)
        if not price_data:
            return False
            
        # 현재가, 이동평균선 비교
        try:
            current_price = int(price_data["stck_prpr"])
            entry_price = self.entry_prices.get(code, 0)
            
            # 5% 이상 수익 또는 이동평균선 아래로 떨어진 경우 매도
            ma_price = self.get_moving_average(code)
            profit_ratio = (current_price - entry_price) / entry_price if entry_price else 0
            
            if (profit_ratio >= 0.05) or (ma_price and current_price < ma_price):
                send_message(f"{code} 매도 조건 충족(현재가: {current_price}, 매수가: {entry_price}, 수익률: {profit_ratio:.2%}) 매도를 시도합니다.")
                result = self.api.sell_stock(code, qty)
                
                if result:
                    self.bought_list.remove(code)
                    if code in self.entry_prices:
                        del self.entry_prices[code]
                    self.api.get_stock_balance()
                    return True
        except Exception as e:
            send_message(f"매도 시도 중 오류 발생: {e}")
            
        return False
    
    def sell_all_stocks(self):
        """보유 종목 일괄 매도"""
        self.stock_dict = self.api.get_stock_balance()
        
        for code, qty in self.stock_dict.items():
            if self.api.sell_stock(code, qty):
                send_message(f"{code} {qty}주 매도 완료")
                
        self.soldout = True
        self.bought_list = []
        self.stock_dict = {}
        wait(1)
        
        # 잔고 갱신
        self.stock_dict = self.api.get_stock_balance()
    
    def run_single_trading_cycle(self):
        """단일 매매 사이클 실행"""
        # 아직 관심종목이 선정되지 않았다면 선정
        if not self.symbol_list:
            self.select_interest_stocks()
            
        # 매수 여유가 있는 경우 매수 시도
        if len(self.bought_list) < self.target_buy_count:
            for code in self.symbol_list:
                if self.try_buy(code):
                    send_message(f"{code} 매수 완료")
                wait(1)
        
        # 보유 종목 매도 조건 확인
        for code, qty in list(self.stock_dict.items()):
            if self.try_sell(code, qty):
                send_message(f"{code} 매도 완료")
            wait(1)
        
        # 현금 및 잔고 갱신
        self.total_cash = self.api.get_balance()
        
    def run_trading(self, max_cycles=0):
        """
        자동 매매 실행 함수
        
        Args:
            max_cycles (int): 최대 사이클 수 (0이면 무한 반복)
        """
        send_message("자동 매매 시작")
        
        cycle_count = 0
        
        while True:
            try:
                # 단일 매매 사이클 실행
                self.run_single_trading_cycle()
                
                # 사이클 카운트 증가
                cycle_count += 1
                
                # 최대 사이클 도달 시 종료
                if max_cycles > 0 and cycle_count >= max_cycles:
                    send_message(f"최대 사이클({max_cycles}) 도달. 매매 종료.")
                    break
                    
                # 잠시 대기
                wait(30)
                
            except Exception as e:
                send_message(f"매매 중 오류 발생: {e}")
                wait(60)
                
        # 모든 보유 종목 매도
        self.sell_all_stocks()
        send_message("자동 매매 종료") 