"""
한국 주식 자동매매 - 매매 로직 모듈
자동 매매 관련 클래스 및 함수 정의
"""

import time
import numpy as np

from korea_stock_auto.config import (
    TRADE_CONFIG, STOCK_FILTER
)
from korea_stock_auto.utils.utils import send_message, wait
from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.models.model_manager import ModelVersionManager

class StockTrader:
    """국내 주식 자동 매매 클래스"""
    
    def __init__(self):
        """트레이더 초기화"""
        self.api = KoreaInvestmentApiClient()
        # 모델 관리자 초기화
        self.model_manager = ModelVersionManager()
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
        
        # 위험 관리 관련 변수 초기화
        self.daily_loss_limit = TRADE_CONFIG.get("daily_loss_limit", 0)  # 일일 손실 제한 (원)
        self.daily_loss_limit_pct = TRADE_CONFIG.get("daily_loss_limit_pct", 0)  # 일일 손실 제한 (%)
        self.daily_profit_limit_pct = TRADE_CONFIG.get("daily_profit_limit_pct", 0)  # 일일 수익 제한 (%)
        self.position_loss_limit = TRADE_CONFIG.get("position_loss_limit", 0)  # 포지션당 손실 제한
        self.max_position_size = TRADE_CONFIG.get("max_position_size", 0)  # 최대 포지션 크기
        self.trailing_stop_pct = TRADE_CONFIG.get("trailing_stop_pct", 0)  # 트레일링 스탑 비율
        
        # 포지션 트래킹
        self.position_max_prices = {}  # 종목별 최고가 추적
        self.daily_pl = 0  # 일일 손익
        self.initial_total_assets = self.calculate_total_assets()  # 초기 총 자산 가치
        self.daily_trade_count = 0  # 일일 거래 횟수 (통계용)
        self.risk_level = "NORMAL"  # 위험 수준 (NORMAL, CAUTION, HIGH)
    
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
        
        # 일일 손익 제한 확인
        if self.check_daily_pl_limits():
            return False
        
        # 현재가 조회
        price_data = self.api.get_stock_info(code)
        if not price_data:
            return False
            
        # 현재가, 이동평균선 비교
        try:
            current_price = int(price_data["stck_prpr"])
            ma_price = self.get_moving_average(code)
            
            # 모델 예측 활용 (추가된 부분)
            model_prediction = self.get_model_prediction(code, current_price)
            model_action = model_prediction.get('action', 'HOLD')
            model_confidence = model_prediction.get('confidence', 0.0)
            
            # 기존 매수 조건 + 모델 예측 결합
            buy_condition = (ma_price and current_price > ma_price)
            model_buy_signal = (model_action == 'BUY' and model_confidence >= 0.6)
            
            # 전략 조합: 기존 전략이나 모델 예측이 강한 매수 신호를 보내면 매수
            if buy_condition or model_buy_signal:
                # 매수 금액 기준으로 수량 계산
                buy_qty = int(self.buy_amount // current_price)
                
                # 포지션 크기 제한 확인
                if self.max_position_size > 0 and buy_qty > self.max_position_size:
                    buy_qty = self.max_position_size
                    send_message(f"{code} 매수량 제한: 최대 {self.max_position_size}주로 제한됨")
                
                if buy_qty > 0:
                    # 위험 수준 확인
                    if self.risk_level == "HIGH":
                        send_message(f"{code} 매수 거부: 현재 위험 수준이 높음 (위험 수준: {self.risk_level})")
                        return False
                    
                    # 노출도 제한 확인
                    if not self.check_exposure_limit(code, buy_qty, current_price):
                        send_message(f"{code} 매수 거부: 최대 노출도 제한 초과")
                        return False
                    
                    # 매수 결정 요인 로깅
                    if buy_condition and model_buy_signal:
                        reason = "기술적 조건 및 모델 예측"
                    elif buy_condition:
                        reason = "기술적 조건 (이동평균선 위)"
                    else:
                        reason = f"모델 예측 (신뢰도: {model_confidence:.2f})"
                    
                    send_message(f"{code} 매수 조건 충족({reason}) - 현재가: {current_price}, 이동평균: {ma_price} 매수를 시도합니다.")
                    result = self.api.buy_stock(code, buy_qty)
                    
                    if result:
                        self.bought_list.append(code)
                        self.entry_prices[code] = current_price
                        self.position_max_prices[code] = current_price  # 최고가 초기화
                        self.daily_trade_count += 1
                        self.api.get_stock_balance()
                        
                        # 모델 성능 업데이트를 위한 거래 결과 기록
                        self.update_models_after_trade(code, 'BUY', current_price, buy_qty)
                        
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
            
            # 최고가 업데이트
            if code in self.position_max_prices:
                if current_price > self.position_max_prices[code]:
                    self.position_max_prices[code] = current_price
            else:
                self.position_max_prices[code] = current_price
            
            # 5% 이상 수익 또는 이동평균선 아래로 떨어진 경우 매도
            ma_price = self.get_moving_average(code)
            profit_ratio = (current_price - entry_price) / entry_price if entry_price else 0
            
            # 트레일링 스탑 확인
            trailing_stop_triggered = self.check_trailing_stop(code, current_price)
            
            # 포지션 손실 제한 확인
            position_loss_limit_triggered = self.check_position_loss_limit(code, current_price, entry_price)
            
            # 일일 손익 제한 확인 (이 부분은 '수익 목표 달성' 매도 전략과는 별개)
            daily_pl_limit_triggered = False
            if self.check_daily_pl_limits():
                daily_pl_limit_triggered = True
                
            # 모델 예측 활용 (추가된 부분)
            model_prediction = self.get_model_prediction(code, current_price)
            model_action = model_prediction.get('action', 'HOLD')
            model_confidence = model_prediction.get('confidence', 0.0)
            model_sell_signal = (model_action == 'SELL' and model_confidence >= 0.6)
            
            # 기존 매도 조건
            technical_sell_condition = (profit_ratio >= 0.05) or (ma_price and current_price < ma_price)
            risk_sell_condition = trailing_stop_triggered or position_loss_limit_triggered or daily_pl_limit_triggered
            
            # 기존 조건 또는 모델 매도 신호가 있는 경우 매도
            if technical_sell_condition or risk_sell_condition or model_sell_signal:
                sell_reason = ""
                if profit_ratio >= 0.05:
                    sell_reason = "수익 목표 달성"
                elif ma_price and current_price < ma_price:
                    sell_reason = "이동평균선 아래로 하락"
                elif trailing_stop_triggered:
                    sell_reason = "트레일링 스탑 발동"
                elif position_loss_limit_triggered:
                    sell_reason = "포지션 손실 한도 도달"
                elif daily_pl_limit_triggered:
                    sell_reason = "일일 손익 한도 도달"
                elif model_sell_signal:
                    sell_reason = f"모델 매도 신호 (신뢰도: {model_confidence:.2f})"
                
                send_message(f"{code} 매도 조건 충족({sell_reason}) - 현재가: {current_price}, 매수가: {entry_price}, 수익률: {profit_ratio:.2%} 매도를 시도합니다.")
                result = self.api.sell_stock(code, qty)
                
                if result:
                    # 매도 결과 기록
                    profit = (current_price - entry_price) * qty
                    self.daily_pl += profit
                    self.daily_trade_count += 1
                    
                    # 모델 성능 업데이트를 위한 거래 결과 기록
                    self.update_models_after_trade(code, 'SELL', current_price, qty, profit)
                    
                    self.bought_list.remove(code)
                    if code in self.entry_prices:
                        del self.entry_prices[code]
                    if code in self.position_max_prices:
                        del self.position_max_prices[code]
                    self.api.get_stock_balance()
                    
                    # 손익 한도 확인 후 위험 수준 업데이트
                    self.check_daily_pl_limits()
                    self.update_risk_level()
                    
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
            
        # 일일 손익 제한 확인
        if self.check_daily_pl_limits():
            send_message("일일 손익 한도에 도달하여 거래를 중단합니다.")
            self.sell_all_stocks()
            return
            
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
        
        # 위험 수준 업데이트
        self.update_risk_level()
        
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
    
    def check_trailing_stop(self, code, current_price):
        """
        트레일링 스탑 확인
        
        Args:
            code (str): 종목 코드
            current_price (float): 현재 가격
            
        Returns:
            bool: 트레일링 스탑 발동 여부
        """
        if self.trailing_stop_pct <= 0 or code not in self.position_max_prices:
            return False
            
        max_price = self.position_max_prices[code]
        trailing_stop_price = max_price * (1 - self.trailing_stop_pct / 100)
        
        if current_price <= trailing_stop_price:
            send_message(f"{code} 트레일링 스탑 발동: 최고가 {max_price}, 현재가 {current_price}, 하락률 {((max_price - current_price) / max_price * 100):.2f}%")
            return True
            
        return False
    
    def check_position_loss_limit(self, code, current_price, entry_price):
        """
        포지션 손실 제한 확인
        
        Args:
            code (str): 종목 코드
            current_price (float): 현재 가격
            entry_price (float): 매수 가격
            
        Returns:
            bool: 손실 제한 도달 여부
        """
        if self.position_loss_limit <= 0 or entry_price <= 0:
            return False
            
        qty = self.stock_dict.get(code, 0)
        position_pl = (current_price - entry_price) * qty
        
        if position_pl < -self.position_loss_limit:
            send_message(f"{code} 포지션 손실 한도 도달: 현재 손실 {-position_pl}, 한도 {self.position_loss_limit}")
            return True
            
        return False
    
    def calculate_total_assets(self):
        """
        총 자산 가치 계산
        
        Returns:
            float: 총 자산 가치 (현금 + 보유주식 가치)
        """
        # 현금 조회
        cash = self.api.get_balance()
        
        # 보유 주식 가치 계산
        stock_value = 0
        for code, qty in self.stock_dict.items():
            price_data = self.api.get_stock_info(code)
            if price_data:
                current_price = int(price_data["stck_prpr"])
                stock_value += current_price * qty
        
        return cash + stock_value
    
    def check_daily_pl_limits(self):
        """
        일일 손익 제한 확인
        
        Returns:
            bool: 손익 제한 도달 여부
        """
        # 손익 제한이 모두 0이면 제한 없음
        if self.daily_loss_limit <= 0 and self.daily_loss_limit_pct <= 0 and self.daily_profit_limit_pct <= 0:
            return False
            
        # 현재 총 자산 계산
        current_assets = self.calculate_total_assets()
        
        # 자산 대비 손익 계산
        abs_pl = current_assets - self.initial_total_assets  # 절대적 손익 (원)
        pl_ratio = (abs_pl / self.initial_total_assets) * 100  # 비율 손익 (%)
        
        # 손실 제한 확인 - 비율 또는 절대액 중 하나라도 도달하면 제한
        if (self.daily_loss_limit > 0 and abs_pl <= -self.daily_loss_limit) or \
           (self.daily_loss_limit_pct > 0 and pl_ratio <= -self.daily_loss_limit_pct):
            
            loss_msg = []
            if self.daily_loss_limit > 0 and abs_pl <= -self.daily_loss_limit:
                loss_msg.append(f"절대 손실: {-abs_pl:,.0f}원 (한도: {self.daily_loss_limit:,.0f}원)")
            
            if self.daily_loss_limit_pct > 0 and pl_ratio <= -self.daily_loss_limit_pct:
                loss_msg.append(f"손실률: {-pl_ratio:.2f}% (한도: {self.daily_loss_limit_pct}%)")
            
            send_message(f"일일 손실 한도 도달: {', '.join(loss_msg)}")
            return True
            
        # 수익 제한 확인
        if self.daily_profit_limit_pct > 0 and pl_ratio >= self.daily_profit_limit_pct:
            send_message(f"일일 수익 한도 도달: 현재 수익률 {pl_ratio:.2f}% (한도: {self.daily_profit_limit_pct}%)")
            return True
            
        return False
    
    def update_risk_level(self):
        """현재 위험 수준 업데이트"""
        # 현재 총 자산 계산
        current_assets = self.calculate_total_assets()
        
        # 자산 대비 손익 계산
        abs_pl = current_assets - self.initial_total_assets  # 절대적 손익 (원)
        pl_ratio = (abs_pl / self.initial_total_assets) * 100  # 비율 손익 (%)
        
        # 수익인 경우 위험 수준은 NORMAL
        if pl_ratio >= 0:
            new_risk_level = "NORMAL"
        else:
            # 비율 손실과 절대 손실 중 더 큰 위험도를 계산
            ratio_loss_level = 0
            abs_loss_level = 0
            
            # 비율 기준 위험도 계산
            if self.daily_loss_limit_pct > 0:
                ratio_loss_level = -pl_ratio / self.daily_loss_limit_pct
                
            # 절대액 기준 위험도 계산
            if self.daily_loss_limit > 0:
                abs_loss_level = -abs_pl / self.daily_loss_limit
            
            # 둘 중 더 높은 위험도 사용
            loss_level = max(ratio_loss_level, abs_loss_level)
            
            # 위험 수준 설정
            if loss_level > 0.8:
                new_risk_level = "HIGH"
            elif loss_level > 0.5:
                new_risk_level = "CAUTION"
            else:
                new_risk_level = "NORMAL"
        
        # 위험 수준이 변경되면 알림
        if new_risk_level != self.risk_level:
            self.risk_level = new_risk_level
            
            # 위험 수준 변경 메시지
            risk_info = []
            if pl_ratio < 0:
                risk_info.append(f"손실률: {-pl_ratio:.2f}%")
                if self.daily_loss_limit_pct > 0:
                    risk_info.append(f"비율 한도 대비: {-pl_ratio/self.daily_loss_limit_pct:.2f}")
                if self.daily_loss_limit > 0:
                    risk_info.append(f"금액 한도 대비: {-abs_pl/self.daily_loss_limit:.2f}")
            
            send_message(f"위험 수준 변경: {self.risk_level} ({', '.join(risk_info)})")
    
    def calculate_volatility(self, code, period=20):
        """
        종목 변동성 계산
        
        Args:
            code (str): 종목 코드
            period (int): 계산 기간
            
        Returns:
            float: 일간 변동성
        """
        try:
            # 코드 포맷: 12자리로 변환
            formatted_code = code.zfill(12)
            daily_data = self.api.get_daily_data(formatted_code)
            
            if not daily_data or len(daily_data) < period:
                return None
                
            # 일간 수익률 계산
            daily_returns = []
            for i in range(1, min(period, len(daily_data))):
                prev_close = float(daily_data[i]["stck_clpr"])
                curr_close = float(daily_data[i-1]["stck_clpr"])
                if prev_close > 0:
                    daily_return = (curr_close - prev_close) / prev_close
                    daily_returns.append(daily_return)
            
            # 표준편차 계산
            if daily_returns:
                volatility = np.std(daily_returns)
                return volatility
                
        except Exception as e:
            send_message(f"변동성 계산 중 오류 발생: {e}")
            
        return None
    
    def get_position_diversity(self):
        """
        포지션 다양성 계산
        
        Returns:
            dict: 섹터별 익스포저 정보
        """
        sector_exposure = {}
        
        for code in self.bought_list:
            # 여기서는 간단하게 종목별로 구분하지만, 실제로는 섹터 정보를 가져와서 분류해야 함
            # API에서 섹터 정보 제공 여부에 따라 구현 필요
            sector = "Unknown"  # 실제로는 API에서 섹터 정보 가져와야 함
            
            if sector not in sector_exposure:
                sector_exposure[sector] = 0
                
            qty = self.stock_dict.get(code, 0)
            price_data = self.api.get_stock_info(code)
            if price_data:
                current_price = int(price_data["stck_prpr"])
                sector_exposure[sector] += current_price * qty
        
        return sector_exposure
    
    def check_exposure_limit(self, code, buy_qty, current_price):
        """
        노출도 제한 확인
        
        Args:
            code (str): 종목 코드
            buy_qty (int): 매수 수량
            current_price (float): 현재 가격
            
        Returns:
            bool: 노출도 제한 내 여부 (True: 허용, False: 제한)
        """
        exposure_limit_pct = TRADE_CONFIG.get("exposure_limit_pct", 100)
        if exposure_limit_pct >= 100:  # 제한 없음
            return True
            
        # 현재 총 노출도 계산
        current_exposure = 0
        for c, qty in self.stock_dict.items():
            price_data = self.api.get_stock_info(c)
            if price_data:
                price = int(price_data["stck_prpr"])
                current_exposure += price * qty
                
        # 새로운 매수 후 노출도 계산
        new_exposure = current_exposure + (current_price * buy_qty)
        
        # 최대 허용 노출도 계산 (현금 + 주식 가치 기준)
        total_assets = self.total_cash + current_exposure
        max_allowed_exposure = total_assets * (exposure_limit_pct / 100)
        
        return new_exposure <= max_allowed_exposure
    
    def reset_daily_stats(self):
        """일일 통계 초기화"""
        self.daily_pl = 0
        self.daily_trade_count = 0
        self.initial_total_assets = self.calculate_total_assets()  # 초기 자산 가치 재설정
        send_message(f"일일 거래 통계가 초기화되었습니다. 초기 자산 가치: {self.initial_total_assets:,}원")
    
    def get_model_prediction(self, code, current_price, market_data=None):
        """
        강화학습 모델을 활용한 예측 결과 조회
        
        Args:
            code (str): 종목 코드
            current_price (float): 현재 가격
            market_data (dict, optional): 추가 시장 데이터
            
        Returns:
            dict: 예측 결과 (행동, 신뢰도)
        """
        try:
            # 현재 시장 상태 구성
            state = self.prepare_model_state(code, current_price, market_data)
            
            # 앙상블 모델 예측 수행
            prediction = self.model_manager.create_ensemble_prediction(state)
            
            # 로깅
            action = prediction.get('action', 'HOLD')
            confidence = prediction.get('confidence', 0.0)
            champion_model = self.model_manager.current_champion
            
            send_message(f"{code} 모델 예측: {action} (신뢰도: {confidence:.2f}, 챔피언 모델: {champion_model})")
            
            return prediction
            
        except Exception as e:
            send_message(f"모델 예측 실패: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def prepare_model_state(self, code, current_price, market_data=None):
        """
        강화학습 모델에 입력할 상태 구성
        
        Args:
            code (str): 종목 코드
            current_price (float): 현재 가격
            market_data (dict, optional): 추가 시장 데이터
            
        Returns:
            dict: 상태 벡터
        """
        # 이 부분은 실제 구현 시 필요한 상태 정보를 수집하여 구성해야 함
        # 여기서는 예시로 최소한의 정보만 포함
        
        state = {
            'code': code,
            'current_price': current_price,
            'time': time.time()
        }
        
        # 기술적 지표 추가
        ma_price = self.get_moving_average(code)
        state['ma_price'] = ma_price if ma_price else current_price
        
        # 보유 여부 및 매수가 추가
        state['is_holding'] = code in self.bought_list
        state['entry_price'] = self.entry_prices.get(code, 0)
        
        # 종목 변동성 추가
        volatility = self.calculate_volatility(code)
        state['volatility'] = volatility if volatility is not None else 0
        
        # 시장 데이터 추가
        if market_data:
            state.update(market_data)
        
        return state
    
    def update_models_after_trade(self, code, action, price, quantity, profit=None):
        """
        거래 후 모델 성능 업데이트
        
        Args:
            code (str): 종목 코드
            action (str): 거래 유형 ('BUY', 'SELL')
            price (float): 거래 가격
            quantity (int): 거래 수량
            profit (float, optional): 수익 (매도 시)
        """
        # 이 메서드는 실제 거래 결과를 바탕으로 모델 성능을 평가하고 업데이트
        # 모델 예측에 따른 실제 거래 결과를 저장하여 나중에 성능 평가에 활용
        
        try:
            # 여기에서는 간단한 로깅만 수행
            if action == 'SELL' and profit is not None:
                # 수익성 거래인 경우, 모델 성능 업데이트를 위한 정보 저장
                profit_ratio = profit / (price * quantity) if price and quantity else 0
                
                # DB에 거래 결과 기록 (실제 거래 결과 수집)
                # 나중에 이 데이터를 활용하여 모델 성능 평가 수행
                send_message(f"모델 성능 업데이트 데이터 수집 - {code} {action} 수익률: {profit_ratio:.2%}")
            
        except Exception as e:
            send_message(f"모델 업데이트 실패: {e}") 