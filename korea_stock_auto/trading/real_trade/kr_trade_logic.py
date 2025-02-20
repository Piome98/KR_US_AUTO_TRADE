# 자동 매매 로직
# korea_stock_auto.trading.real_trade.kr_trade_logic

import time
import datetime
import numpy as np
from korea_stock_auto.trading.real_trade.kr_stock_api import *  # buy_stock, sell_stock, get_current_price 등
from korea_stock_auto.utility.kr_config import *
from korea_stock_auto.utility.kr_utils import *

# (모듈 상단에서 real_time_ws의 get_current_price 임포트 제거하여 순환 참조 방지)

# 📌 관심 종목 필터링 기준
TRADE_VOLUME_THRESHOLD = 1_000_000      # 100만 주 이상 
PRICE_THRESHOLD = 3_000                 # 현재가 3,000원 이상
MARKET_CAP_THRESHOLD = 400_000_000_000  # 4,000억 원 이상
MONTHLY_VOLATILITY_THRESHOLD = 10       # 10개월 등락률 평균 ±10% 이내
TRADE_VOLUME_INCREASE_RATIO = 4.0       # 거래량 400% 증가 조건
CLOSE_PRICE_INCREASE_RATIO = 0.05       # 종가 5% 이상 상승 조건

# ✅ 이동평균선 및 MACD 설정
MACD_SHORT = 5
MACD_LONG = 60
MOVING_AVG_PERIOD = 20  # 20일 이동평균선

# 📌 포트폴리오 비율 조정
buy_percentage = 0.25   # 종목당 매수 비율
target_buy_count = 4    # 매수할 종목 수

# 전역 딕셔너리: 각 종목의 매수가(구매 시점의 가격)를 저장
entry_prices = {}
symbol_list = []

from korea_stock_auto.shared.global_vars import symbol_list  # 전역으로 선언되어 있다고 가정

def select_interest_stocks():
    send_message("📊 관심 종목 선정 시작")
    top_stocks = get_top_traded_stocks()
    
    if not top_stocks:
        send_message("⚠️ 관심 종목 없음 (API 응답 실패 또는 데이터 없음)")
        return []
    
    # top_stocks에서 종목 코드와 이름의 매핑을 생성
    top_stock_names = { stock.get("mksc_shrn_iscd"): stock.get("hts_kor_isnm", "N/A") for stock in top_stocks }
    
    candidates = []
    SCORE_THRESHOLD = 2  # 최소 점수 기준

    for stock in top_stocks:
        code = stock.get("mksc_shrn_iscd")
        if not code:
            continue

        # 주식 기본 정보 조회 
        info_data = get_stock_info(code)
        if not info_data:
            send_message(f"❌ {code} 상세 정보 없음")
            continue

        if isinstance(info_data, list):
            info = info_data[0]
        elif isinstance(info_data, dict):
            info = info_data
        else:
            send_message(f"❌ {code} 상세 정보 형식 오류")
            continue

      
        stock_name = info.get("hts_kor_isnm", "N/A")
      
        if stock_name == "N/A":
            stock_name = top_stock_names.get(code, "N/A")

        try:
            current_price = float(info["stck_prpr"])
            listed_shares = float(info["lstn_stcn"])
            market_cap = current_price * listed_shares
        except Exception as e:
            send_message(f"❌ {stock_name} ({code}) 정보 계산 오류: {e}")
            continue

        score = 0
        if market_cap >= MARKET_CAP_THRESHOLD and current_price >= PRICE_THRESHOLD:
            score += 1

        monthly_data = get_monthly_data(code)
        daily_data = get_daily_data(code)
        if not monthly_data or not daily_data:
            send_message(f"❌ {stock_name} ({code}) 차트 데이터 없음")
            continue

        try:
            monthly_changes = [float(candle["prdy_ctrt"]) for candle in monthly_data]
            avg_monthly_change = np.mean(monthly_changes)
            if abs(avg_monthly_change) <= MONTHLY_VOLATILITY_THRESHOLD:
                score += 1
        except Exception as e:
            send_message(f"❌ {stock_name} ({code}) 차트 데이터 오류: {e}")
            continue

        try:
            avg_30d_volume = np.mean([int(candle["acml_vol"]) for candle in daily_data])
            today_volume = int(daily_data[0]["acml_vol"])
            if today_volume >= avg_30d_volume * TRADE_VOLUME_INCREASE_RATIO:
                score += 1
        except Exception as e:
            send_message(f"❌ {stock_name} ({code}) 거래량 데이터 오류: {e}")
            continue

        try:
            today_close = int(daily_data[0]["stck_clpr"])
            prev_close = int(daily_data[1]["stck_clpr"])
            if (today_close - prev_close) / prev_close >= CLOSE_PRICE_INCREASE_RATIO:
                score += 1
        except Exception as e:
            send_message(f"❌ {stock_name} ({code}) 종가 데이터 오류: {e}")
            continue

        if score >= SCORE_THRESHOLD:
            candidates.append((code, stock_name, score))
            send_message(f"✅ 후보 추가: {stock_name} ({code}) - 점수: {score}")
        else:
            send_message(f"ℹ️ 후보 탈락: {stock_name} ({code}) - 점수: {score}")

    candidates.sort(key=lambda x: x[2], reverse=True)
    selected_stocks = [code for (code, name, score) in candidates]
    return selected_stocks



def calculate_macd(prices):
    short_ema = np.mean(prices[-MACD_SHORT:])
    long_ema = np.mean(prices[-MACD_LONG:])
    return short_ema - long_ema

def get_moving_average(code, period=MOVING_AVG_PERIOD):
    # 코드 포맷: 12자리로 변환
    formatted_code = code.zfill(12)
    daily_data = get_daily_data(formatted_code)
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



def set_interest_stocks(selected_stocks):
    global symbol_list
    symbol_list = selected_stocks

def auto_trade():

    send_message("🚀 자동 매매 시작")
    global symbol_list  # 전역 변수 symbol_list를 사용

    # 최초 한 번만 관심종목 선정
    if not symbol_list:
        symbol_list = select_interest_stocks()
    # 이후에는 선정된 종목 목록을 그대로 사용
    while True:
        # 관심종목 리스트를 기반으로 자동 매매 로직 실행
        for code in symbol_list:
            prices = [get_current_price(code) for _ in range(60)]
            macd = calculate_macd(prices)
            moving_avg = get_moving_average(code)
            current_price = prices[-1]

            if macd > 0 and current_price <= moving_avg:
                if code not in entry_prices:
                    send_message(f"💰 {code} 매수 신호 발생! 가격: {current_price}원")
                    if buy_stock(code, 1):
                        entry_prices[code] = current_price
                else:
                    send_message(f"ℹ️ {code} 이미 보유 중입니다. 현재 매수가: {entry_prices[code]}원")

            if code in entry_prices:
                entry_price = entry_prices[code]
                if current_price <= entry_price * 0.98:
                    send_message(f"❌ {code} 손절! 가격: {current_price}원 (매수가: {entry_price}원)")
                    if sell_stock(code, 1):
                        del entry_prices[code]
                elif current_price >= entry_price * 1.2:
                    send_message(f"📈 {code} 익절! 가격: {current_price}원 (매수가: {entry_price}원)")
                    if sell_stock(code, 1):
                        del entry_prices[code]
        time.sleep(60)
