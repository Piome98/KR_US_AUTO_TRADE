# 자동 매매 로직
# korea_stock_auto.trading.real_trade.kr_trade_logic

import time
import datetime
import numpy as np
from korea_stock_auto.trading.real_trade.kr_stock_api import *  # buy_stock, sell_stock will be imported locally as needed
from korea_stock_auto.utility.kr_config import *
from korea_stock_auto.utility.kr_utils import *

# (모듈 상단에서 real_time_ws의 get_current_price 임포트 제거하여 순환 참조 방지)

# 📌 관심 종목 필터링 기준
TRADE_VOLUME_THRESHOLD = 1_000_000  # 100만 주 이상
PRICE_THRESHOLD = 3_000  # 현재가 3,000원 이상
MARKET_CAP_THRESHOLD = 400_000_000_000  # 4,000억 원 이상
MONTHLY_VOLATILITY_THRESHOLD = 10  # 10개월 등락률 평균 ±10% 이내
TRADE_VOLUME_INCREASE_RATIO = 4.0  # 거래량 400% 증가 조건
CLOSE_PRICE_INCREASE_RATIO = 0.05  # 종가 5% 이상 상승 조건

# ✅ 이동평균선 및 MACD 설정
MACD_SHORT = 5
MACD_LONG = 60
MOVING_AVG_PERIOD = 20  # 20일 이동평균선

# 📌 포트폴리오 비율 조정
buy_percentage = 0.25  # 종목당 매수 비율
target_buy_count = 4  # 매수할 종목 수

# 전역 딕셔너리: 각 종목의 매수가(구매 시점의 가격)를 저장
entry_prices = {}

# ✅ 관심종목 필터링 함수
def select_interest_stocks():
    send_message("📊 관심 종목 선정 시작")
    top_stocks = get_top_traded_stocks()
    
    if not top_stocks:
        send_message("⚠️ 관심 종목 없음 (API 응답 실패 또는 데이터 없음)")
        return []

    selected_stocks = []
    for stock in top_stocks:
        code = stock.get("mksc_shrn_iscd")
        if not code:
            continue

        # 📌 주식 기본 정보 조회
        info = get_stock_info(code)
        if not info:
            continue

        # 📌 차트 데이터 조회 (월봉 & 일봉)
        monthly_data = get_monthly_data(code)
        daily_data = get_daily_data(code)

        if not monthly_data or not daily_data:
            continue

        # 📌 10개월 평균 등락율 계산
        monthly_changes = [(float(candle["prdy_ctrt"])) for candle in monthly_data]
        avg_monthly_change = np.mean(monthly_changes)
        
        if not (-MONTHLY_VOLATILITY_THRESHOLD <= avg_monthly_change <= MONTHLY_VOLATILITY_THRESHOLD):
            continue

        # 📌 시가총액, 현재가 조건 필터링
        if (info["market_cap"] < MARKET_CAP_THRESHOLD or info["current_price"] < PRICE_THRESHOLD):
            continue

        # 📌 거래량 400% 증가 필터링
        avg_30d_volume = np.mean([int(candle["acml_vol"]) for candle in daily_data])
        today_volume = int(daily_data[0]["acml_vol"])

        if today_volume < avg_30d_volume * TRADE_VOLUME_INCREASE_RATIO:
            continue

        # 📌 종가 5% 이상 상승 조건 필터링
        today_close = int(daily_data[0]["stck_clpr"])
        prev_close = int(daily_data[1]["stck_clpr"])

        if (today_close - prev_close) / prev_close < CLOSE_PRICE_INCREASE_RATIO:
            continue

        # ✅ 필터링 완료된 종목 추가
        selected_stocks.append(code)
        send_message(f"✅ 관심 종목 추가: {code}")

    return selected_stocks


# ✅ MACD 계산 함수
def calculate_macd(prices):
    short_ema = np.mean(prices[-MACD_SHORT:])
    long_ema = np.mean(prices[-MACD_LONG:])
    return short_ema - long_ema


# ✅ 이동평균선 조회 함수 -> (고가 + 저가 + 종가) / 3, 지수 추종
def get_moving_average(code, period=MOVING_AVG_PERIOD):
    daily_data = get_daily_data(code)
    if not daily_data or len(daily_data) < period:
        return None

    # (고가 + 저가 + 종가) / 3 계산
    typical_prices = [
        (int(candle["stck_hgpr"]) + int(candle["stck_lwpr"]) + int(candle["stck_clpr"])) / 3
        for candle in daily_data[:period]
    ]

    # 지수 이동평균(EMA) 계산
    alpha = 2 / (period + 1)  # EMA 가중치
    ema = typical_prices[0]  # 첫 번째 값 초기화

    for price in typical_prices[1:]:
        ema = (price * alpha) + (ema * (1 - alpha))  # EMA 공식 적용

    return ema  # 최신 EMA 값 반환


# ✅ 실시간 매매 로직
def auto_trade():
    from korea_stock_auto.real_time_ws import get_current_price  # local import to avoid circular dependency
    send_message("🚀 자동 매매 시작")
    interest_stocks = select_interest_stocks()

    # ✅ 관심종목이 없으면 실행 중단
    if not interest_stocks:
        send_message("⚠️ 관심 종목이 없습니다. 자동 매매 중단.")
        return

    while True:
        for code in interest_stocks:
            prices = [get_current_price(code) for _ in range(60)]
            macd = calculate_macd(prices)
            moving_avg = get_moving_average(code)
            current_price = prices[-1]

            # ✅ 20일 이동평균선 터치 시 매수 조건
            if macd > 0 and current_price <= moving_avg:
                # 이미 보유 중이 아니면 매수
                if code not in entry_prices:
                    send_message(f"💰 {code} 매수 신호 발생! 가격: {current_price}원")
                    if buy_stock(code, 1):
                        entry_prices[code] = current_price
                else:
                    send_message(f"ℹ️ {code} 이미 보유 중입니다. 현재 매수가: {entry_prices[code]}원")

            # ✅ 손절 및 익절 조건 (매수가가 있을 때만 확인)
            if code in entry_prices:
                entry_price = entry_prices[code]
                # 손절 조건: 매수가 대비 2% 하락
                if current_price <= entry_price * 0.98:
                    send_message(f"❌ {code} 손절! 가격: {current_price}원 (매수가: {entry_price}원)")
                    if sell_stock(code, 1):
                        del entry_prices[code]
                # 익절 조건: 매수가 대비 20% 상승
                elif current_price >= entry_price * 1.2:
                    send_message(f"📈 {code} 익절! 가격: {current_price}원 (매수가: {entry_price}원)")
                    if sell_stock(code, 1):
                        del entry_prices[code]

        time.sleep(60)
