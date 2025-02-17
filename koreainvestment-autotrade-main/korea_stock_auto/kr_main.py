# 프로그램 실행 파일
# korea_stock_auto.trading.real_trade.kr_main

from korea_stock_auto.trading.real_trade.kr_trade_logic import auto_trade, select_interest_stocks
from korea_stock_auto.real_time_ws import run_realtime_ws
from korea_stock_auto.utility.kr_utils import send_message

if __name__ == '__main__':
    send_message("프로그램 시작")
    
    # ✅ 관심 종목 필터링을 먼저 수행
    interest_stocks = select_interest_stocks()

    if not interest_stocks:
        print("⚠️ 관심 종목이 없습니다. 프로그램을 종료합니다.")
    else:
        print(f"✅ 관심 종목: {interest_stocks}")

        try:
            # ✅ 관심 종목이 있는 경우에만 웹소켓 실행
            run_realtime_ws()
        except Exception as e:
            print(f"❌ 웹소켓 실행 중 오류 발생: {e}")
            exit(1)  # 웹소켓 실패 시 프로그램 종료

        # ✅ 자동 매매 실행
        auto_trade()
