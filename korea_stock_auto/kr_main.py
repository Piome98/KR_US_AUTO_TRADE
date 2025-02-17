# 프로그램 실행 파일
# korea_stock_auto.kr_main

from korea_stock_auto.trading.real_trade.kr_trade_logic import auto_trade, select_interest_stocks, set_interest_stocks
from korea_stock_auto.real_time_ws import run_realtime_ws
from korea_stock_auto.utility.kr_utils import send_message

if __name__ == '__main__':
    send_message("프로그램 시작")
    
    # 최초 관심종목 선정 후 전역 변수에 저장
    interest_stocks = select_interest_stocks()
    if not interest_stocks:
        print("⚠️ 관심 종목이 없습니다. 프로그램을 종료합니다.")
        exit(1)
    else:
        print(f"✅ 관심 종목: {interest_stocks}")
        # 전역 관심종목 리스트 업데이트
        set_interest_stocks(interest_stocks)
        try:
            run_realtime_ws()  
        except Exception as e:
            print(f"❌ 웹소켓 실행 중 오류 발생: {e}")
            exit(1)
        auto_trade()
