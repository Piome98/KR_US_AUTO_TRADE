# korea_stock_auto.kr_main

from korea_stock_auto.trading.real_trade.kr_trade_logic import auto_trade, select_interest_stocks
from korea_stock_auto.real_time_ws import run_realtime_ws
from korea_stock_auto.utility.kr_utils import send_message
from korea_stock_auto.shared.global_vars import symbol_list  # 전역 변수 모듈

if __name__ == '__main__':
    send_message("프로그램 시작")
    
    # 최초 관심종목 선정 후 전역 변수 업데이트 (in-place 수정)
    selected_stocks = select_interest_stocks()
    if not selected_stocks:
        print("⚠️ 관심 종목이 없습니다. 프로그램을 종료합니다.")
        exit(1)
    else:
        print(f"✅ 관심 종목: {selected_stocks}")
        symbol_list[:] = selected_stocks  # 기존 리스트를 업데이트트
    
    try:
        run_realtime_ws()  
    except Exception as e:
        print(f"❌ 웹소켓 실행 중 오류 발생: {e}")
        exit(1)
    
    auto_trade()
    send_message("프로그램 종료")