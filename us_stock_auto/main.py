"""
미국 주식 자동매매 - 메인 실행 모듈
자동 매매 프로그램 진입점
"""

import time

from us_stock_auto.utils import send_message, is_market_open
from us_stock_auto.trading import StockTrader

def main():
    """프로그램 메인 실행 함수"""
    send_message("===미국 주식 자동매매 프로그램을 시작합니다===")
    
    try:
        # 트레이더 인스턴스 생성
        trader = StockTrader()
        
        # 매매 루프 실행
        while True:
            if not is_market_open():
                send_message("현재는 주식 거래 시간이 아닙니다. 프로그램을 종료합니다.")
                break
                
            # 매매 사이클 실행
            if not trader.run_trading_cycle():
                send_message("매매 종료 조건이 충족되어 프로그램을 종료합니다.")
                break
                
            # 잠시 대기
            time.sleep(3)
            
    except Exception as e:
        send_message(f"[오류 발생] {e}")
    
    send_message("===미국 주식 자동매매 프로그램이 종료되었습니다===")

if __name__ == "__main__":
    main() 