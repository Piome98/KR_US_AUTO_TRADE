"""
한국 주식 자동매매 - 메인 실행 모듈
자동 매매 프로그램 진입점
"""

import time
import signal
import sys

from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.trading.trading import StockTrader
from korea_stock_auto.api.websocket import StockWebSocket

def signal_handler(sig, frame):
    """
    종료 시그널 처리 함수
    
    Args:
        sig: 시그널 번호
        frame: 현재 스택 프레임
    """
    send_message("프로그램 종료 신호를 받았습니다. 자원을 정리하고 종료합니다.")
    # 여기서 추가적인 정리 작업 가능
    sys.exit(0)

def main():
    """프로그램 메인 실행 함수"""
    send_message("===한국 주식 자동매매 프로그램을 시작합니다===")
    
    # 종료 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 트레이더 인스턴스 생성
        trader = StockTrader()
        
        # 관심 종목 선정
        selected_stocks = trader.select_interest_stocks()
        if not selected_stocks:
            send_message("관심 종목이 없습니다. 프로그램을 종료합니다.")
            return
        
        send_message(f"선정된 관심 종목: {selected_stocks}")
        
        # 웹소켓 인스턴스 생성 및 시작
        websocket = StockWebSocket()
        websocket.set_symbol_list(selected_stocks)
        
        if not websocket.start():
            send_message("웹소켓 연결에 실패했습니다. 프로그램을 종료합니다.")
            return
        
        # 웹소켓 연결 후 잠시 대기
        send_message("웹소켓 연결 후 5초 대기...")
        time.sleep(5)
        
        # 트레이딩 시작
        trader.run_trading()
        
    except Exception as e:
        send_message(f"[오류 발생] {e}")
    
    send_message("===한국 주식 자동매매 프로그램이 종료되었습니다===")

if __name__ == "__main__":
    main() 