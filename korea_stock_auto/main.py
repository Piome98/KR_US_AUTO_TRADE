"""
한국 주식 자동매매 - 메인 모듈
"""

import argparse
import logging
import sys
from datetime import datetime

from korea_stock_auto.config import setup_logger
from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.utils.utils import send_message

# 로깅 설정
logger = logging.getLogger("stock_auto")
setup_logger()

def show_market_info(api_client):
    """
    현재 시장 상황 출력
    
    Args:
        api_client: API 클라이언트 객체
    """
    try:
        # 시장 상황 종합 조회
        market_status = api_client.get_market_status()
        summary = market_status["summary"]
        
        # 코스피 정보
        kospi = market_status["kospi"]
        kospi_info = (f"KOSPI: {kospi['current']:,.2f} "
                      f"({kospi['status']} {abs(kospi['change_rate']):,.2f}%)")
        
        # 코스닥 정보
        kosdaq = market_status["kosdaq"]
        kosdaq_info = (f"KOSDAQ: {kosdaq['current']:,.2f} "
                       f"({kosdaq['status']} {abs(kosdaq['change_rate']):,.2f}%)")
        
        # 결과 출력
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"=== 시장 정보 ({now}) ===")
        logger.info(f"시장 분위기: {summary['market_mood']}")
        logger.info(kospi_info)
        logger.info(kosdaq_info)
        
        return True
    except Exception as e:
        logger.error(f"시장 정보 조회 실패: {e}", exc_info=True)
        send_message(f"[오류] 시장 정보 조회 실패: {e}")
        return False

def show_account_info(api_client):
    """
    계좌 정보 출력
    
    Args:
        api_client: API 클라이언트 객체
    """
    try:
        # 예수금 조회
        deposit = api_client.fetch_deposit()
        if deposit:
            logger.info(f"=== 계좌 정보 ===")
            logger.info(f"주문 가능 금액: {deposit['order_executable_amount']:,}원")
            logger.info(f"출금 가능 금액: {deposit['withdrawable']:,}원")
            logger.info(f"총 평가 금액: {deposit['total_balance']:,}원")
        
        # 계좌 잔고 조회
        balance = api_client.fetch_balance()
        if balance and balance["stocks"]:
            logger.info(f"=== 보유 종목 ({len(balance['stocks'])}개) ===")
            
            for stock in balance["stocks"]:
                stock_info = (
                    f"{stock['name']} ({stock['code']}): "
                    f"{stock['quantity']}주, "
                    f"평가금액: {stock['evaluation_amount']:,}원, "
                    f"수익률: {stock['earning_rate']:.2f}%, "
                    f"수익금: {stock['profit_loss']:,}원"
                )
                logger.info(stock_info)
        
        return True
    except Exception as e:
        logger.error(f"계좌 정보 조회 실패: {e}", exc_info=True)
        send_message(f"[오류] 계좌 정보 조회 실패: {e}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="한국 주식 자동매매 프로그램")
    parser.add_argument("--status", action="store_true", help="계좌 및 시장 상태 조회")
    
    args = parser.parse_args()
    
    try:
        logger.info("한국 주식 자동매매 프로그램 시작")
        
        # API 클라이언트 초기화
        api_client = KoreaInvestmentApiClient()
        
        # 단순 상태 조회 모드
        if args.status:
            logger.info("상태 조회 모드")
            show_market_info(api_client)
            show_account_info(api_client)
            return
        
        # 실제 매매 로직은 여기에 추가
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}", exc_info=True)
        send_message(f"[오류] 프로그램 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 