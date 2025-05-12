"""
한국 주식 자동매매 - 메인 스크립트
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime

from korea_stock_auto.config import TRADE_CONFIG
from korea_stock_auto.utils.utils import send_message, setup_logger
from korea_stock_auto.trading import Trader

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='한국 주식 자동매매 시스템')
    
    # 기본 설정
    parser.add_argument('--mode', type=str, default='trade',
                        choices=['trade', 'backtest', 'analyze'],
                        help='실행 모드 (trade: 실제 매매, backtest: 백테스트, analyze: 종목 분석)')
    parser.add_argument('--cycles', type=int, default=0,
                        help='매매 사이클 수 (0: 무한)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='로그 레벨')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=['macd', 'ma', 'rsi'],
                        help='매매 전략 (macd, ma, rsi)')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    args = parse_args()
    
    # 로그 설정
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'stock_auto_{datetime.now().strftime("%Y%m%d")}.log')
    setup_logger(args.log_level, log_file)
    
    # 로거 가져오기
    logger = logging.getLogger("stock_auto")
    
    # 시작 메시지
    logger.info("한국 주식 자동매매 시스템 시작")
    send_message("[시스템 시작] 한국 주식 자동매매 시스템이 시작되었습니다.")
    
    # 설정 정보 출력
    logger.info(f"실행 모드: {args.mode}")
    logger.info(f"매매 사이클 수: {'무한' if args.cycles == 0 else args.cycles}")
    
    # 전략 설정 (명령행 인자가 있는 경우 우선 적용)
    if args.strategy:
        TRADE_CONFIG["strategy"] = args.strategy
    
    try:
        # 모드에 따른 처리
        if args.mode == 'trade':
            # 트레이더 초기화
            trader = Trader()
            
            # 매매 실행
            trader.run_trading(max_cycles=args.cycles)
            
        elif args.mode == 'backtest':
            logger.info("백테스트 모드는 아직 구현되지 않았습니다.")
            send_message("[알림] 백테스트 모드는 아직 구현되지 않았습니다.")
            
        elif args.mode == 'analyze':
            logger.info("종목 분석 모드는 아직 구현되지 않았습니다.")
            send_message("[알림] 종목 분석 모드는 아직 구현되지 않았습니다.")
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 프로그램이 중단되었습니다.")
        send_message("[알림] 사용자에 의해 프로그램이 중단되었습니다.")
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}", exc_info=True)
        send_message(f"[오류] 프로그램 실행 중 오류 발생: {e}")
        
    finally:
        # 종료 메시지
        logger.info("한국 주식 자동매매 시스템 종료")
        send_message("[시스템 종료] 한국 주식 자동매매 시스템이 종료되었습니다.")

if __name__ == "__main__":
    main() 