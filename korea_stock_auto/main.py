"""
한국 주식 자동매매 - 메인 스크립트
의존성 주입 컨테이너를 사용하여 서비스들을 관리합니다.
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime

from korea_stock_auto.config import get_config, AppConfig
from korea_stock_auto.utils.utils import send_message, setup_logger
from korea_stock_auto.service_factory import configure_services, get_service
from korea_stock_auto.trading.trader_v2 import TraderV2


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
    """메인 함수 (의존성 주입 적용)"""
    # 명령행 인자 파싱
    args = parse_args()
    
    # 로그 설정
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'stock_auto_{datetime.now().strftime("%Y%m%d")}.log')
    setup_logger(args.log_level, log_file)
    
    # 로거 가져오기
    logger = logging.getLogger("stock_auto")
    
    # 설정 로드 및 검증
    try:
        config = get_config()
        if not config.validate_all():
            logger.error("설정 검증 실패")
            send_message("[오류] 설정 검증에 실패했습니다.", config.notification.discord_webhook_url)
            return
    except Exception as e:
        logger.error(f"설정 로드 실패: {e}")
        send_message(f"[오류] 설정 로드에 실패했습니다: {e}", config.notification.discord_webhook_url)
        return
    
    # 시작 메시지
    logger.info("한국 주식 자동매매 시스템 시작")
    send_message("[시스템 시작] 한국 주식 자동매매 시스템이 시작되었습니다.", config.notification.discord_webhook_url)
    
    # 설정 정보 출력
    logger.info(f"실행 모드: {args.mode}")
    logger.info(f"매매 사이클 수: {'무한' if args.cycles == 0 else args.cycles}")
    
    # 전략 설정 (명령행 인자가 있는 경우 우선 적용)
    if args.strategy:
        config.trading.strategy = args.strategy
        logger.info(f"명령행에서 전략 변경: {args.strategy}")
    
    try:
        # DI 컨테이너 설정
        configure_services()
        logger.info("의존성 주입 컨테이너 설정 완료")
        
        # 모드에 따른 처리
        if args.mode == 'trade':
            # TraderV2 가져오기 (DI 컨테이너에서)
            trader = get_service(TraderV2)
            
            # API 설정 정보 출력
            api_type = "실전투자" if config.use_realtime_api else "모의투자"
            trade_mode = "실제매매" if config.use_realtime_api else "시뮬레이션"
            
            logger.info(f"API 설정: {api_type}")
            logger.info(f"거래 모드: {trade_mode}")
            logger.info(f"설정된 전략: {config.trading.strategy}")
            logger.info("TraderV2 (서비스 계층) 사용")
            
            send_message(f"[시스템 설정] API: {api_type}, 거래모드: {trade_mode}, 전략: {config.trading.strategy}, TraderV2 활성화")
            
            # 매매 실행
            trader.run_trading(max_cycles=args.cycles)
            
        elif args.mode == 'backtest':
            logger.info("백테스트 모드는 아직 구현되지 않았습니다.")
            send_message("[알림] 백테스트 모드는 아직 구현되지 않았습니다.", config.notification.discord_webhook_url)
            
        elif args.mode == 'analyze':
            logger.info("종목 분석 모드는 아직 구현되지 않았습니다.")
            send_message("[알림] 종목 분석 모드는 아직 구현되지 않았습니다.", config.notification.discord_webhook_url)
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 프로그램이 중단되었습니다.")
        send_message("[알림] 사용자에 의해 프로그램이 중단되었습니다.", config.notification.discord_webhook_url)
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}", exc_info=True)
        send_message(f"[오류] 프로그램 실행 중 오류 발생: {e}", config.notification.discord_webhook_url)
        
    finally:
        # 종료 메시지
        logger.info("한국 주식 자동매매 시스템 종료")
        send_message("[시스템 종료] 한국 주식 자동매매 시스템이 종료되었습니다.", config.notification.discord_webhook_url)

if __name__ == "__main__":
    main() 