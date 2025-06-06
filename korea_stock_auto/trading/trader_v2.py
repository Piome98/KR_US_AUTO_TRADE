"""
한국 주식 자동매매 - 메인 트레이더 V2 (서비스 계층 적용)

3단계 개선: God Object 문제 해결
- TradingService: 매매 실행 로직
- PortfolioService: 포트폴리오 관리
- MonitoringService: 모니터링 및 알림
- MarketDataService: 시장 데이터 관리

단일 책임 원칙 적용으로 유지보수성과 테스트 가능성을 크게 향상시켰습니다.
"""

import logging
import time
from typing import Dict, List, Any, Optional

from korea_stock_auto.config import AppConfig
from korea_stock_auto.services import (
    TradingService, PortfolioService, 
    MonitoringService, MarketDataService
)
from korea_stock_auto.trading.stock_selector import StockSelector
from korea_stock_auto.utils.utils import send_message

logger = logging.getLogger("stock_auto")


class TraderV2:
    """
    자동매매 트레이더 V2 (서비스 계층 적용)
    
    기존 Trader 클래스의 God Object 문제를 해결하기 위해
    책임을 서비스별로 분리한 새로운 트레이더입니다.
    
    핵심 개선사항:
    - 단일 책임 원칙 적용
    - 서비스 계층을 통한 관심사 분리
    - 테스트 용이성 향상
    - 코드 재사용성 향상
    """
    
    def __init__(self,
                 config: AppConfig,
                 trading_service: TradingService,
                 portfolio_service: PortfolioService,
                 monitoring_service: MonitoringService,
                 market_data_service: MarketDataService,
                 selector: StockSelector):
        """
        트레이더 V2 초기화
        
        Args:
            config: 애플리케이션 설정
            trading_service: 매매 실행 서비스
            portfolio_service: 포트폴리오 관리 서비스
            monitoring_service: 모니터링 서비스
            market_data_service: 시장 데이터 서비스
            selector: 종목 선택기
        """
        self.config = config
        self.trading_service = trading_service
        self.portfolio_service = portfolio_service
        self.monitoring_service = monitoring_service
        self.market_data_service = market_data_service
        self.selector = selector
        
        # 기본 설정
        self.target_buy_count = config.trading.target_buy_count
        self.strategy_run_interval = config.system.strategy_run_interval
        
        # 상태 관리
        self.symbol_list: List[str] = []
        self.last_strategy_run_time: float = 0.0
        
        logger.info("TraderV2 초기화 완료 (서비스 계층 적용)")
    
    def initialize_trading(self) -> bool:
        """
        트레이딩 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            logger.info("TraderV2 트레이딩 초기화 시작")
            
            # 포트폴리오 초기화
            if not self.portfolio_service.initialize():
                logger.error("포트폴리오 초기화 실패")
                return False
            
            # 시장 데이터 서비스 초기화
            if not self.market_data_service.initialize():
                logger.error("시장 데이터 서비스 초기화 실패")
                return False
            
            # 초기 상태 로깅
            portfolio_status = self.portfolio_service.get_portfolio_status()
            logger.info(f"초기 포트폴리오: 현금 {portfolio_status['cash']:,}원, "
                       f"보유 종목 {len(portfolio_status['positions'])}개")
            
            send_message(f"[초기화 완료] 현금: {portfolio_status['cash']:,}원, "
                        f"보유 종목: {len(portfolio_status['positions'])}개")
            
            # 모니터링 시작
            self.monitoring_service.start_monitoring()
            
            logger.info("TraderV2 트레이딩 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"트레이딩 초기화 중 오류: {e}")
            return False
    
    def select_interest_stocks(self) -> List[str]:
        """
        관심 종목 선정
        
        Returns:
            List[str]: 선정된 관심 종목 코드 리스트
        """
        try:
            self.symbol_list = self.selector.select_interest_stocks(self.target_buy_count)
            
            if self.symbol_list:
                logger.info(f"관심 종목 {len(self.symbol_list)}개 선정 완료")
                self.monitoring_service.log_info(f"관심 종목 선정: {', '.join(self.symbol_list)}")
            else:
                logger.warning("관심 종목 선정 실패")
                self.monitoring_service.log_warning("관심 종목 선정 실패")
            
            return self.symbol_list
            
        except Exception as e:
            logger.error(f"관심 종목 선정 중 오류: {e}")
            return []
    
    def run_single_trading_cycle(self) -> bool:
        """
        단일 매매 사이클 실행
        
        Returns:
            bool: 사이클 실행 성공 여부
        """
        try:
            current_time = time.time()
            logger.info("TraderV2 단일 매매 사이클 시작...")
            
            # 1. 포트폴리오 상태 업데이트
            if not self.portfolio_service.update_portfolio():
                logger.warning("포트폴리오 업데이트 실패")
                return False
            
            # 2. 시장 데이터 업데이트
            portfolio_status = self.portfolio_service.get_portfolio_status()
            all_symbols = list(set(self.symbol_list + list(portfolio_status['positions'].keys())))
            
            if not self.market_data_service.update_market_data(all_symbols):
                logger.warning("시장 데이터 업데이트 실패")
                return False
            
            # 3. 전략 실행 간격 확인
            if current_time - self.last_strategy_run_time < self.strategy_run_interval:
                logger.info(f"전략 실행 간격({self.strategy_run_interval}초) 미달, 이번 사이클 건너뜀")
                return True
            
            self.last_strategy_run_time = current_time
            
            # 4. 관심 종목 선정 (필요시)
            if not self.symbol_list:
                logger.info("관심 종목이 없어 선정을 시도합니다.")
                self.select_interest_stocks()
                if self.symbol_list:
                    # 새로 선정된 종목들의 시장 데이터 업데이트
                    self.market_data_service.update_market_data(self.symbol_list)
            
            # 5. 매수 로직 실행
            self._execute_buy_logic()
            
            # 6. 매도 로직 실행
            self._execute_sell_logic()
            
            # 7. 포트폴리오 상태 로깅
            self.monitoring_service.log_portfolio_status(
                self.portfolio_service.get_portfolio_status()
            )
            
            logger.info("TraderV2 단일 매매 사이클 완료")
            return True
            
        except Exception as e:
            logger.error(f"매매 사이클 실행 중 오류: {e}")
            return False
    
    def _execute_buy_logic(self) -> None:
        """매수 로직 실행"""
        try:
            portfolio_status = self.portfolio_service.get_portfolio_status()
            current_positions_count = len(portfolio_status['positions'])
            available_cash = portfolio_status['cash']
            
            # 매수 가능 조건 확인
            if current_positions_count >= self.target_buy_count:
                logger.info(f"목표 보유 종목 수({self.target_buy_count}개) 도달, 추가 매수 안함")
                return
            
            if available_cash < self.config.trading.min_order_amount:
                logger.info(f"매수 가능 금액({available_cash:,.0f}원)이 최소 주문 금액 미만")
                return
            
            logger.info(f"매수 대상 종목 탐색... 관심 종목 수: {len(self.symbol_list)}")
            
            # 각 관심 종목에 대해 매수 시도
            for code in self.symbol_list:
                if code in portfolio_status['positions']:
                    continue  # 이미 보유 중인 종목은 건너뜀
                
                # 현재가 정보 가져오기
                current_price = self.market_data_service.get_price_only(code)
                if not current_price or current_price <= 0:
                    logger.warning(f"{code} 현재가 정보 없음, 매수 건너뜀")
                    continue
                
                # 매수 주문 실행
                success, result = self.trading_service.execute_buy_order(
                    code=code,
                    current_price=current_price,
                    available_cash=available_cash,
                    current_positions_count=current_positions_count
                )
                
                if success:
                    # 포트폴리오 업데이트
                    self.portfolio_service.add_position(code, current_price, result.get('quantity', 0))
                    current_positions_count += 1
                    available_cash -= result.get('order_amount', 0)
                    
                    logger.info(f"{code} 매수 성공")
                    
                    # 목표 종목 수 도달 시 중단
                    if current_positions_count >= self.target_buy_count:
                        logger.info("목표 보유 종목 수 도달, 추가 매수 중단")
                        break
                else:
                    logger.debug(f"{code} 매수 건너뜀: {result.get('reason', '알 수 없음')}")
            
        except Exception as e:
            logger.error(f"매수 로직 실행 중 오류: {e}")
    
    def _execute_sell_logic(self) -> None:
        """매도 로직 실행"""
        try:
            portfolio_status = self.portfolio_service.get_portfolio_status()
            positions = portfolio_status['positions']
            
            if not positions:
                logger.info("보유 종목이 없어 매도 진행 안함")
                return
            
            logger.info(f"매도 대상 종목 탐색... 보유 종목 수: {len(positions)}")
            
            # 각 보유 종목에 대해 매도 검토
            for code, position_info in list(positions.items()):
                # 현재가 정보 가져오기
                current_price = self.market_data_service.get_price_only(code)
                if not current_price or current_price <= 0:
                    logger.warning(f"{code} 현재가 정보 없음, 매도 건너뜀")
                    continue
                
                entry_price = position_info.get('entry_price', 0)
                quantity = position_info.get('quantity', 0)
                
                # 매도 주문 실행
                success, result = self.trading_service.execute_sell_order(
                    code=code,
                    current_price=current_price,
                    entry_price=entry_price,
                    quantity=quantity
                )
                
                if success:
                    # 포트폴리오에서 포지션 제거
                    self.portfolio_service.remove_position(code)
                    logger.info(f"{code} 매도 성공")
                else:
                    logger.debug(f"{code} 매도 건너뜀: {result.get('reason', '알 수 없음')}")
            
        except Exception as e:
            logger.error(f"매도 로직 실행 중 오류: {e}")
    
    def sell_all_positions(self) -> bool:
        """
        모든 포지션 매도
        
        Returns:
            bool: 전량 매도 성공 여부
        """
        try:
            portfolio_status = self.portfolio_service.get_portfolio_status()
            positions = portfolio_status['positions']
            
            if not positions:
                logger.info("매도할 보유 종목이 없습니다.")
                return True
            
            # 현재가 정보 수집
            current_prices = {}
            for code in positions.keys():
                price = self.market_data_service.get_price_only(code)
                current_prices[code] = price
            
            # 전량 매도 실행
            success_count, fail_count, fail_list = self.trading_service.execute_sell_all_positions(
                positions, current_prices
            )
            
            # 성공한 포지션들 제거
            for code in positions.keys():
                if code not in fail_list:
                    self.portfolio_service.remove_position(code)
            
            # 결과 로깅
            total_count = len(positions)
            self.monitoring_service.log_info(
                f"전량 매도 완료: {success_count}/{total_count} 성공"
            )
            
            return fail_count == 0
            
        except Exception as e:
            logger.error(f"전량 매도 중 오류: {e}")
            return False
    
    def run_trading(self, max_cycles: int = 0) -> None:
        """
        자동 매매 실행
        
        Args:
            max_cycles: 최대 거래 사이클 수 (0이면 무제한)
        """
        try:
            logger.info("TraderV2 자동 매매 시작")
            
            # 1. 트레이딩 초기화
            if not self.initialize_trading():
                logger.error("트레이딩 초기화 실패")
                return
            
            # 2. 관심 종목 선정
            self.select_interest_stocks()
            
            # 3. 매매 사이클 실행
            logger.info("매매 사이클 시작")
            cycle_count = 0
            
            while True:
                cycle_count += 1
                
                # 최대 사이클 수 확인
                if max_cycles > 0 and cycle_count > max_cycles:
                    logger.info(f"최대 거래 사이클 수({max_cycles}) 도달")
                    break
                
                logger.info(f"===== TraderV2 거래 사이클 {cycle_count} 시작 =====")
                
                # 단일 사이클 실행
                success = self.run_single_trading_cycle()
                if not success:
                    logger.warning(f"사이클 {cycle_count} 실행 실패")
                
                logger.info(f"===== TraderV2 거래 사이클 {cycle_count} 종료 =====")
                
                # 대기
                logger.info("다음 거래 사이클까지 대기...")
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("사용자에 의한 거래 중단")
        except Exception as e:
            logger.error(f"거래 실행 중 오류 발생: {e}")
            send_message(f"[오류] TraderV2 거래 실행 중 오류: {e}", config.notification.discord_webhook_url)
        finally:
            self._finalize_trading()
    
    def _finalize_trading(self) -> None:
        """거래 종료 정리"""
        try:
            logger.info("TraderV2 거래 종료 정리 시작")
            
            # 최종 포트폴리오 상태 로깅
            portfolio_status = self.portfolio_service.get_portfolio_status()
            self.monitoring_service.log_portfolio_status(portfolio_status)
            
            # 모니터링 종료
            self.monitoring_service.stop_monitoring()
            
            logger.info("TraderV2 자동 매매 정리 작업 완료")
            send_message("[TraderV2] 자동 매매 종료", config.notification.discord_webhook_url)
            
        except Exception as e:
            logger.error(f"거래 종료 정리 중 오류: {e}") 