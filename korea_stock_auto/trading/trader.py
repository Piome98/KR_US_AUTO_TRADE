"""
한국 주식 자동매매 - 메인 트레이더 모듈
자동 매매 시스템의 핵심 클래스 정의
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from korea_stock_auto.config import TRADE_CONFIG
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.trading.stock_selector import StockSelector
from korea_stock_auto.trading.technical_analyzer import TechnicalAnalyzer
from korea_stock_auto.trading.trade_executor import TradeExecutor
from korea_stock_auto.trading.risk_manager import RiskManager
from korea_stock_auto.trading.trading_strategy import TradingStrategy, MACDStrategy, MovingAverageStrategy, RSIStrategy

logger = logging.getLogger("stock_auto")

class Trader:
    """국내 주식 자동 매매 트레이더 클래스"""
    
    def __init__(self):
        """트레이더 초기화"""
        # API 클라이언트 생성
        self.api = KoreaInvestmentApiClient()
        
        # 거래 상태 변수
        self.bought_list = []  # 매수한 종목 리스트
        self.symbol_list = []  # 관심 종목 리스트
        self.total_cash = 0    # 총 현금
        self.stock_dict = {}   # 보유 종목 정보
        self.entry_prices = {} # 종목별 매수 가격
        self.buy_amount = 0    # 종목별 매수 금액
        self.buy_percentage = TRADE_CONFIG.get("buy_percentage", 0.2)  # 매수 비율 (기본: 현금의 20%)
        self.target_buy_count = TRADE_CONFIG.get("target_buy_count", 5)  # 목표 매수 종목 수
        self.soldout = False   # 전량 매도 여부
        
        # 먼저 필요한 컴포넌트들 생성
        # 기술적 분석기 생성
        self.analyzer = TechnicalAnalyzer(self.api)
        
        # 위험 관리자 생성
        self.risk_manager = RiskManager(self.api)
        
        # 매매 실행기 생성
        self.executor = TradeExecutor(self.api)
        
        # 종목 선택기 생성
        self.selector = StockSelector(self.api)
        
        # 전략 생성 - analyzer를 사용하므로 analyzer 초기화 후에 생성
        strategy_type = TRADE_CONFIG.get("strategy", "ma")
        self.strategy = self._create_strategy(strategy_type)
        
        # 계좌 정보 업데이트 주기 (초)
        self.account_update_interval = 60  # 1분
        self.last_account_update_time = 0
        
        # 초기화
        self.initialize_trading()
    
    def _create_strategy(self, strategy_type: str) -> TradingStrategy:
        """
        매매 전략 객체 생성
        
        Args:
            strategy_type: 전략 유형 (macd, ma, rsi)
            
        Returns:
            TradingStrategy: 매매 전략 객체
        """
        if strategy_type == "ma":
            return MovingAverageStrategy(self.api, self.analyzer)
        elif strategy_type == "rsi":
            return RSIStrategy(self.api, self.analyzer)
        else:  # 기본값은 MACD
            return MACDStrategy(self.api, self.analyzer)
    
    def initialize_trading(self):
        """매매 초기화"""
        # 계좌 정보 및 보유 주식 정보 동시에 조회하여 중복 API 호출 방지
        # 초기화 시에는 무조건 업데이트하도록 시간 초기화
        self.last_account_update_time = 0
        self._update_account_info(is_init=True)
        
        # 매수가 정보 초기화
        self.entry_prices = {}
        for code, stock_info in self.stock_dict.items():
            if isinstance(stock_info, dict):
                self.entry_prices[code] = stock_info.get("avg_buy_price", 0)
            else:
                logger.warning(f"{code} 종목 정보가 딕셔너리가 아닙니다: {type(stock_info)}")
        
        # 매수 금액 설정
        self.buy_amount = self.total_cash * self.buy_percentage
        
        # 위험 관리 초기화
        self.risk_manager.reset_daily_stats()
        
        logger.info(f"트레이딩 초기화 완료: 현금 {self.total_cash:,}원, 보유 종목 {len(self.bought_list)}개")
        send_message(f"[초기화 완료] 현금: {self.total_cash:,}원, 보유 종목: {len(self.bought_list)}개")
        
        if self.bought_list:
            stock_names = []
            for code in self.bought_list:
                stock_info = self.stock_dict.get(code, {})
                name = "N/A"
                if isinstance(stock_info, dict):
                    name = stock_info.get("prdt_name", code)
                stock_names.append(f"{code}({name})")
            
            logger.info(f"보유 종목: {', '.join(stock_names)}")
            send_message(f"[보유 종목] {', '.join(stock_names)}")
    
    def _update_account_info(self, is_init=False):
        """
        계좌 정보 및 보유 종목 정보 업데이트
        
        Args:
            is_init (bool): 초기화 과정에서 호출된 경우 True, 업데이트만 하는 경우 False
        """
        # 현재 시각 확인
        current_time = time.time()
        
        # 마지막 업데이트 이후 일정 시간이 경과한 경우에만 업데이트
        if current_time - self.last_account_update_time < self.account_update_interval:
            return
            
        # 계좌 잔고 조회
        if is_init:
            logger.info("계좌 잔고 및 보유 종목 정보 업데이트")
        balance_info = self.api.get_balance()
        stock_balance = self.api.get_stock_balance()
        
        # 계좌 잔고 정보 업데이트
        if balance_info:
            self.total_cash = balance_info.get("cash", 0)
            if is_init:
                logger.info(f"계좌 잔고 조회 성공: 현금={balance_info['cash']:,.0f}원, "
                           f"주식={balance_info['stocks_value']:,.0f}원, "
                           f"총평가={balance_info['total_assets']:,.0f}원, "
                           f"수익률={balance_info['total_profit_loss_rate']:.2f}%")
        else:
            logger.warning("계좌 잔고 정보 조회 실패")
            
        # 보유 종목 정보 업데이트
        if stock_balance:
            if "stocks" in stock_balance and isinstance(stock_balance["stocks"], list):
                # 보유 종목 딕셔너리 초기화
                self.stock_dict = {}
                
                # 각 종목 정보를 종목 코드를 키로 하여 딕셔너리에 저장
                for stock in stock_balance["stocks"]:
                    code = stock.get("code")
                    if code:
                        self.stock_dict[code] = stock
                
                # 보유 종목 리스트 업데이트
                self.bought_list = list(self.stock_dict.keys())
                
                if is_init:
                    logger.info(f"보유 종목 {len(self.bought_list)}개 정보 업데이트 완료")
                    logger.info(f"주식 보유 종목 조회 성공: {len(self.bought_list)}종목, 총평가액: {balance_info['total_assets']:,.0f}원")
            else:
                logger.warning("보유 종목 정보 형식이 예상과 다릅니다.")
                # 응답 형식 로깅
                logger.debug(f"stock_balance: {stock_balance}")
        else:
            logger.warning("보유 종목 정보 조회 실패")
        
        self.last_account_update_time = current_time
    
    def select_interest_stocks(self):
        """
        관심 종목 선정
        
        Returns:
            list: 선정된 관심 종목 코드 리스트
        """
        self.symbol_list = self.selector.select_interest_stocks(self.target_buy_count)
        
        if self.symbol_list:
            logger.info(f"관심 종목 {len(self.symbol_list)}개 선정 완료")
            send_message(f"[관심 종목] {len(self.symbol_list)}개 선정 완료")
        else:
            logger.warning("관심 종목 선정 실패")
            send_message("[주의] 관심 종목 선정 실패")
            
        return self.symbol_list
    
    def try_buy(self, code: str):
        """
        종목 매수 시도
        
        Args:
            code: 종목 코드
            
        Returns:
            bool: 매수 성공 여부
        """
        if code in self.bought_list:
            logger.info(f"{code} 이미 보유 중인 종목")
            return False
            
        # 현재가 조회
        current_price_info = self.api.fetch_stock_current_price(code)
        if not current_price_info:
            logger.warning(f"{code} 현재가 조회 실패")
            return False
            
        current_price = current_price_info.get("current_price", 0)
        if current_price <= 0:
            logger.warning(f"{code} 유효하지 않은 현재가: {current_price}")
            return False
            
        # 매수 시그널 확인
        if not self.strategy.should_buy(code, current_price):
            logger.info(f"{code} 매수 시그널 없음")
            return False
            
        # 매수 가능 수량 계산
        quantity = self.executor.calculate_buy_quantity(current_price, self.buy_amount)
        if quantity <= 0:
            logger.warning(f"{code} 매수 가능 수량 없음")
            return False
            
        # 노출 한도 확인
        if self.risk_manager.check_exposure_limit(code, quantity, current_price):
            logger.warning(f"{code} 노출 한도 초과로 매수 제한")
            return False
            
        # 매수 실행
        result = self.executor.execute_buy(code, current_price, quantity)
        
        if result.get("success", False):
            # 매수 성공 처리
            self.bought_list.append(code)
            self.entry_prices[code] = current_price
            
            # 계좌 정보 업데이트
            self._update_account_info()
            
            # 위험 관리 업데이트
            self.risk_manager.update_daily_pl()
            
            return True
        else:
            logger.warning(f"{code} 매수 실패: {result.get('message', '알 수 없는 오류')}")
            return False
    
    def try_sell(self, code: str):
        """
        종목 매도 시도
        
        Args:
            code: 종목 코드
            
        Returns:
            bool: 매도 성공 여부
        """
        if code not in self.bought_list:
            logger.warning(f"{code} 보유하지 않은 종목")
            return False
            
        # 보유 수량 확인
        quantity = self.stock_dict.get(code, {}).get("units", 0)
        if quantity <= 0:
            logger.warning(f"{code} 보유 수량 없음")
            return False
            
        # 현재가 조회
        current_price_info = self.api.fetch_stock_current_price(code)
        if not current_price_info:
            logger.warning(f"{code} 현재가 조회 실패")
            return False
            
        current_price = current_price_info.get("current_price", 0)
        if current_price <= 0:
            logger.warning(f"{code} 유효하지 않은 현재가: {current_price}")
            return False
            
        # 매수가 확인
        entry_price = self.entry_prices.get(code, 0)
        
        # 매도 시그널 확인
        should_sell = False
        
        # 1. 전략 기반 매도 시그널
        if self.strategy.should_sell(code, current_price, entry_price):
            logger.info(f"{code} 전략 기반 매도 시그널 발생")
            should_sell = True
            
        # 2. 트레일링 스탑
        elif self.risk_manager.check_trailing_stop(code, current_price):
            logger.info(f"{code} 트레일링 스탑 매도 시그널 발생")
            should_sell = True
            
        # 3. 손실 제한
        elif self.risk_manager.check_position_loss_limit(code, current_price, entry_price):
            logger.info(f"{code} 손실 제한 매도 시그널 발생")
            should_sell = True
            
        if not should_sell:
            logger.info(f"{code} 매도 시그널 없음")
            return False
            
        # 매도 실행
        result = self.executor.execute_sell(code, current_price, quantity)
        
        if result.get("success", False):
            # 매도 성공 처리
            self.bought_list.remove(code)
            if code in self.entry_prices:
                del self.entry_prices[code]
                
            # 계좌 정보 업데이트
            self._update_account_info()
            
            # 위험 관리 업데이트
            self.risk_manager.update_daily_pl()
            
            return True
        else:
            logger.warning(f"{code} 매도 실패: {result.get('message', '알 수 없는 오류')}")
            return False
    
    def sell_all_stocks(self):
        """
        모든 보유 주식 매도
        
        Returns:
            bool: 전체 매도 성공 여부
        """
        # 보유 종목이 없으면 종료
        if not self.bought_list:
            logger.info("매도할 보유 종목이 없습니다.")
            return True
            
        # 전체 매도 실행
        results = self.executor.sell_all_stocks(self.stock_dict)
        
        # 결과 처리
        success_count = sum(results.values())
        if success_count == len(results):
            # 모든 종목 매도 성공
            self.bought_list = []
            self.entry_prices = {}
            self.soldout = True
            
            # 계좌 정보 업데이트
            self._update_account_info()
            
            # 위험 관리 업데이트
            self.risk_manager.update_daily_pl()
            
            logger.info("모든 보유 주식 매도 성공")
            return True
        else:
            # 일부 종목 매도 실패
            logger.warning(f"일부 종목 매도 실패: 성공 {success_count}/{len(results)}")
            
            # 성공한 종목 처리
            for code, success in results.items():
                if success and code in self.bought_list:
                    self.bought_list.remove(code)
                    if code in self.entry_prices:
                        del self.entry_prices[code]
            
            # 계좌 정보 업데이트
            self._update_account_info()
            
            # 위험 관리 업데이트
            self.risk_manager.update_daily_pl()
            
            return False
    
    def run_single_trading_cycle(self):
        """
        단일 매매 사이클 실행
        
        Returns:
            bool: 매매 사이클 성공 여부
        """
        try:
            logger.info("매매 사이클 시작")
            
            # 위험 수준 확인
            if self.risk_manager.should_stop_trading():
                logger.warning("위험 수준으로 인해 매매 중단")
                self.sell_all_stocks()
                return False
                
            # 관심 종목 선정 (아직 선정되지 않은 경우)
            if not self.symbol_list:
                self.select_interest_stocks()
                
            if not self.symbol_list:
                logger.warning("관심 종목이 없어 매매 사이클 중단")
                return False
                
            # 보유 종목 매도 검토
            for code in list(self.bought_list):  # 복사본으로 반복 (매도 시 리스트 변경됨)
                self.try_sell(code)
                time.sleep(1)  # API 호출 간격 유지
                
            # 매수 가능 종목 수 계산
            available_slots = self.target_buy_count - len(self.bought_list)
            
            # 매수 가능한 경우 매수 시도
            if available_slots > 0 and self.total_cash >= self.buy_amount:
                # 아직 매수하지 않은 관심 종목에 대해 매수 시도
                for code in self.symbol_list:
                    if code not in self.bought_list:
                        if self.try_buy(code):
                            available_slots -= 1
                            
                        time.sleep(1)  # API 호출 간격 유지
                        
                        # 매수 가능 종목 수 또는 현금 소진 시 중단
                        if available_slots <= 0 or self.total_cash < self.buy_amount:
                            break
            
            # 위험 관리 업데이트
            self.risk_manager.update_daily_pl()
            
            logger.info("매매 사이클 완료")
            return True
            
        except Exception as e:
            logger.error(f"매매 사이클 중 오류 발생: {e}")
            send_message(f"[오류] 매매 사이클 중 오류 발생: {e}")
            return False
    
    def run_trading(self, max_cycles: int = 0):
        """
        매매 실행
        
        Args:
            max_cycles: 최대 매매 사이클 수 (0: 무한)
        """
        try:
            logger.info(f"매매 시작 (전략: {self.strategy.get_strategy_name()})")
            send_message(f"[매매 시작] 전략: {self.strategy.get_strategy_name()}")
            
            cycle_count = 0
            
            while True:
                # 최대 사이클 수 확인
                if max_cycles > 0 and cycle_count >= max_cycles:
                    logger.info(f"최대 사이클 수({max_cycles}) 도달로 매매 종료")
                    break
                    
                # 단일 매매 사이클 실행
                success = self.run_single_trading_cycle()
                cycle_count += 1
                
                # 매매 사이클 결과 출력
                logger.info(f"매매 사이클 {cycle_count} 완료: {'성공' if success else '실패'}")
                send_message(f"[매매 사이클 {cycle_count}] {'성공' if success else '실패'}")
                
                # 위험 수준 확인
                if self.risk_manager.should_stop_trading():
                    logger.warning("위험 수준으로 인해 매매 종료")
                    send_message("[매매 종료] 위험 수준 도달로 인한 종료")
                    break
                    
                # 대기
                wait_time = TRADE_CONFIG.get("cycle_interval", 300)  # 기본 5분
                logger.info(f"{wait_time}초 대기 후 다음 사이클 시작")
                time.sleep(wait_time)
                
            # 매매 종료 처리
            self._finalize_trading()
            
        except Exception as e:
            logger.error(f"매매 실행 중 오류 발생: {e}")
            send_message(f"[오류] 매매 실행 중 오류 발생: {e}")
            self._finalize_trading()
    
    def _finalize_trading(self):
        """매매 종료 처리"""
        # 최종 자산 계산
        final_assets = self.risk_manager.calculate_total_assets()
        initial_assets = self.risk_manager.initial_total_assets
        
        # 수익률 계산
        profit = final_assets - initial_assets
        profit_rate = (profit / initial_assets) * 100 if initial_assets > 0 else 0
        
        # 결과 출력
        logger.info(f"매매 종료: 최종 자산 {final_assets:,}원 (수익: {profit:,}원, {profit_rate:.2f}%)")
        send_message(f"[매매 종료] 최종 자산: {final_assets:,}원 (수익: {profit:,}원, {profit_rate:.2f}%)")
        
        # 보유 종목 정보 출력
        if self.bought_list:
            stock_info = []
            for code in self.bought_list:
                try:
                    stock_data = self.stock_dict.get(code, {})
                    if not isinstance(stock_data, dict):
                        logger.warning(f"{code} 종목 정보가 딕셔너리가 아닙니다: {type(stock_data)}")
                        continue
                    
                    name = stock_data.get("prdt_name", "N/A")
                    qty = stock_data.get("units", 0)
                    current_price = stock_data.get("current_price", 0)
                    entry_price = self.entry_prices.get(code, 0)
                    profit_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                    
                    stock_info.append(f"{name}({code}): {qty}주, {profit_pct:.2f}%")
                except Exception as e:
                    logger.error(f"{code} 종목 정보 처리 중 오류 발생: {e}")
                
            send_message(f"[보유 종목] {', '.join(stock_info)}")
        else:
            send_message("[보유 종목] 없음") 