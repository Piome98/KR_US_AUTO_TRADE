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
        self.bought_list: List[str] = []  # 매수한 종목 리스트
        self.symbol_list: List[str] = []  # 관심 종목 리스트
        self.total_cash: float = 0    # 총 현금
        self.stock_dict: Dict[str, Dict[str, Any]] = {}   # 보유 종목 정보
        self.entry_prices: Dict[str, float] = {} # 종목별 매수 가격
        self.buy_amount: float = 0    # 종목별 매수 금액
        self.buy_percentage: float = TRADE_CONFIG.get("buy_percentage", 0.2)  # 매수 비율 (기본: 현금의 20%)
        self.target_buy_count: int = TRADE_CONFIG.get("target_buy_count", 5)  # 목표 매수 종목 수
        self.soldout: bool = False   # 전량 매도 여부
        
        # API 호출 기반 데이터 관리
        self.current_prices: Dict[str, Dict[str, Any]] = {} # 종목 코드 키, {'price': 현재가, 'volume': 거래량, 'timestamp': ...}
        self.last_market_data_fetch_time: float = 0.0
        self.market_data_fetch_interval: int = 300  # 5분 (300초)
        self.last_strategy_run_time: float = 0.0 # 마지막 전략 실행 시간 (중복 실행 방지용)
        self.strategy_run_interval: int = 60 # 전략 실행 최소 간격 (초)
        
        # 먼저 필요한 컴포넌트들 생성
        # 기술적 분석기 생성
        self.analyzer = TechnicalAnalyzer(self.api)
        
        # 위험 관리자 생성
        self.risk_manager = RiskManager(self.api)
        
        # 매매 실행기 생성
        self.executor = TradeExecutor(self.api)
        
        # 종목 선택기 생성
        self.selector = StockSelector(self.api)
        
        
        # 호가 데이터 보관용 딕셔너리
        # self.quote_data = {}
        
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
        
        # 웹소켓 관련 코드 제거 - 관심종목 선정 후 별도 함수에서 처리
        
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
            self.total_cash = 0  # 기본값 설정
            
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
                    if balance_info and 'total_assets' in balance_info:
                        logger.info(f"주식 보유 종목 조회 성공: {len(self.bought_list)}종목, 총평가액: {balance_info['total_assets']:,.0f}원")
                    else:
                        logger.info(f"주식 보유 종목 조회 성공: {len(self.bought_list)}종목, 총평가액: 조회 실패")
            else:
                logger.warning("보유 종목 정보 형식이 예상과 다릅니다.")
                # 응답 형식 로깅
                logger.debug(f"stock_balance: {stock_balance}")
        else:
            logger.warning("보유 종목 정보 조회 실패")
        
        self.last_account_update_time = current_time
    
    def _fetch_and_process_market_data(self):
        """5분 간격으로 시장 데이터를 가져오고 처리합니다."""
        current_time = time.time()
        if current_time - self.last_market_data_fetch_time < self.market_data_fetch_interval:
            return # 아직 5분이 지나지 않았으면 반환

        logger.info("주기적 시장 데이터 업데이트 시작 (5분 간격)...")
        self.last_market_data_fetch_time = current_time

        target_symbols_to_fetch = list(set(self.symbol_list + self.bought_list))
        if not target_symbols_to_fetch:
            logger.info("데이터를 가져올 관심/보유 종목이 없습니다.")
            return

        for code in target_symbols_to_fetch:
            try:
                # 1. 현재가 조회 (API 호출)
                price_data = self.api.get_current_price(code) # API에 해당 메소드 필요
                if price_data and isinstance(price_data, dict) and 'current_price' in price_data:
                    current_price = int(price_data['current_price'])
                    self.current_prices[code] = {
                        'price': current_price,
                        'bid': int(price_data.get('lowest_bid', 0)), # 매수호가
                        'ask': int(price_data.get('highest_ask', 0)), # 매도호가
                        'volume': int(price_data.get('volume', 0)), # 누적거래량
                        'timestamp': current_time
                    }
                    logger.debug(f"{code} 현재가 업데이트: {current_price}")
                else:
                    logger.warning(f"{code} 현재가 정보 조회 실패 또는 형식 오류: {price_data}")
                    if code in self.current_prices: # 이전 데이터가 있다면 유지, 없다면 초기화
                        pass 
                    else:
                        self.current_prices[code] = {'price': 0, 'timestamp': current_time} # 오류 시 가격 0으로 초기화

                # 2. 5분봉 데이터 조회 및 기술적 지표 업데이트 (TechnicalAnalyzer 사용)
                # TechnicalAnalyzer가 내부적으로 API 호출하고 지표 계산/저장하도록 구현되어 있다고 가정
                self.analyzer.update_symbol_data(code, interval='5') # 5분봉 데이터 업데이트
                self.analyzer.update_symbol_data(code, interval='D') # 일봉 데이터도 주기적으로 업데이트
                
                # 필요한 경우, 업데이트된 지표를 로깅하거나 self.strategy에 직접 전달할 준비
                # 예: rsi_5min = self.analyzer.get_rsi(code, interval='5')
                #     macd_daily, macdsignal_daily, _ = self.analyzer.get_macd(code, interval='D') 
                #     logger.debug(f"{code} - 5분봉 RSI: {rsi_5min}, 일봉 MACD: {macd_daily}, Signal: {macdsignal_daily}")

            except Exception as e:
                logger.error(f"{code} 종목 데이터 처리 중 오류: {e}", exc_info=True)
                if code not in self.current_prices: # 오류 발생 시 해당 종목 데이터 초기화
                     self.current_prices[code] = {'price': 0, 'timestamp': current_time}
        
        logger.info("주기적 시장 데이터 업데이트 완료.")

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
        """지정된 종목 매수 시도"""
        if code in self.bought_list:
            logger.info(f"{code} 종목은 이미 보유 중입니다. 추가 매수하지 않습니다.")
            return

        if len(self.bought_list) >= self.target_buy_count:
            logger.info(f"목표 보유 종목 수({self.target_buy_count}개)에 도달하여 신규 매수하지 않습니다.")
            return
        
        current_price_info = self.current_prices.get(code)
        if not current_price_info or not current_price_info.get('price'):
            logger.warning(f"{code} 종목의 현재가 정보가 없어 매수 시도 불가.")
            return
        
        current_price = current_price_info['price']

        # 매수 조건 확인 (TradingStrategy 사용)
        # generate_signal은 (종목코드, 현재가격정보, 기술적분석기) 등을 받아 매수/매도/보류 신호 반환 가정
        buy_signal = self.strategy.should_buy(code, current_price) # current_price_info 대신 current_price 전달

        if buy_signal:
            # 매수 금액 결정 (가용 현금의 일정 비율 또는 고정 금액 등)
            # self.buy_amount는 __init__에서 self.total_cash * self.buy_percentage 로 설정됨
            order_cash = min(self.buy_amount, self.total_cash) # 실제 주문 가능 금액
            if order_cash < TRADE_CONFIG.get("min_order_amount", 10000): # 최소 주문 금액 확인
                logger.info(f"{code} 매수 주문 금액({order_cash:,.0f}원)이 최소 주문 금액 미만입니다.")
                return
            
            # 리스크 관리 (예: 최대 투자 비중 등) - TradeExecutor 내부 또는 여기서 확인 가능
            if not self.risk_manager.can_buy(code, order_cash, self.total_cash, len(self.bought_list)):
                logger.info(f"{code} 리스크 관리 조건에 따라 매수 보류.")
                return

            # 매수 실행 (TradeExecutor 사용)
            success, order_details = self.executor.place_order(code, "buy", current_price, order_cash=order_cash)
            if success:
                self.bought_list.append(code)
                self.entry_prices[code] = current_price # 단순화된 매수가 기록 (평균 매수가로 변경 필요)
                # self._update_account_info() # 매수 성공 시 즉시 계좌 정보 업데이트 (API 호출 부하 고려)
                logger.info(f"{code} 매수 주문 성공: {order_details}")
                send_message(f"[매수 성공] {code} 가격: {current_price:,} ({order_details.get('quantity',0)}주)")
            else:
                logger.error(f"{code} 매수 주문 실패: {order_details}")
                send_message(f"[매수 실패] {code} 가격: {current_price:,} ({order_details})")
        else: # not buy_signal
            logger.info(f"{code} 매수 보류 신호 수신 (should_buy 결과 False).")
        # else (sell 신호는 여기서 처리 안함)

    def try_sell(self, code: str):
        """지정된 종목 매도 시도"""
        if code not in self.bought_list:
            logger.warning(f"{code} 종목을 보유하고 있지 않아 매도할 수 없습니다.")
            return

        current_price_info = self.current_prices.get(code)
        if not current_price_info or not current_price_info.get('price'):
            logger.warning(f"{code} 종목의 현재가 정보가 없어 매도 시도 불가.")
            return
        
        current_price = current_price_info['price']
        entry_price = self.entry_prices.get(code, 0)

        # 매도 조건 확인 (TradingStrategy 사용 + RiskManager의 손절/익절 조건)
        # 1. 리스크 관리 조건 우선 확인 (손절/익절)
        if self.risk_manager.check_stop_loss(code, current_price, entry_price):
            logger.info(f"{code} 손절 조건 충족. 매도 시도. 현재가: {current_price}, 매수가: {entry_price}")
            signal = "sell" # 리스크 관리 매도 신호
        elif self.risk_manager.check_take_profit(code, current_price, entry_price):
            logger.info(f"{code} 익절 조건 충족. 매도 시도. 현재가: {current_price}, 매수가: {entry_price}")
            signal = "sell" # 리스크 관리 매도 신호
        else:
            # 2. 전략 기반 매도 신호 확인
            sell_signal = self.strategy.should_sell(code, current_price, entry_price)
            if sell_signal:
                signal = "sell"
            else:
                signal = "hold" # should_sell이 False이면 hold로 간주

        if signal == "sell":
            stock_info = self.stock_dict.get(code, {})
            quantity_to_sell = 0
            if isinstance(stock_info, dict):
                quantity_to_sell = stock_info.get("hldg_qty", 0) # 보유수량 (API 응답 필드명 확인 필요)
            
            if quantity_to_sell <= 0:
                logger.warning(f"{code} 매도 가능 수량이 없습니다. ({stock_info})")
                return

            # 매도 실행 (TradeExecutor 사용)
            success, order_details = self.executor.place_order(code, "sell", current_price, quantity=quantity_to_sell)
            if success:
                if code in self.bought_list: # 정상적으로 매도 처리된 경우 bought_list에서 제거
                    self.bought_list.remove(code)
                if code in self.entry_prices:
                    del self.entry_prices[code]
                # self._update_account_info() # 매도 성공 시 즉시 계좌 정보 업데이트 (API 호출 부하 고려)
                logger.info(f"{code} 매도 주문 성공: {order_details}")
                send_message(f"[매도 성공] {code} 가격: {current_price:,} ({quantity_to_sell}주)")
            else:
                logger.error(f"{code} 매도 주문 실패: {order_details}")
                send_message(f"[매도 실패] {code} 가격: {current_price:,} ({order_details})")
        elif signal == "hold":
            logger.info(f"{code} 매도 보류 신호 수신 (should_sell 결과 False 또는 리스크 관리 조건 미충족).")
        # else (buy 신호는 여기서 처리 안함)

    def sell_all_stocks(self):
        """
        모든 보유 종목 매도
        
        Returns:
            bool: 모든 종목 매도 성공 여부
        """
        if not self.bought_list:
            logger.info("매도할 보유 종목이 없습니다.")
            return True
            
        logger.info(f"모든 보유 종목 매도 시작 ({len(self.bought_list)}개)")
        send_message(f"[전량 매도] {len(self.bought_list)}개 종목 매도 시작")
        
        success_count = 0
        fail_list = []
        
        for code in list(self.bought_list):  # 복사본으로 순회 (매도 시 원본 리스트가 변경되므로)
            stock_info = self.stock_dict.get(code, {})
            name = "N/A"
            quantity = 0
            
            if isinstance(stock_info, dict):
                name = stock_info.get("prdt_name", code)
                quantity = stock_info.get("hldg_qty", 0) 
            
            if quantity <= 0:
                logger.warning(f"{code}({name}) 보유 수량이 없음")
                continue
                
            # 현재가 정보 확인 (self.current_prices 우선 활용)
            current_price = 0
            current_price_data = self.current_prices.get(code)
            if current_price_data and isinstance(current_price_data, dict) and current_price_data.get('price', 0) > 0:
                current_price = current_price_data['price']
                logger.info(f"{code} ({name}) - current_prices 데이터 활용 - 현재가: {current_price}")
            else: # current_prices에 없거나 유효하지 않으면 API 직접 조회
                logger.info(f"{code} ({name}) - current_prices에 유효한 가격 정보 없음. API로 현재가 조회 시도.")
                price_data_api = self.api.get_current_price(code) # API에 해당 메소드 필요
                if price_data_api and isinstance(price_data_api, dict) and 'stck_prpr' in price_data_api:
                    try:
                        current_price = int(price_data_api['stck_prpr'])
                        if current_price <= 0:
                            logger.warning(f"{code} ({name}) API 조회 현재가가 0 이하: {current_price}. 매도 주문 불가.")
                            fail_list.append(f"{code}({name}) - 현재가 0 이하")
                            continue
                        logger.info(f"{code} ({name}) - API 직접 조회 활용 - 현재가: {current_price}")
                    except ValueError:
                        logger.warning(f"{code} ({name}) API 조회 현재가 변환 오류: {price_data_api['stck_prpr']}. 매도 주문 불가.")
                        fail_list.append(f"{code}({name}) - 현재가 변환오류")
                        continue
                else:
                    logger.warning(f"{code} ({name}) API 현재가 조회 실패. 매도 주문 불가. Response: {price_data_api}")
                    fail_list.append(f"{code}({name}) - 현재가 API 조회실패")
                    continue 
            
            # 시장가 매도 (TradeExecutor 사용)
            # self.executor.place_order는 (종목코드, 주문유형, 가격, 수량, 주문타입) 등을 인자로 받는다고 가정
            success, order_details = self.executor.place_order(code, "sell", current_price, quantity=quantity, order_type="시장가") 
            
            if success:
                logger.info(f"{code}({name}) 매도 주문 성공: {quantity}주")
                success_count += 1
                
                # 매도 성공한 종목 목록에서 제거
                if code in self.bought_list:
                    self.bought_list.remove(code)
                if code in self.entry_prices:
                    del self.entry_prices[code]
                if code in self.stock_dict:
                    del self.stock_dict[code]
            else:
                logger.error(f"{code}({name}) 매도 실패: {order_details}")
                fail_list.append(f"{code}({name})")
        
        # 계좌 정보 업데이트
        self._update_account_info()
        
        # 위험 관리 업데이트
        self.risk_manager.update_daily_pl()
        
        # 결과 로깅
        result_msg = f"전량 매도 결과: {success_count}개 성공"
        if fail_list:
            result_msg += f", {len(fail_list)}개 실패 ({', '.join(fail_list)})"
        
        logger.info(result_msg)
        send_message(result_msg)
        
        return len(fail_list) == 0
    
    def run_single_trading_cycle(self):
        """단일 매매 사이클 실행"""
        current_time = time.time()
        logger.info("단일 매매 사이클 시작...")
        send_message("단일 매매 사이클 시작")

        # 1. 계좌 정보 업데이트 (주기적으로)
        self._update_account_info()

        # 2. 시장 데이터 업데이트 (5분 간격)
        self._fetch_and_process_market_data()

        # 3. 매매 전략 실행 (너무 자주 실행되지 않도록 제어)
        if current_time - self.last_strategy_run_time < self.strategy_run_interval:
            logger.info(f"최근 전략 실행 후 {self.strategy_run_interval}초가 지나지 않아 이번 사이클은 건너뜁니다.")
            return
        
        self.last_strategy_run_time = current_time

        # 3.1. 관심 종목 선정 (필요시, 예: 하루 한 번 또는 특정 조건 만족 시)
        # 현재는 매 사이클마다 실행하지 않고, 필요시 외부에서 호출하거나 초기화 시 실행
        if not self.symbol_list: # 관심 종목이 없으면 선정 시도
            logger.info("관심 종목이 없어 선정을 시도합니다.")
            self.select_interest_stocks()
            # 관심종목 선정 후에는 바로 데이터를 가져오도록 호출 (다음 5분 주기까지 기다리지 않음)
            if self.symbol_list:
                 self.last_market_data_fetch_time = 0 # 강제 업데이트 유도
                 self._fetch_and_process_market_data()

        # 3.2. 매수 로직: 관심 종목 중 매수 조건 충족 시 매수 시도
        logger.info(f"매수 대상 종목 탐색 시작... 관심 종목 수: {len(self.symbol_list)}")
        if len(self.bought_list) < self.target_buy_count and self.total_cash >= TRADE_CONFIG.get("min_order_amount", 10000):
            for code in self.symbol_list:
                if code not in self.bought_list: # 이미 보유한 종목은 매수 시도 안 함
                    logger.debug(f"{code} 매수 조건 확인 중...")
                    self.try_buy(code) # try_buy 내부에서 signal 확인 후 주문
                    if len(self.bought_list) >= self.target_buy_count:
                        logger.info("목표 보유 종목 수에 도달하여 추가 매수 중단.")
                        break # 목표 보유 수 도달 시 매수 중단
        else:
            if len(self.bought_list) >= self.target_buy_count:
                logger.info(f"이미 목표 보유 종목 수({self.target_buy_count}개)를 보유 중입니다.")
            if self.total_cash < TRADE_CONFIG.get("min_order_amount", 10000):
                logger.info(f"매수 가능 금액({self.total_cash:,.0f}원)이 최소 주문 금액 미만입니다.")

        # 3.3. 매도 로직: 보유 종목 중 매도 조건 충족 시 매도 시도
        logger.info(f"매도 대상 종목 탐색 시작... 보유 종목 수: {len(self.bought_list)}")
        if not self.bought_list:
            logger.info("보유 종목이 없어 매도를 진행하지 않습니다.")
        else:
            # bought_list를 복사하여 반복 (매도 성공 시 bought_list에서 제거되기 때문)
            for code in list(self.bought_list):
                logger.debug(f"{code} 매도 조건 확인 중...")
                self.try_sell(code) # try_sell 내부에서 signal 확인 후 주문
        
        # 4. 포트폴리오 현황 로깅
        self.log_portfolio_status()
        logger.info("단일 매매 사이클 종료.")

    def run_trading(self, max_cycles: int = 0):
        """
        자동 매매 실행
        
        Args:
            max_cycles: 최대 거래 사이클 수 (0이면 무제한)
        """
        try:
            # 1. 거래 초기화 (액세스 토큰 발급 및 계좌 정보 확인)
            logger.info("1. 트레이딩 초기화 시작 (액세스 토큰 발급 및 계좌 정보 확인)")
            self.initialize_trading()
            
            # 2. 관심 종목 선정
            send_message("관심 종목 선정 시작")
            self.select_interest_stocks()
            
            # 3. 웹소켓 관련 로직 전체 제거
            logger.info("API 호출 기반으로 거래를 진행합니다.")
            send_message("[정보] API 호출 기반으로 거래를 진행합니다.")
            
            # 4. 매매 사이클 시작
            logger.info("4. 매매 사이클 시작")
            cycle_count = 0
            self.soldout = False
            
            while True:
                cycle_count += 1
                
                if max_cycles > 0 and cycle_count > max_cycles:
                    logger.info(f"최대 거래 사이클 수({max_cycles}) 도달")
                    break
                    
                logger.info(f"===== 거래 사이클 {cycle_count} 시작 =====")
                
                # 단일 거래 사이클 실행
                self.run_single_trading_cycle()
                
                logger.info(f"===== 거래 사이클 {cycle_count} 종료 =====")
                logger.info(f"보유 종목: {len(self.bought_list)}개, 관심 종목: {len(self.symbol_list)}개, 잔고: {self.total_cash:,.0f}원")
                
                # 다음 사이클까지 대기
                logger.info("다음 거래 사이클까지 대기...")
                time.sleep(60)  # 60초 대기
                
        except KeyboardInterrupt:
            logger.info("사용자에 의한 거래 중단")
        except Exception as e:
            logger.error(f"거래 실행 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            send_message(f"[오류] 거래 실행 중 오류 발생: {e}")
        finally:
            # 정리 작업
            self._finalize_trading()
    
    def log_portfolio_status(self):
        """현재 포트폴리오 상태를 로깅합니다."""
        logger.info("===== 포트폴리오 현황 =====")
        logger.info(f"총 현금: {self.total_cash:,.0f}원")
        logger.info(f"보유 종목 수: {len(self.bought_list)} / {self.target_buy_count}")
        total_stock_value = 0
        if not self.bought_list:
            logger.info("보유 종목 없음")
        else:
            for code in self.bought_list:
                stock_info = self.stock_dict.get(code, {})
                current_price = self.current_prices.get(code, {}).get('price', 0)
                quantity = stock_info.get("hldg_qty", 0)
                avg_buy_price = self.entry_prices.get(code, stock_info.get("pchs_avg_pric", 0)) # API 필드명 확인 필요
                eval_value = current_price * quantity
                profit_loss = (current_price - avg_buy_price) * quantity if avg_buy_price > 0 else 0
                profit_loss_rate = (profit_loss / (avg_buy_price * quantity)) * 100 if avg_buy_price > 0 and quantity > 0 else 0
                total_stock_value += eval_value
                logger.info(f"  - {stock_info.get('prdt_name', code)} ({code}): {quantity}주, 현재가 {current_price:,}, 평가액 {eval_value:,.0f}, 수익률 {profit_loss_rate:.2f}% (매수가: {avg_buy_price:,})")
        
        total_assets = self.total_cash + total_stock_value
        logger.info(f"총 주식 평가액: {total_stock_value:,.0f}원")
        logger.info(f"총 자산 평가액: {total_assets:,.0f}원")
        initial_assets = self.risk_manager.initial_total_assets
        if initial_assets > 0:
            overall_profit_loss_rate = ((total_assets - initial_assets) / initial_assets) * 100
            logger.info(f"전체 수익률 (초기 자산 대비): {overall_profit_loss_rate:.2f}%")
        logger.info("===========================")

    def _finalize_trading(self):
        """거래 종료 정리"""
        
        logger.info("자동 매매 정리 작업 완료")
        send_message("[자동 매매] 종료") 