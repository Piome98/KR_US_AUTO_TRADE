"""
한국 주식 자동매매 - 메인 트레이더 모듈
자동 매매 시스템의 핵심 클래스 정의
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from korea_stock_auto.config import AppConfig
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.trading.stock_selector import StockSelector
from korea_stock_auto.trading.technical_analyzer import TechnicalAnalyzer
from korea_stock_auto.trading.trade_executor import TradeExecutor
from korea_stock_auto.trading.risk_manager import RiskManager
from korea_stock_auto.trading.trading_strategy import TradingStrategy


logger = logging.getLogger("stock_auto")

class Trader:
    """국내 주식 자동 매매 트레이더 클래스 (의존성 주입 적용)"""
    
    def __init__(self, 
                 api: KoreaInvestmentApiClient,
                 config: AppConfig,
                 analyzer: TechnicalAnalyzer,
                 risk_manager: RiskManager,
                 executor: TradeExecutor,
                 selector: StockSelector,
                 strategy: TradingStrategy):
        """
        트레이더 초기화 (의존성 주입)
        
        Args:
            api: 한국투자증권 API 클라이언트
            config: 애플리케이션 설정
            analyzer: 기술적 분석기
            risk_manager: 리스크 관리자
            executor: 거래 실행기
            selector: 종목 선택기
            strategy: 매매 전략
        """
        self.api = api
        self.config = config
        self.analyzer = analyzer
        self.risk_manager = risk_manager
        self.executor = executor
        self.selector = selector
        self.strategy = strategy
        
        # 트레이딩 상태 변수들
        self.bought_list: List[str] = []
        self.symbol_list: List[str] = []
        self.total_cash: float = 0
        self.stock_dict: Dict[str, Dict[str, Any]] = {}
        self.entry_prices: Dict[str, float] = {}
        self.buy_amount: float = 0
        self.buy_percentage: float = config.trading.buy_percentage
        self.target_buy_count: int = config.trading.target_buy_count
        self.soldout: bool = False
        
        # 시장 데이터 관리
        self.current_prices: Dict[str, Dict[str, Any]] = {}
        self.last_market_data_fetch_time: float = 0.0
        self.market_data_fetch_interval: int = config.system.data_update_interval
        self.last_strategy_run_time: float = 0.0
        self.strategy_run_interval: int = config.system.strategy_run_interval
        
        # 계좌 업데이트 설정
        self.account_update_interval = config.system.account_update_interval
        self.last_account_update_time = 0
        
        self.initialize_trading()
    

    
    def initialize_trading(self):
        """
        트레이딩 초기화
        - 액세스 토큰 발급
        - 계좌 정보 조회
        - 보유 종목 정보 업데이트
        """
        self._update_account_info(is_init=True)
        
        self.buy_amount = self.total_cash * self.buy_percentage
        
        self.risk_manager.reset_daily_stats()
        
        logger.info(f"트레이딩 초기화 완료: 현금 {self.total_cash:,}원, 보유 종목 {len(self.bought_list)}개")
        send_message(f"[초기화 완료] 현금: {self.total_cash:,}원, 보유 종목: {len(self.bought_list)}개, 초기 자산: {self.risk_manager.initial_total_assets:,}원", self.config.notification.discord_webhook_url)
        
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
        current_time = time.time()
        
        if current_time - self.last_account_update_time < self.account_update_interval:
            return
            
        if is_init:
            logger.info("계좌 잔고 및 보유 종목 정보 업데이트")
        balance_info = self.api.get_balance()
        stock_balance = self.api.get_stock_balance()
        
        if balance_info:
            self.total_cash = balance_info.get("cash", 0)
            if is_init:
                logger.info(f"계좌 잔고 조회 성공: 현금={balance_info['cash']:,.0f}원, "
                           f"주식={balance_info['stocks_value']:,.0f}원, "
                           f"총평가={balance_info['total_assets']:,.0f}원, "
                           f"수익률={balance_info['total_profit_loss_rate']:.2f}%")
        else:
            logger.warning("계좌 잔고 정보 조회 실패")
            self.total_cash = 0
            
        if stock_balance:
            if "stocks" in stock_balance and isinstance(stock_balance["stocks"], list):
                self.stock_dict = {}
                
                for stock in stock_balance["stocks"]:
                    code = stock.get("code")
                    if code:
                        self.stock_dict[code] = stock
                
                self.bought_list = list(self.stock_dict.keys())
                
                if is_init:
                    logger.info(f"보유 종목 {len(self.bought_list)}개 정보 업데이트 완료")
                    if balance_info and 'total_assets' in balance_info:
                        logger.info(f"주식 보유 종목 조회 성공: {len(self.bought_list)}종목, 총평가액: {balance_info['total_assets']:,.0f}원")
                    else:
                        logger.info(f"주식 보유 종목 조회 성공: {len(self.bought_list)}종목, 총평가액: 조회 실패")
            else:
                logger.warning("보유 종목 정보 형식이 예상과 다릅니다.")
                logger.debug(f"stock_balance: {stock_balance}")
        else:
            logger.warning("보유 종목 정보 조회 실패")
        
        self.last_account_update_time = current_time
    
    def _fetch_and_process_market_data(self):
        """5분 간격으로 시장 데이터를 가져오고 처리합니다."""
        current_time = time.time()
        if current_time - self.last_market_data_fetch_time < self.market_data_fetch_interval:
            return

        logger.info("주기적 시장 데이터 업데이트 시작 (5분 간격)...")
        self.last_market_data_fetch_time = current_time

        target_symbols_to_fetch = list(set(self.symbol_list + self.bought_list))
        if not target_symbols_to_fetch:
            logger.info("데이터를 가져올 관심/보유 종목이 없습니다.")
            return

        for code in target_symbols_to_fetch:
            try:
                # 현재가 정보 업데이트
                price_data = self.api.get_current_price(code)
                if price_data and isinstance(price_data, dict) and 'current_price' in price_data:
                    current_price = int(price_data['current_price'])
                    self.current_prices[code] = {
                        'price': current_price,
                        'bid': int(price_data.get('lowest_bid', 0)),
                        'ask': int(price_data.get('highest_ask', 0)),
                        'volume': int(price_data.get('volume', 0)),
                        'timestamp': current_time
                    }
                    logger.debug(f"{code} 현재가 업데이트: {current_price}")
                else:
                    logger.warning(f"{code} 현재가 정보 조회 실패 또는 형식 오류: {price_data}")
                    if code in self.current_prices:
                        pass 
                    else:
                        self.current_prices[code] = {'price': 0, 'timestamp': current_time}

                # 캐시된 데이터 확인 - 충분한 데이터가 있으면 추가 크롤링 하지 않음
                cached_data = self.analyzer._get_cached_ohlcv(code, 'D')
                if cached_data is not None and len(cached_data) >= 200:
                    logger.debug(f"{code} 충분한 캐시 데이터 존재 ({len(cached_data)}개), 추가 크롤링 생략")
                    # 기술적 지표만 업데이트 (데이터 크롤링 없이)
                    self.analyzer._update_indicators_only(code, 'D')
                else:
                    # 캐시된 데이터가 부족한 경우에만 데이터 업데이트
                    logger.debug(f"{code} 캐시 데이터 부족 ({len(cached_data) if cached_data is not None else 0}개), 데이터 업데이트 수행")
                    self.analyzer.update_symbol_data(code, interval='D', limit=50)  # 최소한의 데이터만 추가 수집

            except Exception as e:
                logger.error(f"{code} 종목 데이터 처리 중 오류: {e}", exc_info=True)
                if code not in self.current_prices:
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
            
        else:
            logger.warning("관심 종목 선정 실패")
            
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

        buy_signal = self.strategy.should_buy(code, current_price)

        if buy_signal:
            order_cash = min(self.buy_amount, self.total_cash)
            if order_cash < TRADE_CONFIG.get("min_order_amount", 10000):
                logger.info(f"{code} 매수 주문 금액({order_cash:,.0f}원)이 최소 주문 금액 미만입니다.")
                return
            
            if not self.risk_manager.can_buy(code, order_cash, self.total_cash, len(self.bought_list)):
                logger.info(f"{code} 리스크 관리 조건에 따라 매수 보류.")
                return

            success, order_details = self.executor.place_order(code, "buy", current_price, order_cash=order_cash)
            if success:
                self.bought_list.append(code)
                self.entry_prices[code] = current_price
                logger.info(f"{code} 매수 주문 성공: {order_details}")
                send_message(f"[매수 성공] {code} 가격: {current_price:,} ({order_details.get('quantity',0)}주)")
            else:
                logger.error(f"{code} 매수 주문 실패: {order_details}")
                send_message(f"[매수 실패] {code} 가격: {current_price:,} ({order_details})")
        else:
            logger.info(f"{code} 매수 보류 신호 수신 (should_buy 결과 False).")

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

        if self.risk_manager.check_stop_loss(code, current_price, entry_price):
            logger.info(f"{code} 손절 조건 충족. 매도 시도. 현재가: {current_price}, 매수가: {entry_price}")
            signal = "sell"
        elif self.risk_manager.check_take_profit(code, current_price, entry_price):
            logger.info(f"{code} 익절 조건 충족. 매도 시도. 현재가: {current_price}, 매수가: {entry_price}")
            signal = "sell"
        else:
            sell_signal = self.strategy.should_sell(code, current_price, entry_price)
            if sell_signal:
                signal = "sell"
            else:
                signal = "hold"

        if signal == "sell":
            stock_info = self.stock_dict.get(code, {})
            quantity_to_sell = 0
            if isinstance(stock_info, dict):
                quantity_to_sell = stock_info.get("hldg_qty", 0)
            
            if quantity_to_sell <= 0:
                logger.warning(f"{code} 매도 가능 수량이 없습니다. ({stock_info})")
                return

            success, order_details = self.executor.place_order(code, "sell", current_price, quantity=quantity_to_sell)
            if success:
                if code in self.bought_list:
                    self.bought_list.remove(code)
                if code in self.entry_prices:
                    del self.entry_prices[code]
                logger.info(f"{code} 매도 주문 성공: {order_details}")
                send_message(f"[매도 성공] {code} 가격: {current_price:,} ({quantity_to_sell}주)")
            else:
                logger.error(f"{code} 매도 주문 실패: {order_details}")
                send_message(f"[매도 실패] {code} 가격: {current_price:,} ({order_details})")
        elif signal == "hold":
            logger.info(f"{code} 매도 보류 신호 수신 (should_sell 결과 False 또는 리스크 관리 조건 미충족).")

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
        
        for code in list(self.bought_list):
            stock_info = self.stock_dict.get(code, {})
            name = "N/A"
            quantity = 0
            
            if isinstance(stock_info, dict):
                name = stock_info.get("prdt_name", code)
                quantity = stock_info.get("hldg_qty", 0) 
            
            if quantity <= 0:
                logger.warning(f"{code}({name}) 보유 수량이 없음")
                continue
                
            current_price = 0
            current_price_data = self.current_prices.get(code)
            if current_price_data and isinstance(current_price_data, dict) and current_price_data.get('price', 0) > 0:
                current_price = current_price_data['price']
                logger.info(f"{code} ({name}) - current_prices 데이터 활용 - 현재가: {current_price}")
            else:
                logger.info(f"{code} ({name}) - current_prices에 유효한 가격 정보 없음. API로 현재가 조회 시도.")
                price_data_api = self.api.get_current_price(code)
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
            
            success, order_details = self.executor.place_order(code, "sell", current_price, quantity=quantity, order_type="시장가") 
            
            if success:
                logger.info(f"{code}({name}) 매도 주문 성공: {quantity}주")
                success_count += 1
                
                if code in self.bought_list:
                    self.bought_list.remove(code)
                if code in self.entry_prices:
                    del self.entry_prices[code]
                if code in self.stock_dict:
                    del self.stock_dict[code]
            else:
                logger.error(f"{code}({name}) 매도 실패: {order_details}")
                fail_list.append(f"{code}({name})")
        
        self._update_account_info()
        
        self.risk_manager.update_daily_pl()
        
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

        self._update_account_info()

        self._fetch_and_process_market_data()

        if current_time - self.last_strategy_run_time < self.strategy_run_interval:
            logger.info(f"최근 전략 실행 후 {self.strategy_run_interval}초가 지나지 않아 이번 사이클은 건너뜁니다.")
            return
        
        self.last_strategy_run_time = current_time

        if not self.symbol_list:
            logger.info("관심 종목이 없어 선정을 시도합니다.")
            self.select_interest_stocks()
            if self.symbol_list:
                 self.last_market_data_fetch_time = 0
                 self._fetch_and_process_market_data()

        logger.info(f"매수 대상 종목 탐색 시작... 관심 종목 수: {len(self.symbol_list)}")
        if len(self.bought_list) < self.target_buy_count and self.total_cash >= TRADE_CONFIG.get("min_order_amount", 10000):
            for code in self.symbol_list:
                if code not in self.bought_list:
                    logger.debug(f"{code} 매수 조건 확인 중...")
                    self.try_buy(code)
                    if len(self.bought_list) >= self.target_buy_count:
                        logger.info("목표 보유 종목 수에 도달하여 추가 매수 중단.")
                        break
        else:
            if len(self.bought_list) >= self.target_buy_count:
                logger.info(f"이미 목표 보유 종목 수({self.target_buy_count}개)를 보유 중입니다.")
            if self.total_cash < TRADE_CONFIG.get("min_order_amount", 10000):
                logger.info(f"매수 가능 금액({self.total_cash:,.0f}원)이 최소 주문 금액 미만입니다.")

        logger.info(f"매도 대상 종목 탐색 시작... 보유 종목 수: {len(self.bought_list)}")
        if not self.bought_list:
            logger.info("보유 종목이 없어 매도를 진행하지 않습니다.")
        else:
            for code in list(self.bought_list):
                logger.debug(f"{code} 매도 조건 확인 중...")
                self.try_sell(code)
        
        self.log_portfolio_status()
        logger.info("단일 매매 사이클 종료.")

    def run_trading(self, max_cycles: int = 0):
        """
        자동 매매 실행
        
        Args:
            max_cycles: 최대 거래 사이클 수 (0이면 무제한)
        """
        try:
            logger.info("1. 트레이딩 초기화 시작 (액세스 토큰 발급 및 계좌 정보 확인)")
            self.initialize_trading()
            
            self.select_interest_stocks()
            
            logger.info("API 호출 기반으로 거래를 진행합니다.")
            send_message("[정보] API 호출 기반으로 거래를 진행합니다.")
            
            logger.info("4. 매매 사이클 시작")
            cycle_count = 0
            self.soldout = False
            
            while True:
                cycle_count += 1
                
                if max_cycles > 0 and cycle_count > max_cycles:
                    logger.info(f"최대 거래 사이클 수({max_cycles}) 도달")
                    break
                    
                logger.info(f"===== 거래 사이클 {cycle_count} 시작 =====")
                
                self.run_single_trading_cycle()
                
                logger.info(f"===== 거래 사이클 {cycle_count} 종료 =====")
                logger.info(f"보유 종목: {len(self.bought_list)}개, 관심 종목: {len(self.symbol_list)}개, 잔고: {self.total_cash:,.0f}원")
                
                logger.info("다음 거래 사이클까지 대기...")
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("사용자에 의한 거래 중단")
        except Exception as e:
            logger.error(f"거래 실행 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            send_message(f"[오류] 거래 실행 중 오류 발생: {e}")
        finally:
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
                avg_buy_price = self.entry_prices.get(code, stock_info.get("pchs_avg_pric", 0))
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