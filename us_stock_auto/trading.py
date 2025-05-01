"""
미국 주식 자동매매 - 매매 로직 모듈
자동 매매 관련 클래스 및 함수 정의
"""

import datetime
from pytz import timezone

from us_stock_auto.config import (
    EXCHANGE_CODE_MAP, TRADE_CONFIG, TIMEZONE
)
from us_stock_auto.utils import send_message, get_market_time, wait
from us_stock_auto.api_client import APIClient

class StockTrader:
    """미국 주식 자동 매매 클래스"""
    
    def __init__(self):
        """트레이더 초기화"""
        self.api = APIClient()
        self.initialize_trading()
    
    def initialize_trading(self):
        """매매 초기화"""
        # 매수 희망 종목 리스트
        self.nasd_symbol_list = ["AAPL"]  # NASDAQ
        self.nyse_symbol_list = ["KO"]    # NYSE
        self.amex_symbol_list = ["LIT"]   # AMEX
        self.symbol_list = self.nasd_symbol_list + self.nyse_symbol_list + self.amex_symbol_list
        
        # 매매 관련 변수 초기화
        self.bought_list = []             # 매수 완료된 종목 리스트
        self.total_cash = self.api.get_balance()   # 보유 현금 조회
        self.exchange_rate = self.api.get_exchange_rate()  # 환율 조회
        
        # 현재 보유 주식 조회
        self.stock_dict = self.api.get_stock_balance()
        for sym in self.stock_dict.keys():
            self.bought_list.append(sym)
        
        # 매매 설정
        self.target_buy_count = TRADE_CONFIG["target_buy_count"]  # 매수할 종목 수
        self.buy_percent = TRADE_CONFIG["buy_percent"]           # 종목당 매수 금액 비율
        self.buy_amount = self.total_cash * self.buy_percent / self.exchange_rate  # 종목별 매수 금액(달러)
        self.soldout = False
    
    def get_market_code(self, symbol, for_price=False):
        """
        종목 코드에 따른 시장 코드 반환
        
        Args:
            symbol (str): 종목 코드
            for_price (bool): 시세 조회용 코드 여부
            
        Returns:
            str: 시장 코드
        """
        if symbol in self.nasd_symbol_list:
            return EXCHANGE_CODE_MAP["NASDAQ_PRICE"] if for_price else EXCHANGE_CODE_MAP["NASDAQ"]
        elif symbol in self.nyse_symbol_list:
            return EXCHANGE_CODE_MAP["NYSE_PRICE"] if for_price else EXCHANGE_CODE_MAP["NYSE"]
        elif symbol in self.amex_symbol_list:
            return EXCHANGE_CODE_MAP["AMEX_PRICE"] if for_price else EXCHANGE_CODE_MAP["AMEX"]
        else:
            return EXCHANGE_CODE_MAP["NASDAQ_PRICE"] if for_price else EXCHANGE_CODE_MAP["NASDAQ"]
    
    def sell_all_stocks(self):
        """보유 종목 일괄 매도"""
        self.stock_dict = self.api.get_stock_balance()
        
        for sym, qty in self.stock_dict.items():
            market_api = self.get_market_code(sym)
            market_price = self.get_market_code(sym, for_price=True)
            current_price = self.api.get_current_price(market=market_price, code=sym)
            
            if current_price and self.api.sell(market=market_api, code=sym, qty=qty, price=current_price):
                send_message(f"{sym} {qty}주 매도 완료")
        
        self.soldout = True
        self.bought_list = []
        wait(1)
        # 잔고 갱신
        self.stock_dict = self.api.get_stock_balance()
    
    def process_market_open_sell(self):
        """장 시작 직후 잔여 수량 매도"""
        if self.soldout:
            return
            
        self.sell_all_stocks()
    
    def try_buy(self, symbol):
        """
        매수 시도
        
        Args:
            symbol (str): 종목 코드
            
        Returns:
            bool: 매수 성공 여부
        """
        # 매수 종목 수 제한 확인
        if len(self.bought_list) >= self.target_buy_count:
            return False
                
        # 이미 매수한 종목 스킵
        if symbol in self.bought_list:
            return False
        
        # 시장 코드 설정
        market_api = self.get_market_code(symbol)
        market_price = self.get_market_code(symbol, for_price=True)
        
        # 목표가 확인 및 매수
        target_price = self.api.get_target_price(market=market_price, code=symbol)
        current_price = self.api.get_current_price(market=market_price, code=symbol)
        
        if not target_price or not current_price:
            return False
        
        if target_price < current_price:
            # 매수 수량 계산
            buy_qty = int(self.buy_amount // current_price)
            
            if buy_qty > 0:
                send_message(f"{symbol} 목표가 달성({target_price} < {current_price}) 매수를 시도합니다.")
                result = self.api.buy(market=market_api, code=symbol, qty=buy_qty, price=current_price)
                
                if result:
                    self.soldout = False
                    self.bought_list.append(symbol)
                    self.api.get_stock_balance()
                    return True
        
        return False
    
    def process_market_trading(self):
        """장중 매매 처리"""
        # 각 종목에 대해 매수 조건 확인 및 매수
        for symbol in self.symbol_list:
            if self.try_buy(symbol):
                send_message(f"{symbol} 매수 완료")
            wait(1)
        
        # 현재 시간 확인
        now = datetime.datetime.now(timezone(TIMEZONE))
        
        # 30분마다 잔고 조회
        if now.minute == 30 and now.second <= 5: 
            self.api.get_stock_balance()
            wait(5)
    
    def process_market_close_sell(self):
        """장 마감 전 모든 종목 일괄 매도"""
        if not self.soldout:
            self.sell_all_stocks()
    
    def run_trading_cycle(self):
        """현재 시간에 따른 매매 사이클 실행"""
        # 현재 시간 (뉴욕 기준)
        now = datetime.datetime.now(timezone(TIMEZONE))
        t_9 = get_market_time('open')        # 장 시작 시간 (9:30)
        t_start = get_market_time('buy_start')  # 매수 시작 시간 (9:35)
        t_sell = get_market_time('sell_start')  # 매도 시작 시간 (15:45)
        t_exit = get_market_time('exit')     # 프로그램 종료 시간 (15:50)
        today = now.weekday()
        
        # 주말이면 종료
        if today == 5 or today == 6:
            return False
        
        # 장 시작 직후 잔여 수량 매도
        if t_9 < now < t_start:
            self.process_market_open_sell()
        
        # 장중 매수 시간
        elif t_start < now < t_sell:
            self.process_market_trading()
        
        # 장 마감 전 모든 종목 일괄 매도
        elif t_sell < now < t_exit:
            self.process_market_close_sell()
        
        # 프로그램 종료 시간
        elif t_exit < now:
            return False
            
        return True 