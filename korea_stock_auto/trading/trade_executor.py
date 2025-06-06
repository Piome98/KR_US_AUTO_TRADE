"""
한국 주식 자동매매 - 매매 실행 모듈
주식 매수/매도 실행 관련 클래스 및 함수 정의
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from korea_stock_auto.config import AppConfig
from korea_stock_auto.utils.utils import send_message, wait
from korea_stock_auto.api import KoreaInvestmentApiClient

logger = logging.getLogger("stock_auto")

class TradeExecutor:
    """주식 매매 실행 클래스"""
    
    def __init__(self, api_client: KoreaInvestmentApiClient, config: AppConfig):
        """
        매매 실행기 초기화
        
        Args:
            api_client: API 클라이언트 인스턴스
            config: 애플리케이션 설정
        """
        self.api = api_client
        self.config = config
        self.max_buy_attempts = config.trading.max_buy_attempts
        self.max_sell_attempts = config.trading.max_sell_attempts
        self.order_wait_time = config.trading.order_wait_time
        self.slippage_tolerance = config.trading.slippage_tolerance
    
    def execute_buy(self, code: str, price: float, quantity: int) -> Dict[str, Any]:
        """
        매수 주문 실행
        
        Args:
            code: 종목 코드
            price: 주문 가격
            quantity: 주문 수량
            
        Returns:
            dict: 주문 결과 정보
        """
        if not code or price <= 0 or quantity <= 0:
            logger.error(f"매수 주문 파라미터 오류: code={code}, price={price}, quantity={quantity}")
            return {"success": False, "message": "매수 주문 파라미터 오류"}
            
        logger.info(f"{code} 매수 주문 시작: {quantity}주 @ {price}원")
        send_message(f"[매수 시도] {code}: {quantity}주 @ {price}원", self.config.notification.discord_webhook_url)
        
        # 시장가 주문 여부 확인 (0원이면 시장가)
        is_market_order = (price == 0)
        
        # 매수 주문 실행
        result = self._execute_order(code, quantity, price, "buy", is_market_order)
        
        if result["success"]:
            logger.info(f"{code} 매수 주문 성공: 주문번호 {result.get('order_no', 'N/A')}")
            send_message(f"[매수 완료] {code}: {quantity}주 @ {price}원", self.config.notification.discord_webhook_url)
        else:
            logger.error(f"{code} 매수 주문 실패: {result.get('message', '알 수 없는 오류')}")
            send_message(f"[매수 실패] {code}: {result.get('message', '알 수 없는 오류')}")
            
        return result
    
    def execute_sell(self, code: str, price: float, quantity: int) -> Dict[str, Any]:
        """
        매도 주문 실행
        
        Args:
            code: 종목 코드
            price: 주문 가격
            quantity: 주문 수량
            
        Returns:
            dict: 주문 결과 정보
        """
        if not code or price < 0 or quantity <= 0:
            logger.error(f"매도 주문 파라미터 오류: code={code}, price={price}, quantity={quantity}")
            return {"success": False, "message": "매도 주문 파라미터 오류"}
            
        logger.info(f"{code} 매도 주문 시작: {quantity}주 @ {price}원")
        send_message(f"[매도 시도] {code}: {quantity}주 @ {price}원", self.config.notification.discord_webhook_url)
        
        # 시장가 주문 여부 확인 (0원이면 시장가)
        is_market_order = (price == 0)
        
        # 매도 주문 실행
        result = self._execute_order(code, quantity, price, "sell", is_market_order)
        
        if result["success"]:
            logger.info(f"{code} 매도 주문 성공: 주문번호 {result.get('order_no', 'N/A')}")
            send_message(f"[매도 완료] {code}: {quantity}주 @ {price}원", self.config.notification.discord_webhook_url)
        else:
            logger.error(f"{code} 매도 주문 실패: {result.get('message', '알 수 없는 오류')}")
            send_message(f"[매도 실패] {code}: {result.get('message', '알 수 없는 오류')}")
            
        return result
    
    def _execute_order(self, code: str, quantity: int, price: float, order_type: str, is_market_order: bool) -> Dict[str, Any]:
        """
        주문 실행 내부 함수
        
        Args:
            code: 종목 코드
            quantity: 주문 수량
            price: 주문 가격
            order_type: 주문 유형 (buy/sell)
            is_market_order: 시장가 주문 여부
            
        Returns:
            dict: 주문 결과 정보
        """
        max_attempts = self.max_buy_attempts if order_type == "buy" else self.max_sell_attempts
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            try:
                # 주문 실행
                if order_type == "buy":
                    if is_market_order:
                        order_result = self.api.buy_market_order(code, quantity)
                    else:
                        order_result = self.api.buy_limit_order(code, price, quantity)
                else:  # sell
                    if is_market_order:
                        order_result = self.api.sell_market_order(code, quantity)
                    else:
                        order_result = self.api.sell_limit_order(code, price, quantity)
                
                # 주문 결과 확인
                if not order_result or not order_result.get("success", False):
                    error_msg = order_result.get("message", "알 수 없는 오류")
                    logger.warning(f"{code} {order_type} 주문 실패 (시도 {attempt}/{max_attempts}): {error_msg}")
                    
                    if attempt < max_attempts:
                        wait_time = self.order_wait_time * attempt
                        logger.info(f"{wait_time}초 후 재시도...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {"success": False, "message": f"최대 시도 횟수 초과: {error_msg}"}
                
                # 주문 성공
                order_no = order_result.get("order_no", "")
                
                # 주문 체결 확인
                if not is_market_order:  # 지정가 주문인 경우만 체결 확인
                    executed = self._check_order_execution(order_no, code, quantity, order_type)
                    if executed:
                        return {"success": True, "order_no": order_no, "executed": True}
                    else:
                        # 미체결 주문 취소 (다음 시도를 위해)
                        if attempt < max_attempts:
                            self._cancel_order(order_no, code, order_type)
                            wait_time = self.order_wait_time * attempt
                            logger.info(f"미체결 주문 취소 후 {wait_time}초 대기...")
                            time.sleep(wait_time)
                            continue
                        else:
                            return {"success": True, "order_no": order_no, "executed": False, 
                                    "message": "주문 접수되었으나 체결되지 않음"}
                else:
                    # 시장가 주문은 체결 확인 없이 성공으로 간주
                    return {"success": True, "order_no": order_no, "executed": True}
                    
            except Exception as e:
                logger.error(f"{code} {order_type} 주문 중 예외 발생: {e}")
                if attempt < max_attempts:
                    wait_time = self.order_wait_time * attempt
                    logger.info(f"{wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                    continue
                else:
                    return {"success": False, "message": f"주문 중 예외 발생: {e}"}
        
        return {"success": False, "message": "최대 시도 횟수 초과"}
    
    def _check_order_execution(self, order_no: str, code: str, quantity: int, order_type: str) -> bool:
        """
        주문 체결 확인
        
        Args:
            order_no: 주문번호
            code: 종목 코드
            quantity: 주문 수량
            order_type: 주문 유형 (buy/sell)
            
        Returns:
            bool: 체결 완료 여부
        """
        try:
            # 체결 확인을 위한 대기
            time.sleep(self.order_wait_time)
            
            # 주문 상태 조회
            order_status = self.api.get_order_status(order_no)
            
            if not order_status:
                logger.warning(f"{code} 주문 상태 조회 실패")
                return False
                
            # 체결 수량 확인
            executed_qty = order_status.get("executed_qty", 0)
            
            # 완전 체결 여부
            if executed_qty >= quantity:
                logger.info(f"{code} 주문 완전 체결: {executed_qty}주")
                return True
            else:
                logger.info(f"{code} 주문 부분 체결: {executed_qty}/{quantity}주")
                return False
                
        except Exception as e:
            logger.error(f"{code} 주문 체결 확인 중 예외 발생: {e}")
            return False
    
    def _cancel_order(self, order_no: str, code: str, order_type: str) -> bool:
        """
        미체결 주문 취소
        
        Args:
            order_no: 주문번호
            code: 종목 코드
            order_type: 주문 유형 (buy/sell)
            
        Returns:
            bool: 취소 성공 여부
        """
        try:
            logger.info(f"{code} 미체결 주문 취소 시도: {order_no}")
            
            # 주문 취소 요청
            result = self.api.cancel_order(order_no)
            
            if result and result.get("success", False):
                logger.info(f"{code} 주문 취소 성공")
                return True
            else:
                logger.warning(f"{code} 주문 취소 실패: {result.get('message', '알 수 없는 오류')}")
                return False
                
        except Exception as e:
            logger.error(f"{code} 주문 취소 중 예외 발생: {e}")
            return False
    
    def sell_all_stocks(self, stocks: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """
        모든 보유 주식 매도
        
        Args:
            stocks: 보유 주식 정보 딕셔너리 {종목코드: 주식정보}
            
        Returns:
            dict: 종목별 매도 성공 여부 {종목코드: 성공여부}
        """
        if not stocks:
            logger.info("매도할 주식이 없습니다.")
            return {}
            
        send_message(f"보유 주식 전체 매도 시작 ({len(stocks)}종목)", self.config.notification.discord_webhook_url)
        
        results = {}
        for code, stock_info in stocks.items():
            quantity = stock_info.get("qty", 0)
            if quantity <= 0:
                continue
                
            # 시장가 매도
            result = self.execute_sell(code, 0, quantity)
            results[code] = result.get("success", False)
            
            # 연속 API 호출 방지를 위한 대기
            time.sleep(self.order_wait_time)
            
        send_message(f"보유 주식 전체 매도 완료: 성공 {sum(results.values())}건, 실패 {len(results) - sum(results.values())}건", self.config.notification.discord_webhook_url)
        return results
    
    def calculate_buy_quantity(self, price: float, available_cash: float) -> int:
        """
        매수 가능 수량 계산
        
        Args:
            price: 현재가
            available_cash: 사용 가능 금액
            
        Returns:
            int: 매수 가능 수량
        """
        if price <= 0:
            return 0
            
        # 수수료 및 슬리피지를 고려한 계산
        fee_rate = 0.00015  # 매수 수수료 (0.015%)
        slippage_factor = 1 + self.slippage_tolerance  # 슬리피지 고려
        
        # 실제 매수 가능 금액 (수수료 및 슬리피지 고려)
        effective_cash = available_cash / (1 + fee_rate) / slippage_factor
        
        # 매수 가능 수량
        quantity = int(effective_cash / price)
        
        return max(0, quantity) 