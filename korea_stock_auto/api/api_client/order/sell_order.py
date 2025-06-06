"""
한국 주식 자동매매 - 매도 주문 모듈

주식 매도 관련 기능을 제공합니다.
"""

import json
import requests
import logging
from typing import Dict, Optional, Any, TYPE_CHECKING, cast

from korea_stock_auto.config import get_config
from korea_stock_auto.utils.utils import send_message, hashkey

# 타입 힌트만을 위한 조건부 임포트
if TYPE_CHECKING:
    from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

# 로깅 설정
logger = logging.getLogger("stock_auto")

class SellOrderMixin:
    """
    주식 매도 관련 기능 Mixin
    
    주식 매도 관련 기능을 제공합니다.
    """
    
    def sell_stock(self, code: str, qty: int, price: Optional[int] = None) -> bool:
        """
        주식 매도 함수
        
        Args:
            code: 종목 코드
            qty: 주문 수량
            price: 주문 가격 (지정가시 필수, None 또는 0 이하인 경우 시장가 주문)
            
        Returns:
            bool: 매도 성공 여부
            
        Examples:
            >>> api_client.sell_stock("005930", 10, 70000)  # 삼성전자 10주 70,000원에 지정가 매도
            >>> api_client.sell_stock("005930", 10)  # 삼성전자 10주 시장가 매도
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # 설정 로드
        config = get_config()
        
        path = "uapi/domestic-stock/v1/trading/order-cash"
        url = f"{config.current_api.base_url}/{path}"
        
        # 시장가 또는 지정가 결정
        if price is None or price <= 0:
            ord_dvsn = "01"  # 시장가
            price = 0
        else:
            ord_dvsn = "00"  # 지정가
        
        data = {
            "CANO": config.current_api.account_number,
            "ACNT_PRDT_CD": config.current_api.account_product_code,
            "PDNO": code,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": str(price),
        }
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC0801U" if config.use_realtime_api else "VTTC0801U"
        
        # 해시키 생성
        hash_val = hashkey(data, config.current_api.app_key, config.current_api.app_secret, config.current_api.base_url)
        headers = self._get_headers(tr_id, hashkey_val=hash_val)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            order_type = "시장가" if ord_dvsn == "01" else f"지정가({price:,}원)"
            logger.info(f"{code} {qty}주 매도 주문({order_type}) 요청")
            
            res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            res_json = self._handle_response(res, "매도 주문 실패")
            
            if not res_json:
                return False
            
            if res_json.get("rt_cd") == "0":
                order_no = res_json.get("output", {}).get("ODNO", "알 수 없음")
                success_msg = f"[매도 성공] {code} {qty}주 {order_type} (주문번호: {order_no})"
                logger.info(success_msg)
                send_message(success_msg, config.notification.discord_webhook_url)
                return True
            else:
                error_msg = f"[매도 실패] {res_json}"
                logger.error(error_msg)
                send_message(error_msg, config.notification.discord_webhook_url)
                return False
                
        except Exception as e:
            error_msg = f"매도 주문 중 예외 발생: {e}"
            logger.error(error_msg, exc_info=True)
            send_message(f"[오류] 매도 주문 실패: {e}", config.notification.discord_webhook_url)
            return False
    
    def fetch_sellable_quantity(self, code: str) -> Optional[Dict[str, Any]]:
        """
        주식 매도 가능 수량 조회
        
        Args:
            code: 종목 코드
            
        Returns:
            dict or None: 매도 가능 정보 (실패 시 None)
            
        Notes:
            모의투자 지원 함수입니다.
            
        Examples:
            >>> api_client.fetch_sellable_quantity("005930")  # 삼성전자 매도 가능 수량 조회
        """
        # type hint를 위한 self 타입 지정
        self = cast("KoreaInvestmentApiClient", self)
        
        # 설정 로드
        config = get_config()
        
        path = "uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{config.current_api.base_url}/{path}"
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC8434R" if config.use_realtime_api else "VTTC8434R"
        
        params = {
            "CANO": config.current_api.account_number,
            "ACNT_PRDT_CD": config.current_api.account_product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "N",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        headers = self._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} 매도 가능 수량 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} 매도 가능 수량 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            output1 = result.get("output1", {})
            output2 = result.get("output2", [])
            
            if not output2:
                logger.warning(f"{code} 보유 내역이 없습니다.")
                return {"code": code, "sellable_qty": 0, "current_price": 0, "holding_qty": 0}
            
            # 해당 종목 찾기
            stock_info = None
            for item in output2:
                if item.get("pdno") == code:
                    stock_info = {
                        "code": code,
                        "name": item.get("prdt_name", ""),
                        "holding_qty": int(item.get("hldg_qty", "0").replace(',', '')),
                        "sellable_qty": int(item.get("sll_able_qty", "0").replace(',', '')),
                        "current_price": int(item.get("prpr", "0").replace(',', '')),
                        "avg_buy_price": int(item.get("pchs_avg_pric", "0").replace(',', '')),
                        "profit_loss": int(item.get("evlu_pfls_amt", "0").replace(',', '')),
                        "profit_loss_rate": float(item.get("evlu_pfls_rt", "0").replace(',', ''))
                    }
                    break
            
            if not stock_info:
                logger.warning(f"{code} 보유 내역이 없습니다.")
                return {"code": code, "sellable_qty": 0, "current_price": 0, "holding_qty": 0}
            
            logger.info(f"{code} 매도 가능 수량 조회 성공: {stock_info['sellable_qty']}주")
            return stock_info
            
        except Exception as e:
            logger.error(f"매도 가능 수량 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] 매도 가능 수량 조회 실패: {e}", config.notification.discord_webhook_url)
            return None 