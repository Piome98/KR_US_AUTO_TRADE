"""
한국 주식 자동매매 - 시장 가격 조회 모듈
"""

import requests
import logging
from typing import Dict, List, Optional, Any, Union

from korea_stock_auto.config import URL_BASE
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient

logger = logging.getLogger("stock_auto")

class MarketPriceMixin:
    """시장 가격 조회 관련 기능 Mixin"""
    
    def fetch_stock_current_price(self, code: str) -> Optional[Dict[str, Any]]:
        """
        주식 현재가 조회
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict or None: 주식 현재가 정보
            
        Notes:
            모의투자 지원 함수입니다.
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/inquire-price"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": code,
        }
        
        headers = self._get_headers("FHKST01010100")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} 현재가 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} 현재가 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
            
            output = result.get("output", {})
            
            # 주요 정보 추출 및 가공
            price_info = {
                "stock_code": code,
                "stock_name": output.get("prdt_abrv_name", ""),
                "market": output.get("rprs_mrkt_kor_name", ""),
                "time": output.get("stck_basc_hour", ""),
                "current_price": int(output.get("stck_prpr", "0").replace(',', '')),
                "open_price": int(output.get("stck_oprc", "0").replace(',', '')),
                "high_price": int(output.get("stck_hgpr", "0").replace(',', '')),
                "low_price": int(output.get("stck_lwpr", "0").replace(',', '')),
                "prev_close_price": int(output.get("stck_sdpr", "0").replace(',', '')),
                "price_change": int(output.get("prdy_vrss", "0").replace(',', '')),
                "change_rate": float(output.get("prdy_ctrt", "0")),
                "volume": int(output.get("acml_vol", "0").replace(',', '')),
                "volume_value": int(output.get("acml_tr_pbmn", "0").replace(',', '')),
                "market_cap": int(output.get("hts_avls", "0").replace(',', '')),
                "listed_shares": int(output.get("lstn_stcn", "0").replace(',', '')),
                "highest_52_week": int(output.get("w52_hgpr", "0").replace(',', '')),
                "lowest_52_week": int(output.get("w52_lwpr", "0").replace(',', '')),
                "per": float(output.get("per", "0").replace(',', '')),
                "eps": float(output.get("eps", "0").replace(',', '')),
                "pbr": float(output.get("pbr", "0").replace(',', '')),
                "div_yield": float(output.get("dvr", "0").replace(',', '')),
                "foreign_rate": float(output.get("frgn_hldn_qty_rt", "0").replace(',', ''))
            }
            
            # 추가 분석 정보
            price_info["day_range_rate"] = ((price_info["high_price"] - price_info["low_price"]) / price_info["low_price"] * 100) if price_info["low_price"] > 0 else 0
            price_info["current_to_open_rate"] = ((price_info["current_price"] - price_info["open_price"]) / price_info["open_price"] * 100) if price_info["open_price"] > 0 else 0
            price_info["is_52week_high"] = price_info["current_price"] >= price_info["highest_52_week"]
            price_info["is_52week_low"] = price_info["current_price"] <= price_info["lowest_52_week"]
            
            # 매매 신호 관련 정보 (단순 분석 목적)
            price_info["gap_from_52week_high"] = ((price_info["highest_52_week"] - price_info["current_price"]) / price_info["current_price"] * 100) if price_info["current_price"] > 0 else 0
            price_info["gap_from_52week_low"] = ((price_info["current_price"] - price_info["lowest_52_week"]) / price_info["lowest_52_week"] * 100) if price_info["lowest_52_week"] > 0 else 0
            
            logger.info(f"{code} 현재가 조회 성공: {price_info['current_price']}원 ({price_info['change_rate']}%)")
            return price_info
            
        except Exception as e:
            logger.error(f"{code} 현재가 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 현재가 조회 실패: {e}")
            return None
            
    def get_stock_asking_price(self, code: str) -> Optional[Dict[str, Any]]:
        """
        주식 현재가 호가 조회
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict or None: 주식 호가 정보
            
        Notes:
            모의투자 지원 함수입니다.
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/inquire-asking-price"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": code,
        }
        
        headers = self._get_headers("FHKST01010200")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} 호가 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} 호가 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            # 매도/매수 호가 정보 추출
            output = result.get("output", {})
            
            # 10단계 호가 정보 정리
            asking_prices = {
                "stock_code": code,
                "time": output.get("stck_bsop_hour", ""),
                "total_ask_qty": int(output.get("total_askp_rsqn", "0").replace(',', '')),
                "total_bid_qty": int(output.get("total_bidp_rsqn", "0").replace(',', '')),
                "asks": [],  # 매도호가
                "bids": [],  # 매수호가
                "ask_prices": [],  # 매도호가 가격만
                "bid_prices": [],  # 매수호가 가격만
                "ask_quantities": [],  # 매도호가 수량만
                "bid_quantities": []   # 매수호가 수량만
            }
            
            # 매도호가 정보 추출 (1~10)
            for i in range(1, 11):
                ask_price = int(output.get(f"askp{i}", "0").replace(',', ''))
                ask_qty = int(output.get(f"askp_rsqn{i}", "0").replace(',', ''))
                
                asking_prices["asks"].append({"price": ask_price, "quantity": ask_qty})
                asking_prices["ask_prices"].append(ask_price)
                asking_prices["ask_quantities"].append(ask_qty)
            
            # 매수호가 정보 추출 (1~10)
            for i in range(1, 11):
                bid_price = int(output.get(f"bidp{i}", "0").replace(',', ''))
                bid_qty = int(output.get(f"bidp_rsqn{i}", "0").replace(',', ''))
                
                asking_prices["bids"].append({"price": bid_price, "quantity": bid_qty})
                asking_prices["bid_prices"].append(bid_price)
                asking_prices["bid_quantities"].append(bid_qty)
            
            # 최고/최저 호가
            asking_prices["highest_ask"] = asking_prices["ask_prices"][0]
            asking_prices["lowest_bid"] = asking_prices["bid_prices"][0]
            
            # 호가 스프레드
            asking_prices["spread"] = asking_prices["highest_ask"] - asking_prices["lowest_bid"]
            
            logger.info(f"{code} 호가 조회 성공")
            return asking_prices
            
        except Exception as e:
            logger.error(f"{code} 호가 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 호가 조회 실패: {e}")
            return None
    
    def get_stock_conclusion(self, code: str) -> Optional[Dict[str, Any]]:
        """
        주식 현재가 체결 정보 조회
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict or None: 주식 체결 정보
            
        Notes:
            모의투자 지원 함수입니다.
        """
        self: KoreaInvestmentApiClient  # type hint
        
        path = "uapi/domestic-stock/v1/quotations/inquire-ccnl"
        url = f"{URL_BASE}/{path}"
        
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": code,
        }
        
        headers = self._get_headers("FHPST01010300")
        
        try:
            # API 호출 속도 제한 적용
            self._rate_limit()
            
            logger.info(f"{code} 체결 정보 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self._handle_response(res, f"{code} 체결 정보 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            # 체결 정보 추출
            output1 = result.get("output1", {})
            output2 = result.get("output2", [])
            
            if not output2:
                logger.warning(f"{code} 체결 정보가 없습니다.")
                return {
                    "stock_code": code,
                    "stock_name": output1.get("hts_kor_isnm", ""),
                    "conclusions": []
                }
            
            # 현재가 정보
            current_info = {
                "stock_code": code,
                "stock_name": output1.get("hts_kor_isnm", ""),
                "current_price": int(output1.get("stck_prpr", "0").replace(',', '')),
                "price_change": int(output1.get("prdy_vrss", "0").replace(',', '')),
                "change_rate": float(output1.get("prdy_ctrt", "0").replace(',', '')),
                "volume": int(output1.get("acml_vol", "0").replace(',', '')),
                "conclusions": []
            }
            
            # 체결 내역 정보 추출
            for item in output2:
                conclusion = {
                    "time": item.get("stck_cntg_hour", ""),
                    "price": int(item.get("stck_prpr", "0").replace(',', '')),
                    "quantity": int(item.get("cntg_qty", "0").replace(',', '')),
                    "change_type": item.get("prdy_vrss_sign", ""),  # 1:상한, 2:상승, 3:보합, 4:하한, 5:하락
                    "volume": int(item.get("acml_vol", "0").replace(',', '')),
                }
                current_info["conclusions"].append(conclusion)
            
            logger.info(f"{code} 체결 정보 조회 성공: {len(current_info['conclusions'])}건")
            return current_info
            
        except Exception as e:
            logger.error(f"{code} 체결 정보 조회 실패: {e}", exc_info=True)
            send_message(f"[오류] {code} 체결 정보 조회 실패: {e}")
            return None 