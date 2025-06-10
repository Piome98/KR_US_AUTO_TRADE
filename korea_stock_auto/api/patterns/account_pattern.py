"""
한국 주식 자동매매 - 계정 서비스

계좌 잔고, 예수금, 보유종목 등 계정 관련 기능을 제공합니다.
"""

import json
import requests
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Union

from korea_stock_auto.config import get_config
from korea_stock_auto.utils.utils import send_message

# 도메인 엔터티 import
from korea_stock_auto.domain import Portfolio

# API 매퍼 통합
from korea_stock_auto.api.mappers import AccountMapper, PortfolioMapper, MappingError
from korea_stock_auto.api.mappers.account_mapper import AccountBalance

if TYPE_CHECKING:
    from korea_stock_auto.api.client import KoreaInvestmentApiClient

logger = logging.getLogger("stock_auto")

class AccountService:
    """
    계정 관리 서비스 (API 매퍼 통합)
    
    계좌 잔고, 예수금, 보유종목 등 계정 관련 기능을 제공합니다.
    """
    
    def __init__(self, api_client: 'KoreaInvestmentApiClient'):
        """
        AccountService 초기화
        
        Args:
            api_client: KoreaInvestmentApiClient 인스턴스
        """
        self.api_client = api_client
        self.config = get_config()
        
        # API 매퍼 초기화
        self.account_mapper = AccountMapper(
            enable_cache=True,
            cache_ttl_seconds=60  # 계좌 정보는 1분 캐시
        )
        self.portfolio_mapper = PortfolioMapper(
            enable_cache=True,
            cache_ttl_seconds=120  # 포트폴리오는 2분 캐시
        )
        
        logger.debug("AccountService 초기화 완료 (API 매퍼 통합)")
    
    def get_account_balance(self) -> Optional[Dict[str, float]]:
        """
        주식 계좌 잔고 요약 조회
        
        계좌의 현금, 주식평가금액, 총평가금액 등 요약 정보를 조회합니다.
        
        Returns:
            dict or None: {
                'cash': 현금(원),
                'stocks_value': 주식평가금액(원),
                'total_assets': 총평가금액(원),
                'total_profit_loss': 총평가손익(원),
                'total_profit_loss_rate': 총수익률(%)
            } 또는 조회 실패 시 None
        """
        balance_data = self.fetch_balance()
        
        if not balance_data:
            logger.error("계좌 잔고 조회 실패")
            return None
        
        try:
            # API 응답 구조 파악 및 처리
            output1 = balance_data.get("output1")
            output2 = balance_data.get("output2")
            
            # 계좌 정보 처리 (output1)
            account_info = {}
            if isinstance(output1, list) and len(output1) > 0:
                account_info = output1[0]
            elif isinstance(output1, dict):
                account_info = output1
            elif isinstance(output2, list) and len(output2) > 0:
                account_info = output2[0]
            elif isinstance(output2, dict):
                account_info = output2
            else:
                logger.error("API 응답 형식이 예상과 다릅니다.")
                return None
            
            if not isinstance(account_info, dict):
                logger.error(f"계좌 정보가 딕셔너리가 아닙니다: {type(account_info)}")
                return None
            
            # 금액 정보 추출
            result = {
                "cash": float(account_info.get("dnca_tot_amt", "0").replace(',', '')),  # 예수금총금액
                "stocks_value": float(account_info.get("scts_evlu_amt", "0").replace(',', '')),  # 유가증권평가금액
                "total_assets": float(account_info.get("tot_evlu_amt", "0").replace(',', '')),  # 총평가금액
                "total_profit_loss": float(account_info.get("evlu_pfls_smtl_amt", "0").replace(',', '')),  # 평가손익합계금액
                "total_profit_loss_rate": float(account_info.get("evlu_pfls_rt", "0").replace(',', ''))  # 평가손익률
            }
            
            return result
            
        except Exception as e:
            logger.error(f"계좌 잔고 처리 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] 계좌 잔고 처리 실패: {e}", get_config().notification.discord_webhook_url)
            return None
    
    def get_account_balance_as_entity(self) -> Optional[Portfolio]:
        """
        계좌 잔고를 Portfolio 엔터티로 조회 (NEW)
        
        Returns:
            Portfolio: Portfolio 엔터티 또는 실패 시 None
        """
        try:
            # 원시 API 응답 조회
            raw_response = self._get_account_balance_raw()
            if not raw_response:
                return None
            
            # PortfolioMapper를 통한 Portfolio 엔터티 생성
            try:
                portfolio = self.portfolio_mapper.map_from_balance_response(raw_response)
                logger.debug(f"Portfolio 엔터티 생성 완료: 현금 {portfolio.cash}, 보유종목 {portfolio.get_position_count()}개")
                return portfolio
                
            except MappingError as e:
                logger.error(f"Portfolio 매핑 실패: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Portfolio 엔터티 조회 실패: {e}")
            return None
    
    def _get_account_balance_raw(self) -> Optional[Dict[str, Any]]:
        """
        계좌 잔고 원시 API 응답 조회
        
        Returns:
            dict or None: 원시 API 응답
        """
        path = "uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{self.config.current_api.base_url}/{path}"
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC8434R" if self.config.use_realtime_api else "VTTC8434R"
        
        params = {
            "CANO": self.config.current_api.account_number,
            "ACNT_PRDT_CD": self.config.current_api.account_product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        headers = self.api_client._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self.api_client._rate_limit()
            
            logger.info("계좌 잔고 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self.api_client._handle_response(res, "계좌 잔고 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"계좌 잔고 원시 조회 실패: {e}")
            send_message(f"[오류] 계좌 잔고 조회 실패: {e}", self.config.notification.discord_webhook_url)
            return None
    
    def _convert_portfolio_to_legacy_format(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Portfolio 엔터티를 기존 형식의 딕셔너리로 변환"""
        holdings = []
        
        for code, position in portfolio.positions.items():
            holding_info = {
                "code": position.stock.code,
                "name": position.stock.name,
                "quantity": position.quantity.value,
                "average_price": int(position.average_price.value.to_float()),
                "current_price": int(position.stock.current_price.value.to_float()),
                "evaluation_amount": int(position.current_value().to_float()),
                "purchase_amount": int(position.purchase_amount().to_float()),
                "profit_loss": int(position.unrealized_profit_loss().to_float()),
                "profit_loss_rate": position.unrealized_profit_loss_percentage()
            }
            holdings.append(holding_info)
        
        # 포트폴리오 전체 통계 계산
        from korea_stock_auto.domain.value_objects import Money
        
        # 총 주식 평가액 계산
        total_stock_value = Money.zero()
        total_investment = Money.zero()
        total_pnl = Money.zero()
        
        for position in portfolio.positions.values():
            if not position.is_empty():
                total_stock_value = total_stock_value + position.current_value()
                total_investment = total_investment + position.total_cost()
                
                # 손익 계산
                current_val = position.current_value()
                cost = position.total_cost()
                pnl = Money.won(current_val.amount - cost.amount)
                total_pnl = total_pnl + pnl
        
        total_assets = portfolio.cash + total_stock_value
        
        # 수익률 계산
        total_profit_loss_rate = 0.0
        if not total_investment.is_zero():
            total_profit_loss_rate = (total_pnl.amount / total_investment.amount) * 100
        
        account_info = {
            "total_asset": int(total_assets.to_float()),
            "cash_balance": int(portfolio.cash.to_float()),
            "stock_evaluation": int(total_stock_value.to_float()),
            "total_profit_loss": int(total_pnl.to_float()),
            "total_profit_loss_rate": total_profit_loss_rate,
            "holdings": holdings,
            "holdings_count": len(holdings)
        }
        
        logger.info(f"계좌 잔고 조회 성공 (매퍼): 총자산 {account_info['total_asset']:,}원, 보유종목 {account_info['holdings_count']}개")
        return account_info
    
    def _process_account_balance_legacy(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """기존 방식의 계좌 잔고 처리 (백워드 호환성)"""
        try:
            # 보유종목 정보
            holdings = []
            output1_list = result.get("output1", [])
            
            for stock in output1_list:
                if int(stock.get("hldg_qty", "0")) > 0:  # 보유수량이 0보다 큰 경우만
                    holding_info = {
                        "code": stock.get("pdno", ""),
                        "name": stock.get("prdt_name", ""),
                        "quantity": int(stock.get("hldg_qty", "0")),
                        "average_price": int(stock.get("pchs_avg_pric", "0")),
                        "current_price": int(stock.get("prpr", "0")),
                        "evaluation_amount": int(stock.get("evlu_amt", "0")),
                        "purchase_amount": int(stock.get("pchs_amt", "0")),
                        "profit_loss": int(stock.get("evlu_pfls_amt", "0")),
                        "profit_loss_rate": float(stock.get("evlu_pfls_rt", "0"))
                    }
                    holdings.append(holding_info)
            
            # 계좌 전체 정보
            output2 = result.get("output2", [{}])[0] if result.get("output2") else {}
            
            account_info = {
                "total_asset": int(output2.get("tot_evlu_amt", "0")),  # 총평가금액
                "cash_balance": int(output2.get("dnca_tot_amt", "0")),  # 예수금총액
                "stock_evaluation": int(output2.get("scts_evlu_amt", "0")),  # 유가증권평가금액
                "total_profit_loss": int(output2.get("evlu_pfls_smtl_amt", "0")),  # 평가손익합계금액
                "total_profit_loss_rate": float(output2.get("evlu_pfls_rt", "0")),  # 평가손익률
                "holdings": holdings,
                "holdings_count": len(holdings)
            }
            
            logger.info(f"계좌 잔고 조회 성공 (legacy): 총자산 {account_info['total_asset']:,}원, 보유종목 {account_info['holdings_count']}개")
            return account_info
            
        except Exception as e:
            logger.error(f"Legacy 계좌 잔고 처리 실패: {e}")
            return None

    def get_deposit_info(self) -> Optional[Dict[str, Any]]:
        """
        예수금 상세 정보 조회 (백워드 호환성 유지)
        
        Returns:
            dict or None: 예수금 정보 (실패 시 None)
            
        Examples:
            >>> account_service.get_deposit_info()
        """
        try:
            # 원시 API 응답 조회
            raw_response = self._get_deposit_info_raw()
            if not raw_response:
                return None
            
            # AccountMapper를 통한 AccountBalance 엔터티 생성 시도
            try:
                account_balance = self.account_mapper.map_from_deposit_response(raw_response)
                
                # AccountBalance 엔터티에서 백워드 호환성 딕셔너리 생성
                deposit_info = {
                    "deposited_cash": int(account_balance.cash.to_float()),
                    "available_cash": int(account_balance.available_cash.to_float()),
                    "order_possible_cash": int(account_balance.order_possible_cash.to_float()),
                    "withdrawal_possible_amount": int(account_balance.withdrawal_possible_amount.to_float()),
                    "cma_evaluation_amount": int(account_balance.cma_evaluation_amount.to_float()),
                    "total_evaluation_amount": int(account_balance.total_assets.to_float())
                }
                
                logger.info(f"예수금 정보 조회 성공 (매퍼): 가용현금 {deposit_info['available_cash']:,}원")
                return deposit_info
                
            except MappingError as e:
                logger.warning(f"AccountBalance 매핑 실패, 기존 방식 사용: {e}")
                # 백워드 호환성: 기존 방식으로 처리
                return self._process_deposit_info_legacy(raw_response)
                
        except Exception as e:
            logger.error(f"예수금 정보 조회 실패: {e}")
            return None
    
    def get_deposit_info_as_entity(self) -> Optional[AccountBalance]:
        """
        예수금 정보를 AccountBalance 엔터티로 조회 (NEW)
        
        Returns:
            AccountBalance: AccountBalance 엔터티 또는 실패 시 None
        """
        try:
            # 원시 API 응답 조회
            raw_response = self._get_deposit_info_raw()
            if not raw_response:
                return None
            
            # AccountMapper를 통한 AccountBalance 엔터티 생성
            try:
                account_balance = self.account_mapper.map_from_deposit_response(raw_response)
                logger.debug(f"AccountBalance 엔터티 생성 완료: 가용현금 {account_balance.available_cash}")
                return account_balance
                
            except MappingError as e:
                logger.error(f"AccountBalance 매핑 실패: {e}")
                return None
                
        except Exception as e:
            logger.error(f"AccountBalance 엔터티 조회 실패: {e}")
            return None
    
    def _get_deposit_info_raw(self) -> Optional[Dict[str, Any]]:
        """
        예수금 정보 원시 API 응답 조회
        
        Returns:
            dict or None: 원시 API 응답
        """
        path = "uapi/domestic-stock/v1/trading/inquire-psbl-order"
        url = f"{self.config.current_api.base_url}/{path}"
        
        # 트랜잭션 ID는 실전/모의 환경에 따라 다름
        tr_id = "TTTC8908R" if self.config.use_realtime_api else "VTTC8908R"
        
        params = {
            "CANO": self.config.current_api.account_number,
            "ACNT_PRDT_CD": self.config.current_api.account_product_code,
            "PDNO": "005930",  # 임시 종목코드 (실제로는 종목코드와 무관하게 예수금 정보 조회)
            "ORD_UNPR": "0",
            "ORD_DVSN": "01",
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "N"
        }
        
        headers = self.api_client._get_headers(tr_id)
        
        try:
            # API 호출 속도 제한 적용
            self.api_client._rate_limit()
            
            logger.info("예수금 정보 조회 요청")
            res = requests.get(url, headers=headers, params=params, timeout=10)
            result = self.api_client._handle_response(res, "예수금 정보 조회 실패")
            
            if not result or result.get("rt_cd") != "0":
                return None
                
            return result
            
        except Exception as e:
            logger.error(f"예수금 정보 원시 조회 실패: {e}")
            send_message(f"[오류] 예수금 정보 조회 실패: {e}", self.config.notification.discord_webhook_url)
            return None
    
    def _process_deposit_info_legacy(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """기존 방식의 예수금 정보 처리 (백워드 호환성)"""
        try:
            output = result.get("output", {})
            
            # 결과 가공
            deposit_info = {
                "deposited_cash": int(output.get("dnca_tot_amt", "0")),  # 예수금 총액
                "available_cash": int(output.get("prvs_rcdl_excc_amt", "0")),  # 가용 현금
                "order_possible_cash": int(output.get("ord_psbl_cash", "0")),  # 주문가능현금
                "withdrawal_possible_amount": int(output.get("wdrw_psbl_tot_amt", "0")),  # 출금가능총액
                "cma_evaluation_amount": int(output.get("cma_evlu_amt", "0")),  # CMA평가금액
                "total_evaluation_amount": int(output.get("tot_evlu_amt", "0"))  # 총평가금액
            }
            
            logger.info(f"예수금 정보 조회 성공 (legacy): 가용현금 {deposit_info['available_cash']:,}원")
            return deposit_info
            
        except Exception as e:
            logger.error(f"Legacy 예수금 정보 처리 실패: {e}")
            return None
    
    def clear_mappers_cache(self) -> None:
        """매퍼 캐시 전체 삭제"""
        self.account_mapper.clear_cache()
        self.portfolio_mapper.clear_cache()
        logger.debug("AccountMapper, PortfolioMapper 캐시 삭제 완료")
    
    def get_mappers_cache_stats(self) -> Dict[str, Any]:
        """매퍼 캐시 통계 조회"""
        return {
            'account_mapper': self.account_mapper.get_cache_stats(),
            'portfolio_mapper': self.portfolio_mapper.get_cache_stats()
        }
    
    def get_stock_balance(self) -> Optional[Dict[str, Any]]:
        """
        주식 보유 종목 조회
        
        계좌에 보유 중인 주식 종목 목록과 상세 정보를 조회합니다.
        
        Returns:
            dict or None: {
                'account': {계좌 요약 정보},
                'stocks': [보유 종목 목록]
            } 또는 조회 실패 시 None
        """
        balance_data = self.fetch_balance()
        
        if not balance_data:
            logger.error("계좌 잔고 조회 실패")
            return None
        
        try:
            # API 응답 구조 파악 및 처리
            output1 = balance_data.get("output1")
            output2 = balance_data.get("output2")
            
            # 계좌 정보 처리 (output1)
            account_info = {}
            if isinstance(output1, list) and len(output1) > 0:
                account_info = output1[0]
            elif isinstance(output1, dict):
                account_info = output1
            elif isinstance(output2, list) and len(output2) > 0:
                account_info = output2[0]
            elif isinstance(output2, dict):
                account_info = output2
            else:
                logger.error("API 응답 형식이 예상과 다릅니다.")
                logger.error(f"balance_data: {balance_data}")
                return None
            
            if not isinstance(account_info, dict):
                logger.error(f"계좌 정보가 딕셔너리가 아닙니다: {type(account_info)}")
                return None
            
            # 총액, 평가금액, 수익률 등 정보
            total_data = {
                "dnca_tot_amt": float(account_info.get("dnca_tot_amt", "0").replace(',', '')),  # 예수금총금액
                "scts_evlu_amt": float(account_info.get("scts_evlu_amt", "0").replace(',', '')),  # 유가증권평가금액
                "tot_evlu_amt": float(account_info.get("tot_evlu_amt", "0").replace(',', '')),  # 총평가금액
                "nass_amt": float(account_info.get("nass_amt", "0").replace(',', '')),  # 순자산금액
                "pchs_amt_smtl_amt": float(account_info.get("pchs_amt_smtl_amt", "0").replace(',', '')),  # 매입금액합계금액
                "evlu_pfls_smtl_amt": float(account_info.get("evlu_pfls_smtl_amt", "0").replace(',', '')),  # 평가손익합계금액
                "evlu_pfls_rt": float(account_info.get("evlu_pfls_rt", "0").replace(',', '')),  # 평가손익률
                "total_profit_loss_rate": float(account_info.get("evlu_pfls_rt", "0").replace(',', '')),  # 총수익률
                "cash": float(account_info.get("dnca_tot_amt", "0").replace(',', '')),  # 예수금총액
                "stocks_value": float(account_info.get("scts_evlu_amt", "0").replace(',', '')),  # 주식평가금액
                "total_assets": float(account_info.get("tot_evlu_amt", "0").replace(',', '')),  # 총자산
            }
            
            # 보유종목 목록 처리 (output2)
            stock_list = []
            
            # output2가 리스트인 경우 (일반적인 경우)
            if isinstance(output2, list):
                stock_items = output2[1:] if len(output2) > 1 else []
            # output2가 딕셔너리인 경우 (실전투자 환경에서 응답 형식이 다를 수 있음)
            elif isinstance(output2, dict):
                # output2가 단일 종목 정보를 포함하는 딕셔너리인 경우
                if output2.get("pdno"):  # 종목코드가 있는 경우
                    stock_items = [output2]
                else:
                    stock_items = []
            else:
                stock_items = []
                logger.warning(f"보유종목 정보 형식이 예상과 다릅니다: {type(output2)}")
            
            for stock in stock_items:
                if not isinstance(stock, dict) or not stock.get("pdno"):  # 종목코드가 없는 경우 건너뛰기
                    continue
                    
                stock_data = {
                    "prdt_name": stock.get("prdt_name", ""),  # 종목명
                    "code": stock.get("pdno", ""),  # 종목코드
                    "units": int(stock.get("hldg_qty", "0").replace(',', '')),  # 보유수량
                    "avg_buy_price": int(stock.get("pchs_avg_pric", "0").replace(',', '')),  # 매입평균가격
                    "current_price": int(stock.get("prpr", "0").replace(',', '')),  # 현재가
                    "value": int(stock.get("evlu_amt", "0").replace(',', '')),  # 평가금액
                    "earning_rate": float(stock.get("evlu_pfls_rt", "0").replace(',', '')),  # 평가손익률
                    "purchase_amount": int(stock.get("pchs_amt", "0").replace(',', '')),  # 매입금액
                    "profit_loss": int(stock.get("evlu_pfls_amt", "0").replace(',', '')),  # 평가손익금액
                    "pnl_ratio": float(stock.get("evlu_pfls_rt", "0").replace(',', '')),  # 평가손익률
                    "sellable_units": int(stock.get("sll_able_qty", "0").replace(',', '')),  # 매도가능수량
                }
                
                stock_list.append(stock_data)
            
            result = {
                "account": total_data,
                "stocks": stock_list
            }
            
            return result
            
        except Exception as e:
            logger.error(f"주식 보유 종목 처리 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] 주식 보유 종목 처리 실패: {e}", get_config().notification.discord_webhook_url)
            return None
    
    def fetch_balance(self) -> Optional[Dict[str, Any]]:
        """
        계좌 잔고 정보 조회
        
        한국투자증권 API를 통해 계좌 잔고 정보를 조회합니다.
        
        Returns:
            dict or None: 계좌 잔고 정보 또는 조회 실패 시 None
            
        Notes:
            모의투자와 실전투자 모두 지원하는 함수입니다.
        """
        config = get_config()
        
        path = "uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{config.current_api.base_url}/{path}"
        
        # 실전/모의투자 구분
        tr_id = "TTTC8434R" if config.use_realtime_api else "VTTC8434R"
        
        params = {
            "CANO": config.current_api.account_number,
            "ACNT_PRDT_CD": config.current_api.account_product_code,
            "AFHR_FLPR_YN": "N",  # 시간외단일가여부
            "OFL_YN": "N",  # 오프라인여부
            "INQR_DVSN": "01",  # 조회구분: 01-요약
            "UNPR_DVSN": "01",  # 단가구분: 01-평균단가
            "FUND_STTL_ICLD_YN": "N",  # 펀드결제분포함여부
            "FNCG_AMT_AUTO_RDPT_YN": "N",  # 융자금액자동상환여부
            "PRCS_DVSN": "01",  # 처리구분: 01-조회
            "CTX_AREA_FK100": "",  # 연속조회검색조건
            "CTX_AREA_NK100": ""   # 연속조회키
        }
        
        try:
            headers = self.api_client._get_headers(tr_id)
            
            logger.debug("계좌 잔고 조회 요청")
            result = self.api_client._request_get(url, headers, params, "계좌 잔고 조회 실패")
            
            if result:
                logger.debug("계좌 잔고 조회 성공")
            
            return result
            
        except Exception as e:
            logger.error(f"계좌 잔고 조회 중 예외 발생: {e}", exc_info=True)
            send_message(f"[오류] 계좌 잔고 조회 실패: {e}", config.notification.discord_webhook_url)
            return None 