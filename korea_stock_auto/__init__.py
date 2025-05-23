"""
한국 주식 자동매매 - 메인 패키지
Korea Investment Auto Trading System
"""

__version__ = "1.0.0"

# 모듈별 주요 기능 임포트
from korea_stock_auto.trading.trader import Trader
from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient


__all__ = [
    'Trader',
    'KoreaInvestmentApiClient',
]

# 패키지 인식