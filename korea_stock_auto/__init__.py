"""
한국 주식 자동매매 프로그램
"""

from korea_stock_auto.config import AppConfig
from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.trading import TraderV2
from korea_stock_auto.service_factory import configure_services

__all__ = [
    'AppConfig',
    'KoreaInvestmentApiClient',
    'TraderV2',
    'configure_services',
]

# 패키지 인식