"""
한국 주식 자동매매 - 데이터베이스 패키지

주가 데이터 및 거래 내역 저장을 담당하는 모듈들을 제공합니다.
"""

from korea_stock_auto.data.database.base import DatabaseManager
from korea_stock_auto.data.database.price_data import PriceDataManager
from korea_stock_auto.data.database.transaction_data import TransactionDataManager
from korea_stock_auto.data.database.technical_data import TechnicalDataManager
from korea_stock_auto.data.database.market_data import MarketDataManager
from korea_stock_auto.data.database.news_data import NewsDataManager
from korea_stock_auto.data.database.trading_stats import TradingStatsManager

# 기존 클래스와의 호환성을 위한 통합 클래스
from korea_stock_auto.data.database.stock_database import StockDatabase

__all__ = [
    'DatabaseManager',
    'PriceDataManager',
    'TransactionDataManager',
    'TechnicalDataManager',
    'MarketDataManager',
    'NewsDataManager',
    'TradingStatsManager',
    'StockDatabase',
] 