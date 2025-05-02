"""
한국 주식 자동매매 - 메인 패키지
Korea Investment Auto Trading System
"""

__version__ = "1.0.0"

# 모듈별 주요 기능 임포트
from korea_stock_auto.trading.trading import StockTrader
from korea_stock_auto.api.websocket import StockWebSocket
from korea_stock_auto.data.database import StockDatabase
from korea_stock_auto.models.model_manager import ModelVersionManager

# 패키지 인식