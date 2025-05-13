"""
한국 주식 자동매매 - 주식 주문 모듈

주식 매수, 매도, 정정, 취소 등 주문 관련 기능을 제공합니다.
"""

from korea_stock_auto.api.api_client.order.buy_order import BuyOrderMixin
from korea_stock_auto.api.api_client.order.sell_order import SellOrderMixin
from korea_stock_auto.api.api_client.order.modify_order import OrderModificationMixin
from korea_stock_auto.api.api_client.order.order_info import OrderInfoMixin

class StockOrderMixin(BuyOrderMixin, SellOrderMixin, OrderModificationMixin, OrderInfoMixin):
    """
    주식 주문 관련 기능 Mixin
    
    주식 매수, 매도, 정정, 취소 등 주문 관련 기능을 제공합니다.
    """
    pass 