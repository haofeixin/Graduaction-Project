import unittest
from src.order.orders import Order, OrderDirection, OrderType
from src.order.orderbooks import OrderBook 
class TestOrderBook(unittest.TestCase):

    def setUp(self):
        """初始化订单簿和订单"""
        self.orderbook = OrderBook()

    def test_trader_decides_to_submit_market_order(self):
        """测试交易者在市价合适时选择提交市价单"""
        buy_order = Order(1, OrderType.LIMIT, OrderDirection.BUY, 500, 0, price=10.1)
        sell_orders = [
            Order(2, OrderType.MARKET, OrderDirection.SELL, 200, 1, price=10.0),  # 可成交
            Order(3, OrderType.MARKET, OrderDirection.SELL, 200, 2, price=10.1),  # 可成交
            Order(4, OrderType.LIMIT, OrderDirection.SELL, 500, 3, price=10.2)   # 不可成交
        ]
        self.orderbook.submit_order(buy_order)
        for sell_order in sell_orders:
            self.orderbook.submit_order(sell_order)
        self.assertEqual(self.orderbook.buys[0][2].quantity, 100)
        self.assertEqual(self.orderbook.sells[0][2].quantity, 500)

        new_order = Order(5, OrderType.MARKET, OrderDirection.BUY, 500, 4, price=10.2)
        self.orderbook.submit_order(new_order)
        self.assertEqual(self.orderbook.buys[0][2].quantity, 100)
        self.assertEqual(len(self.orderbook.buys), 1)
        self.assertEqual(len(self.orderbook.sells), 0)
        print(self.orderbook.trade_log)
    

if __name__ == '__main__':
    unittest.main()
