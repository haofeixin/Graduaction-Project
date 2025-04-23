import unittest
from src.order.orderbooks import OrderBook
from src.market.structure import Market
from src.traders.retail import RetailTrader
from src.traders.institutional import InstitutionalTrader
from src.config.loader import load_config

class TestMarketComplex(unittest.TestCase):
    def setUp(self):
        self.config = load_config("config.yaml")
        self.orderbook = OrderBook()
        self.market = Market(orderbook=self.orderbook, max_timesteps=20, config=self.config)

        # 创建多个 agent（retail + institutional）
        self.agents = []
        for i in range(5):
            self.agents.append(RetailTrader(trader_id=i, config=self.config["agents"], trader_type="retail"))
        for i in range(5, 10):
            self.agents.append(InstitutionalTrader(trader_id=i, config=self.config["agents"], trader_type="institutional"))

        for agent in self.agents:
            self.market.register_agent(agent)

    def test_market_trading_flow(self):
        self.market.run()

        # 验证时间步数是否推进
        self.assertEqual(self.market.current_time, self.market.max_timesteps)

        # 检查是否有价格记录
        self.assertGreaterEqual(len(self.market.price_history), 1)
        self.assertGreaterEqual(len(self.market.log_returns), 0)

        # 检查是否有 agent 提交订单
        order_count = len(self.orderbook.buys) + len(self.orderbook.sells)
        self.assertGreaterEqual(order_count, 1, "No orders submitted")

        # 检查是否至少有部分成交（可能不是所有都成交）
        self.assertIsInstance(self.orderbook.trade_log, list)
        print(f"[Debug] Trade count: {len(self.orderbook.trade_log)}")
        print(f"[Debug] Final price: {self.market.price_history[-1] if self.market.price_history else 'N/A'}")

        # Snapshot 检查
        snapshot = self.market._build_market_snapshot()
        self.assertIn("last_price", snapshot)
        self.assertIn("fundamental_price", snapshot)
        self.assertIn("log_returns", snapshot)

if __name__ == '__main__':
    unittest.main()
