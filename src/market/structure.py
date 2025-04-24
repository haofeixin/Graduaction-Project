from src.order.orders import Order
from src.order.orderbooks import OrderBook
import random
import numpy as np

class Market:
    def __init__(self, orderbook: OrderBook, max_timesteps: int, config: dict):
        self.orderbook = orderbook
        self.max_timesteps = max_timesteps
        self.config = config
        self.current_time = 0
        self.agents = []

        self.price_history = []
        self.log_returns = []
        self.fundamental_price = config["market"].get("fundamental_price", 300.0)
        self.sigma_f = config["market"].get("fundamental_volatility", 0.001)

    def register_agent(self, agent):
        self.agents.append(agent)

    def step(self):
        self.orderbook.current_timestep = self.current_time
        self.orderbook.cancel_timed_out_orders(self.current_time)
        print(f"\n--- Time Step {self.current_time} ---")
        print(f"📊 Market Snapshot:")
        print(f"  - Best Bid: {self.orderbook.best_bid()}")
        print(f"  - Best Ask: {self.orderbook.best_ask()}")
        print(f"  - Last Price: {self.price_history[-1] if self.price_history else 'N/A'}")
        print(f"  - Fundamental: {self.fundamental_price:.2f}")

        Z = np.random.normal(0, 1)
        # 股票基础价格遵循几何布朗运动
        self.fundamental_price *= np.exp(-0.5 * self.sigma_f ** 2 + self.sigma_f * Z)

        

        if not self.agents:
            return

        mode = self.config["market"]["mode"]
        if mode == "single_agent_per_step":
            agent = random.choice(self.agents)
            market_snapshot = self._build_market_snapshot()
            self._process_agent(agent, market_snapshot)

        elif mode == "partial_agents_per_step":
            ratio = self.config["market"].get("activation_ratio", 0.1)
            n = max(1, int(len(self.agents) * ratio))
            selected = random.sample(self.agents, n)
            for agent in selected:
                market_snapshot = self._build_market_snapshot()
                self._process_agent(agent, market_snapshot)

        elif mode == "all_agents_per_step":
            for agent in self.agents:
                market_snapshot = self._build_market_snapshot()
                self._process_agent(agent, market_snapshot)

        self._update_price_history()
        self.current_time += 1

    def _process_agent(self, agent, market_snapshot):
        market_snapshot = self._build_market_snapshot()
        print(f"\n🧠 Agent {agent.trader_id} deciding to trade...")

        order = agent.generate_order(self.current_time, market_snapshot)
        if order:
            print(f"✅ Agent {agent.trader_id} submits order: {order}")
            self.orderbook.submit_order(order)
        else:
            print(f"❌ Agent {agent.trader_id} chose not to trade.")

    def run(self):
        for _ in range(self.max_timesteps):
            self.step()

    def _build_market_snapshot(self) -> dict:
        if self.price_history:
            last_price = self.price_history[-1]
        else:
            bid = self.orderbook.best_bid()
            ask = self.orderbook.best_ask()
            last_price = (bid + ask) / 2 if bid and ask else self.fundamental_price

        return {
            "last_price": last_price,
            "fundamental_price": self.fundamental_price,
            "log_returns": self.log_returns,
            "best_ask": self.orderbook.best_ask(),
            "best_bid": self.orderbook.best_bid()
        }

    def _update_price_history(self):
        if self.orderbook.trade_log:
            last_price = self.orderbook.trade_log[-1]['trade_price']
            self.price_history.append(last_price)
            if len(self.price_history) >= 2:
                r_t = np.log(self.price_history[-1] / self.price_history[-2])
                self.log_returns.append(r_t)

    def __repr__(self):
        return (f"<Market | timestep={self.current_time}/{self.max_timesteps} | "
                f"agents={len(self.agents)} | trades={len(self.orderbook.trade_log)} | "
                f"last_price={self.price_history[-1] if self.price_history else 'N/A'}>")
