import random
import numpy as np
from src.order.orders import Order, OrderDirection, OrderType
from src.order.orderbooks import OrderBook

class Market:
    def __init__(self, orderbook: OrderBook, max_timesteps: int, config: dict):
        self.orderbook = orderbook
        self.max_timesteps = max_timesteps
        self.config = config
        self.current_time = 0
        self.agents = []

        # 市场状态
        self.price_history = []
        self.log_returns = []

        # fundamental price 设定（GBM）
        self.fundamental_price = config["market"].get("fundamental_price", 300.0)
        self.sigma_f = config["market"].get("fundamental_volatility", 0.001)

    def register_agent(self, agent):
        self.agents.append(agent)

    def step(self):
        self.orderbook.current_timestep = self.current_time
        self.orderbook.cancel_timed_out_orders(self.current_time)

        # 更新 fundamental price (GBM)
        Z = np.random.normal(0, 1)
        self.fundamental_price *= np.exp(-0.5 * self.sigma_f ** 2 + self.sigma_f * Z)

        # 构造市场快照
        market_snapshot = self._build_market_snapshot()

        # 激活 agent 行为
        if not self.agents:
            
            return

        mode = self.config["market"]["mode"]
        if mode == "single_agent_per_step":
            agent = random.choice(self.agents)
            self._process_agent(agent, market_snapshot)

        elif mode == "partial_agents_per_step":
            ratio = self.config["market"].get("activation_ratio", 0.1)
            n = max(1, int(len(self.agents) * ratio))
            selected = random.sample(self.agents, n)
            for agent in selected:
                self._process_agent(agent, market_snapshot)

        elif mode == "all_agents_per_step":
            for agent in self.agents:
                self._process_agent(agent, market_snapshot)

        self._update_price_history()
        self.current_time += 1

    def _process_agent(self, agent, market_snapshot):
        order = agent.generate_order(self.current_time, market_snapshot)
        if order:
            interpreted = self.interpret_order(order)
            self.orderbook.submit_order(interpreted)

    def interpret_order(self, raw_order: Order) -> Order:
        best_ask = self.orderbook.best_ask()
        best_bid = self.orderbook.best_bid()

        if raw_order.direction == OrderDirection.BUY and best_ask is not None and best_ask <= raw_order.price:
            return Order(
                trader_id=raw_order.trader_id,
                order_type=OrderType.MARKET,
                direction=raw_order.direction,
                quantity=raw_order.quantity,
                timestep=self.current_time
            )
        elif raw_order.direction == OrderDirection.SELL and best_bid is not None and best_bid >= raw_order.price:
            return Order(
                trader_id=raw_order.trader_id,
                order_type=OrderType.MARKET,
                direction=raw_order.direction,
                quantity=raw_order.quantity,
                timestep=self.current_time
            )
        return raw_order

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
            "log_returns": self.log_returns
        }

    def _update_price_history(self):
        if self.orderbook.trade_log:
            last_price = self.orderbook.trade_log[-1][2]
            self.price_history.append(last_price)
            if len(self.price_history) >= 2:
                r_t = np.log(self.price_history[-1] / self.price_history[-2])
                self.log_returns.append(r_t)
