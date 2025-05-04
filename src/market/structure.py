from src.order.orders import Order
from src.order.orderbooks import OrderBook
from src.social.cyberbullying import CyberbullyingModel
import random
import numpy as np
from typing import Optional
class Market:
    def __init__(self, orderbook: OrderBook, max_timesteps: int, config: dict, cyberbullying_model: Optional[CyberbullyingModel] = None):
        self.orderbook = orderbook
        self.max_timesteps = max_timesteps
        self.config = config
        self.current_time = 0
        self.agents = []

        self.price_history = []
        self.log_returns = []
        self.fundamental_price = config["market"].get("fundamental_price", 300.0)
        self.fundamental_price_history = [self.fundamental_price]  # 记录基础价格历史
        self.sigma_f = config["market"].get("fundamental_volatility", 0.001)
        self.cyberbullying_model = cyberbullying_model

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
        # 股票基础价格遵循带正漂移项的几何布朗运动
        mu = self.config["market"].get("fundamental_drift", 0.0001)
        self.fundamental_price *= np.exp((mu - 0.5 * self.sigma_f ** 2) + self.sigma_f * Z)
        self.fundamental_price_history.append(self.fundamental_price)  # 记录新的基础价格

        if self.cyberbullying_model:
            self.cyberbullying_model.propagate()
            
            

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
            # 记录当前成交日志长度
            trade_log_length_before = len(self.orderbook.trade_log)
            # 提交订单
            self.orderbook.submit_order(order)
            # 检查是否有新的成交
            if len(self.orderbook.trade_log) > trade_log_length_before:
                # 只处理新产生的成交
                new_trades = self.orderbook.trade_log[trade_log_length_before:]
                self._process_trades(new_trades)
        else:
            print(f"❌ Agent {agent.trader_id} chose not to trade.")

    def _process_trades(self, trades):
        """处理成交信息并更新交易者资产"""
        for trade in trades:
            buyer = next((t for t in self.agents if t.trader_id == trade['buyer_id']), None)
            seller = next((t for t in self.agents if t.trader_id == trade['seller_id']), None)
            
            if buyer and seller:
                # 更新买家资产
                buyer.cash -= trade['trade_qty'] * trade['trade_price']
                buyer.stock += trade['trade_qty']
                # 强制非负
                buyer.cash = max(buyer.cash, 0)
                buyer.stock = max(buyer.stock, 0)
                
                # 更新卖家资产
                seller.cash += trade['trade_qty'] * trade['trade_price']
                seller.stock -= trade['trade_qty']
                # 强制非负
                seller.cash = max(seller.cash, 0)
                seller.stock = max(seller.stock, 0)
                
                print(f"💰 Asset update after trade:")
                print(f"  - Buyer {buyer.trader_id}: cash={buyer.cash:.2f}, stock={buyer.stock:.2f}")
                print(f"  - Seller {seller.trader_id}: cash={seller.cash:.2f}, stock={seller.stock:.2f}")

    

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
