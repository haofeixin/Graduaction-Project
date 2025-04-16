from agents import Agent
from orders import OrderBook, Order
import numpy as np
from typing import List, Dict


class Market:
    """人工股票市场模拟器，基于Chiarella et al. (2009)的ABM框架"""
    def __init__(
        self,
        num_agents: int = 100,
        tick_size: float = 0.01,
        fundamental_value: float = 100.0,
        sigma1: float = 10.0,  # 基本面权重分布参数
        sigma2: float = 1.0,   # 技术面权重分布参数
        sigman: float = 1.0    # 噪声权重分布参数
    ):
        self.order_book = OrderBook(tick_size)
        self.fundamental_value = fundamental_value
        self.current_price = fundamental_value
        self.time = 0
        self.agents = [self._create_agent(i, sigma1, sigma2, sigman) for i in range(num_agents)]
        self.price_history = [fundamental_value]

    def _create_agent(self, agent_id: int, sigma1: float, sigma2: float, sigman: float) -> 'Agent':
        """生成异质性Agent"""
        return Agent(
            agent_id=agent_id,
            sigma1=sigma1,
            sigma2=sigma2,
            sigman=sigman,
            fundamental_value=self.fundamental_value
        )

    def step(self) -> Dict:
        """运行一个模拟步，返回市场状态"""
        self.time += 1
        
        # 1. 更新基本面价值（随机游走）
        self._update_fundamental()
        
        # 2. 每个Agent提交订单
        for agent in self.agents:
            order = agent.generate_order(
                current_price=self.current_price,
                fundamental_value=self.fundamental_value,
                timestamp=self.time
            )
            self.order_book.add_order(order)
        
        # 3. 更新市场价格
        self._update_price()
        
        # 4. 记录数据
        self.price_history.append(self.current_price)
        
        return self._get_market_state()

    def _update_fundamental(self):
        """基本面价值随机游走（几何布朗运动）"""
        self.fundamental_value *= np.exp(np.random.normal(0, 0.001))  # 微小波动

    def _update_price(self):
        """根据订单簿更新市场价格"""
        if self.order_book.last_trade_price is not None:
            self.current_price = self.order_book.last_trade_price
        else:
            best_bid, best_ask = self.order_book.get_best_bid_ask()
            if best_bid and best_ask:
                self.current_price = (best_bid + best_ask) / 2

    def _get_market_state(self) -> Dict:
        """返回当前市场状态快照"""
        best_bid, best_ask = self.order_book.get_best_bid_ask()
        return {
            'time': self.time,
            'price': self.current_price,
            'fundamental': self.fundamental_value,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': best_ask - best_bid if (best_bid and best_ask) else None
        }


