import numpy as np
from orders import Order

class Agent:
    """异质性交易者，包含基本面、技术面和噪声策略"""
    def __init__(
        self,
        agent_id: int,
        sigma1: float,
        sigma2: float,
        sigman: float,
        fundamental_value: float
    ):
        self.agent_id = agent_id
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigman = sigman
        self.fundamental_value = fundamental_value
        self.stock = np.random.uniform(0, 50)  # 初始随机持仓
        self.cash = np.random.uniform(0, 10000)  # 初始随机现金
        self.g1, self.g2, self.n = self._generate_weights()
        self.tau = self._calculate_time_horizon()
        self.alpha = self._calculate_risk_aversion()
        self.price_memory = []  # 用于技术分析的价格历史

    def _generate_weights(self) -> tuple:
        """生成策略权重（指数分布）"""
        return (
            np.random.exponential(scale=self.sigma1),
            np.random.exponential(scale=self.sigma2),
            np.random.exponential(scale=self.sigman)
        )

    def _calculate_time_horizon(self) -> int:
        """计算时间跨度（式4）"""
        tau_bar = 200  # 参考时间跨度
        return int(np.ceil(tau_bar * (1 + self.g1) / (1 + self.g2)))

    def _calculate_risk_aversion(self) -> float:
        """计算风险厌恶系数（式6）"""
        alpha_bar = 0.1  # 参考风险厌恶
        return alpha_bar * (1 + self.g1) / (1 + self.g2)

    def generate_order(self, current_price: float, fundamental_value: float, timestamp: int) -> Order:
        """生成订单（限价单或市价单）"""
        self.price_memory.append(current_price)
        if len(self.price_memory) > self.tau:
            self.price_memory.pop(0)
        
        # 1. 计算预期价格（式1-3）
        expected_price = self._calculate_expected_price(current_price, fundamental_value)
        
        # 2. 确定买卖方向和价格
        direction = 'buy' if expected_price > current_price else 'sell'
        price = self._determine_price(expected_price, current_price)
        
        # 3. 计算最优持仓量（式8）
        optimal_holding = self._calculate_optimal_holding(current_price, expected_price)
        
        # 4. 生成订单
        if direction == 'buy':
            quantity = max(0, optimal_holding - self.stock)
        else:
            quantity = max(0, self.stock - optimal_holding)
        
        # 至少交易1手（A股规则）
        quantity = max(100, quantity - quantity % 100)  
        
        return Order(
            order_id=None,  # 由OrderBook分配
            agent_id=self.agent_id,
            price=price,
            quantity=quantity,
            direction=direction,
            timestamp=timestamp,
            is_market_order=False  # 默认为限价单
        )

    def _calculate_expected_price(self, current_price: float, fundamental_value: float) -> float:
        """计算预期价格（式1-3）"""
        # 基本面成分
        fundamental_component = (1 / self.tau) * np.log(fundamental_value / current_price)
        
        # 技术面成分（过去tau期的平均收益）
        if len(self.price_memory) >= 2:
            returns = np.diff(np.log(self.price_memory))
            chartist_component = np.mean(returns[-self.tau:]) if self.tau > 0 else 0
        else:
            chartist_component = 0
        
        # 噪声成分
        noise_component = np.random.normal(0, 0.01)
        
        # 综合预期（式1）
        weighted_return = (
            self.g1 * fundamental_component + 
            self.g2 * chartist_component + 
            self.n * noise_component
        ) / (self.g1 + self.g2 + self.n)
        
        # 预期价格（式3）
        return current_price * np.exp(weighted_return * self.tau)

    def _determine_price(self, expected_price: float, current_price):
        self.time += 1
        # 每个Agent提交订单
        for agent in self.agents:
            order = agent.submit_order(self.current_price, self.time)
            self.order_book.add_order(order)

        # 更新市场价格（简化：取最新成交价或中间价）
        best_bid, best_ask = self.order_book.get_best_bid_ask()
        if best_bid and best_ask:
            self.current_price = (best_bid + best_ask) / 2