import numpy as np
from abc import ABC, abstractmethod
from src.order.orders import Order

class BaseTrader(ABC):
    def __init__(self, trader_id: int, config: dict, trader_type: str):
        self.trader_id = trader_id
        self.type = trader_type
        self.config = config
        fundarmental_price = config['fundamental_price']
        initial_stock_max = config[trader_type]["initial_stock_max"]
        self.stock = int(np.random.uniform(0.1 * initial_stock_max, initial_stock_max))
        self.cash = fundarmental_price * self.stock

        # 策略权重参数
        self.g1 = np.random.exponential(scale=config[trader_type]["fundamental_sigma"])
        self.g2 = np.random.exponential(scale=config[trader_type]["chartist_sigma"])
        self.n = np.random.exponential(scale=config[trader_type]["noise_sigma"])

        # 风险厌恶 和 投资期
        
        self.tau = config["reference_tau_f"] / (1 + self.g1 / (1 + self.g2))

        # 网络暴力/情绪参数
        self.emotion_bias = np.random.normal(0, config.get("emotion_initial_bias", 0.01))
        self.suppression = np.random.normal(0, 1.0)
        self.emotion_weight = np.random.exponential(scale=config.get("emotion_weight_sigma", 0.5))

        self.is_bullied = False            # 当前是否被网暴
        self.exposure = 0.0                # 累积攻击强度
        self.resilience = 0.0              # 抗压性（成长型）
        self.bully_cooldown = 0            # 喷子攻击后冷却时间
        self.is_natural_bully = False      # 是否天生喷子
        self.neighbors = []                # trader 社交图中的邻居


    def __repr__(self):
        return (f"<Trader {self.trader_id} ({self.type}) | "
                f"cash={self.cash:.1f}, stock={self.stock}, "
                f"g1={self.g1:.2f}, g2={self.g2:.2f}, n={self.n:.2f}, "
                f"alpha={self.alpha:.2f}, tau={self.tau:.2f}>")
        
    @abstractmethod
    def generate_order(self, timestep: int, market_snapshot: dict) -> Order:
        pass
