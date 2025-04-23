import numpy as np
from abc import ABC, abstractmethod
from src.order.orders import Order, OrderDirection

class BaseTrader(ABC):
    def __init__(self, trader_id: int, config: dict, trader_type: str):
        self.trader_id = trader_id
        self.type = trader_type
        self.config = config

        self.cash = config[trader_type]["initial_cash"]
        self.stock = config[trader_type]["initial_stock"]

        # 策略权重参数
        self.g1 = np.random.exponential(scale=config[trader_type]["fundamental_sigma"])
        self.g2 = np.random.exponential(scale=config[trader_type]["chartist_sigma"])
        self.n = np.random.exponential(scale=config[trader_type]["noise_sigma"])

        # 风险厌恶 和 投资期
        self.alpha = config["reference_alpha"] * (1 + self.g1) / (1 + self.g2)
        self.tau = config["reference_tau"] / (1 + self.g1 / (1 + self.g2))

        # 网络暴力/情绪参数
        self.emotion_bias = np.random.normal(0, config.get("emotion_initial_std", 1.0))
        self.social_pressure = np.random.normal(0, 1.0)
        self.suppression = np.random.normal(0, 1.0)
        self.conformity_level = np.random.normal(0, 1.0)

        self.emotion_weight = config.get("emotion_weight", 0.0)
        self.pressure_weight = config.get("pressure_weight", 0.0)
        self.suppression_weight = config.get("suppression_weight", 0.0)
        self.conformity_weight = config.get("conformity_weight", 0.0)

    def apply_social_impact(self, shock_dict):
        for k, v in shock_dict.items():
            if hasattr(self, k):
                setattr(self, k, getattr(self, k) + v)

    def decide_to_trade(self):
        score = 1.0 - (self.social_pressure + self.suppression)
        prob_trade = max(0.0, min(1.0, score))
        return np.random.rand() < prob_trade

    def __repr__(self):
        return super().__repr__()
    
    @abstractmethod
    def generate_order(self, timestep: int, market_snapshot: dict) -> Order:
        pass
