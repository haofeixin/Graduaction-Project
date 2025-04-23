import numpy as np
from scipy.optimize import brentq
from src.traders.base import BaseTrader
from src.order.orders import Order, OrderDirection

class InstitutionalTrader(BaseTrader):
    def generate_order(self, timestep: int, market_snapshot: dict) -> Order:
        if not self.decide_to_trade():
            return None

        p_t = market_snapshot.get("last_price", 10.0)
        p_f = market_snapshot.get("fundamental_price", 10.0)
        returns = market_snapshot.get("log_returns", [])
        tau_i = int(self.tau)
        recent_returns = returns[-tau_i:] if len(returns) >= tau_i else returns

        trend = np.mean(recent_returns) if recent_returns else 0.0
        volatility = np.std(recent_returns) if recent_returns else 0.02

        epsilon = np.random.normal(0, self.config.get("noise_std", 0.01))
        bias = self.emotion_weight * self.emotion_bias
        tau_f = self.config.get("reference_tau_f", 30)

        # 预期收益率（Eq. 1）
        expected_return = (
            (self.g1 / tau_f) * np.log(p_f / p_t) +
            self.g2 * trend +
            self.n * epsilon +
            bias
        ) / (self.g1 + self.g2 + self.n + 1e-6)

        # 预期未来价格（Eq. 3）
        expected_price = p_t * np.exp(expected_return * self.tau)

        # pi(p) 函数（Eq. 8）
        def pi(p):
            return np.log(expected_price / p) / (self.alpha * volatility * p)

        # 求 p*（Eq. 10）：pi(p*) = stock
        try:
            p_star = brentq(lambda p: pi(p) - self.stock, 0.01, expected_price * 2)
        except ValueError:
            p_star = p_t

        # 求 p_m（Eq. 11）：p * (pi(p) - stock) = cash
        try:
            p_m = brentq(lambda p: p * (pi(p) - self.stock) - self.cash, 0.01, expected_price)
        except ValueError:
            p_m = 0.01

        p_M = expected_price
        order_price = np.random.uniform(p_m, p_M)
        pi_order = pi(order_price)
        delta_position = pi_order - self.stock

        if abs(delta_position) < 1e-2:
            return None

        direction = OrderDirection.BUY if delta_position > 0 else OrderDirection.SELL
        quantity = int(abs(delta_position))
        price = round(order_price, 2)

        return Order(
            trader_id=self.trader_id,
            direction=direction,
            order_type=None,
            quantity=quantity,
            price=price,
            timestep=timestep,
            max_wait_time=tau_i
        )
