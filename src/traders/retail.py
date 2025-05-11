import numpy as np
from src.traders.base import BaseTrader
from src.order.orders import Order, OrderDirection, OrderType

class RetailTrader(BaseTrader):
    def generate_order(self, timestep: int, market_snapshot: dict) -> Order:
        if self.is_bullied and np.random.rand() < self.suppression:
            # print(f"🛑 Trader {self.trader_id} is bullied and chooses to stay silent.")
            return None
                
        # 获取市场信息
        p_t = market_snapshot.get("last_price", 300.0)
        p_f = market_snapshot.get("fundamental_price", 300.0)
        best_ask = market_snapshot.get("best_ask", None)
        best_bid = market_snapshot.get("best_bid", None)
        returns = market_snapshot.get("log_returns", [])

        # 计算预期价格
        tau_i = int(self.tau)
        recent_returns = returns[-tau_i:] if len(returns) >= tau_i else returns
        trend = np.mean(recent_returns) if recent_returns else 0.0
        
        # 计算预期收益率
        noise_std = self.config.get("noise_std", 0.01)
        if self.is_bullied:
            noise_std *= self.config.get("bully_noise_amplify", 3.0)
        epsilon = np.random.normal(0, noise_std)
        bias = self.emotion_weight * self.emotion_bias
        tau_f = self.config.get("reference_tau_f", 200)

        expected_return = (
            (self.g1 / tau_f) * np.log(p_f / p_t) +
            self.g2 * trend +
            self.n * epsilon +
            bias
        ) / (self.g1 + self.g2 + self.n + 1e-6)

        expected_price = p_t * np.exp(expected_return * tau_i)

        # 计算价格偏离程度
        price_deviation = (expected_price - p_t) / p_t

        # 决定交易方向
        if abs(price_deviation) < 0.0005:  # 如果价格偏离小于0.05%，不交易
            # print(f"❌ Agent {self.trader_id} chose not to trade: Price deviation too small")
            return None

        direction = OrderDirection.BUY if price_deviation > 0 else OrderDirection.SELL

        # 决定订单类型和价格（先定价）
        order_type = OrderType.LIMIT
        price = None
        if direction == OrderDirection.BUY:
            if best_ask is not None and best_ask <= expected_price:
                order_type = OrderType.MARKET
                price = best_ask
            else:
                price = expected_price * (1 - abs(price_deviation) * 0.5)
        else:
            if best_bid is not None and best_bid >= expected_price:
                order_type = OrderType.MARKET
                price = best_bid
            else:
                price = expected_price * (1 + abs(price_deviation) * 0.5)
        # 价格锚定：限制在p_t的±10%区间
        lower_bound = p_t * 0.9
        upper_bound = p_t * 1.1
        price = min(max(price, lower_bound), upper_bound)
        price = round(price, 2)

        # 再根据锚定后的价格与p_t的偏离决定交易量
        price_diff = abs(price - p_t) / p_t
        if direction == OrderDirection.BUY:
            base_quantity = int(self.cash * 0.2 / p_t)
        else:
            base_quantity = int(self.stock * 0.2)
        risk_adjusted_quantity = int(base_quantity * (1 + price_diff * 2))
        if self.is_bullied:
            risk_adjusted_quantity = int(risk_adjusted_quantity * 1.2)
        quantity = min(risk_adjusted_quantity, int(self.cash / p_t))
        if quantity < 1:
            # print(f"❌ Agent {self.trader_id} chose not to trade: Calculated quantity too small")
            return None

        # 打印决策过程
        # print(f"\n🎯 {self.type} Agent {self.trader_id} decision process:")
        # print(f"  - Current price: {p_t:.2f}")
        # print(f"  - Fundamental price: {p_f:.2f}")
        # print(f"  - Expected price: {expected_price:.2f}")
        # print(f"  - Price deviation: {price_deviation:.4f}")
        # print(f"  - Best ask: {best_ask if best_ask else 'N/A'}")
        # print(f"  - Best bid: {best_bid if best_bid else 'N/A'}")
        # print(f"  - Parameters:")
        # print(f"    * g1: {self.g1:.2f}")
        # print(f"    * g2: {self.g2:.2f}")
        # print(f"    * n: {self.n:.2f}")
        # print(f"    * tau_i: {tau_i}")
        # print(f"    * alpha: {self.alpha:.2f}")
        # print(f"    * stock: {self.stock:.2f}")
        # print(f"    * cash: {self.cash:.2f}")
        # print(f"    * emotion_bias: {self.emotion_bias:.2f}")
        # print(f"    * suppression: {self.suppression:.2f}")

        # print(f"✅ Agent {self.trader_id} decided to trade:")
        # print(f"  - Direction: {'BUY' if direction == OrderDirection.BUY else 'SELL'}")
        # print(f"  - Quantity: {quantity}")
        # print(f"  - Price: {price:.2f}")
        # print(f"  - Order type: {order_type.name}")

        return Order(
            trader_id=self.trader_id,
            direction=direction,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestep=timestep,
            max_wait_time=tau_i
        )
