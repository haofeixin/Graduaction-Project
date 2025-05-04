import numpy as np
from src.traders.base import BaseTrader
from src.order.orders import Order, OrderDirection, OrderType
from scipy.optimize import brentq

class InstitutionalTrader(BaseTrader):
    def generate_order(self, timestep: int, market_snapshot: dict) -> Order:
        
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
        epsilon = np.random.normal(0, self.config.get("noise_std", 0.01))
      
        tau_f = self.config.get("reference_tau_f", 200)

        expected_return = (
            (self.g1 / tau_f) * np.log(p_f / p_t) +
            self.g2 * trend +
            self.n * epsilon 
         
        ) / (self.g1 + self.g2 + self.n + 1e-6)

        expected_price = p_t * np.exp(expected_return * tau_i)

        # 计算价格偏离程度
        price_deviation = (expected_price - p_t) / p_t
        

        # 决定交易方向
        if abs(price_deviation) < 0.0005:  # 如果价格偏离小于0.05%，不交易
            print(f"❌ Agent {self.trader_id} chose not to trade: Price deviation too small")
            return None

        direction = OrderDirection.BUY if price_deviation > 0 else OrderDirection.SELL

        # 计算交易数量
        # 数量与价格偏离程度和风险偏好相关
        if direction == OrderDirection.BUY:
            base_quantity = int(self.cash  * 0.2 / p_t )  # 买：按现金的 20% 推出买量
        else:
            base_quantity = int(self.stock * 0.1)        # 卖：当前持仓的 10%
            
        
        quantity = min(base_quantity, int(self.cash / p_t))  # 确保不超过可用资金

        if quantity < 1:  # 如果计算出的数量小于1，不交易
            print(f"❌ Agent {self.trader_id} chose not to trade: Calculated quantity too small")
            return None

        # 决定订单类型和价格
        order_type = OrderType.LIMIT
        price = None

        if direction == OrderDirection.BUY:
            if best_ask is not None and best_ask <= expected_price:
                # 如果最优卖价低于预期价格，使用市价单
                order_type = OrderType.MARKET
                price = best_ask
            else:
                # 否则使用限价单，价格略低于预期价格
                price = expected_price * (1 - abs(price_deviation) * 0.5)
        else:  # SELL
            if best_bid is not None and best_bid >= expected_price:
                # 如果最优买价高于预期价格，使用市价单
                order_type = OrderType.MARKET
                price = best_bid
            else:
                # 否则使用限价单，价格略高于预期价格
                price = expected_price * (1 + abs(price_deviation) * 0.5)

        # 确保价格合理
        price = max(0.01, min(price, expected_price * 2))
        price = round(price, 2)

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
