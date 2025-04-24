import heapq
from typing import List, Optional, Tuple
from .orders import Order, OrderType, OrderDirection, OrderStatus


class OrderBook:
    def __init__(self):
        self.buys: List[Tuple[float, int, Order]] = []
        self.sells: List[Tuple[float, int, Order]] = []
        self.trade_log = []

    def submit_order(self, order: Order):
        print(f"📥 OrderBook received: {order}")
        if order.order_type == OrderType.MARKET:
            self._match_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            self._submit_limit_order(order)

    def _submit_limit_order(self, order: Order):
        entry = (self._priority(order), order.order_id, order)
        if order.direction == OrderDirection.BUY:
            heapq.heappush(self.buys, entry)
        else:
            heapq.heappush(self.sells, entry)

    def _match_market_order(self, order: Order):
        book = self.sells if order.direction == OrderDirection.BUY else self.buys
        remaining_quantity = order.quantity  # 剩余的市价单数量

        # 获取当前市场最优价格
        if order.direction == OrderDirection.BUY:
            trade_price = self.best_ask()  # 买单需要匹配卖单的最优价格
        elif order.direction == OrderDirection.SELL:
            trade_price = self.best_bid()  # 卖单需要匹配买单的最优价格

        if trade_price is None:
            print("No matching market price found, cannot execute the order")
            return
        
        _, _, top_order = book[0]
        # 遍历订单簿，成交所有符合条件的订单
        while remaining_quantity > 0 and book:
            _, _, top_order = heapq.heappop(book)

            # 只有当 top_order 的价格符合当前市价单的价格时才成交
            # 买单，top_order 的卖价 <= 当前买单的买价； 卖单，top_order 的买价 >= 当前卖单的卖价
            if (order.direction == OrderDirection.BUY and top_order.price <= order.price) or \
            (order.direction == OrderDirection.SELL and top_order.price >= order.price):
                trade_qty = min(remaining_quantity, top_order.quantity)
                trade_price = top_order.price if top_order.price else 0.0

                # 执行成交
                order.execute(trade_price, order.timestep, trade_qty)
                top_order.execute(trade_price, order.timestep, trade_qty)
                self.trade_log.append({
                    'buyer_id': order.order_id if order.direction == OrderDirection.BUY else top_order.order_id, 
                    'seller_id': top_order.order_id if order.direction == OrderDirection.BUY else order.order_id, 
                    'trade_timestamp': order.timestep,
                    'trade_price': trade_price, 
                    'trade_qty': trade_qty
                })
                print(f"💥 TRADE: {order.order_id} <-> {top_order.order_id} at price={trade_price} qty={trade_qty}")
                remaining_quantity -= trade_qty  # 更新剩余未成交部分的数量


                # 如果卖单部分成交，说明order成交完全,剩余部分继续在订单簿中
                if top_order.quantity > 0:
                    heapq.heappush(book, (self._priority(top_order), top_order.order_id, top_order))
            else:
                break
        
        # 只有当市价单仍有剩余量时，才将剩余部分转为限价单
        if remaining_quantity > 0:
            limit_order = Order(
                trader_id=order.trader_id,
                order_type=OrderType.LIMIT,
                direction=order.direction,
                quantity=remaining_quantity,
                timestep=order.timestep,
                price=order.price,  # 使用市价单的价格作为限价单价格
                max_wait_time=order.max_wait_time
            )
            self.submit_order(limit_order)
            print(f"Remaining part of the market order converted to limit order at price {order.price}")


    def _priority(self, order: Order) -> float:
        if order.price is None:
            raise ValueError("Limit orders must have price")
        return -order.price if order.direction == OrderDirection.BUY else order.price

    def best_bid(self) -> Optional[float]:
        while self.buys:
            price, _, order = self.buys[0]
            if order.status == OrderStatus.PENDING and order.quantity > 0:
                return -price
            else:
                heapq.heappop(self.buys)
        return None

    def best_ask(self) -> Optional[float]:
        while self.sells:
            price, _, order = self.sells[0]
            if order.status == OrderStatus.PENDING and order.quantity > 0:
                return price
            else:
                heapq.heappop(self.sells)
        return None

    def snapshot(self):
        return {
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask(),
            "buy_depth": len(self.buys),
            "sell_depth": len(self.sells)
        }

    def reset(self):
        self.buys.clear()
        self.sells.clear()
        self.trade_log.clear()

    def cancel_timed_out_orders(self, current_timestamp):
        """检查并取消所有超时未成交的订单"""
        for entry in self.buys[:]:
            _, _, order = entry
            if order.check_timeout(current_timestamp):  # 使用时间步来检查超时
                self.buys.remove(entry)  # 从买单簿中移除超时订单
                print(f"Order {order.order_id} has been cancelled due to timeout.")

        for entry in self.sells[:]:
            _, _, order = entry
            if order.check_timeout(current_timestamp):  # 使用时间步来检查超时
                self.sells.remove(entry)  # 从卖单簿中移除超时订单
                print(f"Order {order.order_id} has been cancelled due to timeout.")
        
