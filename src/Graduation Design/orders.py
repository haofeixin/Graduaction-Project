from typing import List, Dict, Optional, Tuple

class Order:
    def __init__(self, direction, quantity, price, timestamp, agent_id, order_id):
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.agent_id = agent_id
        self.order_id = order_id
        self.timestamp = timestamp
        self._validate()

    def _validate(self):
        assert self.direction in ['buy', 'sell'], '订单方向无效！必须为buy或者sell'
        assert self.quantity > 0, '订单交易数量无效，必须为正数'
        assert self.price > 0, '订单交易价格无效, 必须为正数'

class OrderBook:
    def __init__(self, tick_size: float = 0.01, initial_price: float = 300.0):
        """
        Paras:

        - tick_size: float 最小价格变动单位
        - initial_price: float 初始价格
        """
        self.tick_size = tick_size
        self.buy_orders = []
        self.sell_orders = []
        self.last_trade_price = initial_price
        self.trade_history = []
    
    def add_order(self, order: Order):
        # 处理新订单
        immediate_trade = self._execute_immediate(order)
        if immediate_trade:
            # 可作为市价单成交
            self.trade_history.append(immediate_trade)
        
        else:
            # 作为限价单插入订单簿
            self._insert_order(order)
        return
    
    def _execute_immediate(self, order: Order):
        best_opposite = self._get_best_opposite(order.direction)
        if best_opposite and (
            (order.direction == 'buy' and order.price >= best_opposite.price) or
            (order.direction == 'sell' and order.price <= best_opposite.price)
        ):
            # 以对手方价格成交
            trade_price = best_opposite.price
            trade_qty = min(order.quantity, best_opposite.quantity)
            
            self.trade_history.append({
                'price': trade_price,
                'quantity': trade_qty,
                'buy_order_id': order.order_id if order.direction == 'buy' else best_opposite.order_id,
                'sell_order_id': order.order_id if order.direction == 'sell' else best_opposite.order_id,
                'timestamp': order.timestamp
            })
            # 更新订单簿
            best_opposite.quantity -= trade_qty
            if best_opposite.quantity == 0:
                self._remove_order(best_opposite)

    def _get_best_opposite(self, direction) -> Optional[Order]:
        # 获取对手最优报价订单
        if direction == 'buy':
            return self.sell_orders[0] if self.sell_orders else None
        else:
            return self.buy_orders[0] if self.buy_orders else None
    
    def _remove_order(self, order: Order):
        # 从订单簿中移除订单
        if order.direction == 'buy':
            self.buy_orders.remove(order)
        else:
            self.sell_orders.remove(order)




