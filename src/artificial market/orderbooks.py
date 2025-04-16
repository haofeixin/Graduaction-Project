import bisect
import numpy as np
from typing import List, Dict, Optional, Tuple
from orders import Order

class OrderBook:
    """订单簿类，实现Chiarella et al. (2009)的订单驱动市场逻辑
    
    核心功能：
    1. 处理限价单和市价单
    2. 按卖方报价成交（价格优先>时间优先）
    3. 动态更新市场价格（最后成交价或中点价）
    """
    def __init__(self, tick_size: float = 0.01, initial_fundamental: float = 300.0):
        """
        :param tick_size: 最小价格变动单位（默认0.01元）
        :param initial_fundamental: 初始基本面价值（默认300）
        """
        self.tick_size = tick_size
        self.buy_orders = []  # 买单列表，按价格降序、时间升序排列
        self.sell_orders = []  # 卖单列表，按价格升序、时间升序排列
        self.last_trade_price = initial_fundamental  # 最后成交价（初始为基本面价值）
        self.fundamental_value = initial_fundamental  # 基本面价值（几何布朗运动）
        self.trade_history = []  # 历史成交记录

    def add_order(self, order: 'Order') -> List[Dict]:
        """处理新订单并返回成交记录
        
        :param order: 订单对象（限价单或市价单）
        :return: 生成的成交记录列表
        """
        if order.is_market_order:
            trades = self._execute_market_order(order)
        else:
            trades = self._process_limit_order(order)
        
        self._update_fundamental()  # 更新基本面价值
        self._update_market_price(trades)
        return trades

    def _process_limit_order(self, order: 'Order') -> List[Dict]:
        """处理限价单（可能触发即时成交）"""
        # Step 1: 插入订单到订单簿
        self._insert_order(order)
        
        # Step 2: 尝试撮合订单
        trades = self._match_orders()
        
        # Step 3: 处理部分成交后的剩余订单
        if order.quantity > 0 and not order.is_market_order:
            self._insert_remaining_order(order)
            
        return trades

    def _execute_market_order(self, order: 'Order') -> List[Dict]:
        """执行市价单（立即以对手方最优价成交）"""
        trades = []
        remaining_quantity = order.quantity
        
        while remaining_quantity > 0:
            best_opposite = self._get_best_opposite_price(order.direction)
            if best_opposite is None:
                break  # 无对手方订单
            
            # 以对手方最优价成交
            trade_price = best_opposite.price
            trade_qty = min(remaining_quantity, best_opposite.quantity)
            
            trades.append({
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
            
            remaining_quantity -= trade_qty
        
        return trades

    def _insert_order(self, order: 'Order'):
        """将订单插入到订单簿正确位置（价格优先>时间优先）"""
        if order.direction == 'buy':
            bisect.insort_right(
                self.buy_orders, 
                order, 
                key=lambda x: (-x.price, x.timestamp)  # 买单按价格降序
            )
        else:
            bisect.insort_right(
                self.sell_orders, 
                order, 
                key=lambda x: (x.price, x.timestamp)  # 卖单按价格升序
            )

    def _match_orders(self) -> List[Dict]:
        """撮合可匹配的限价单（按卖方报价成交）"""
        trades = []
        while self._has_valid_match():
            best_buy = self.buy_orders[0]
            best_sell = self.sell_orders[0]
            
            # 确定成交价（卖方报价优先）
            trade_price = best_sell.price
            
            # 计算成交量
            trade_qty = min(best_buy.quantity, best_sell.quantity)
            
            # 记录成交
            trades.append({
                'price': trade_price,
                'quantity': trade_qty,
                'buy_order_id': best_buy.order_id,
                'sell_order_id': best_sell.order_id,
                'timestamp': max(best_buy.timestamp, best_sell.timestamp)
            })
            
            # 更新订单簿
            best_buy.quantity -= trade_qty
            best_sell.quantity -= trade_qty
            if best_buy.quantity == 0:
                self.buy_orders.pop(0)
            if best_sell.quantity == 0:
                self.sell_orders.pop(0)
        
        return trades

    def _has_valid_match(self) -> bool:
        """检查是否存在可匹配的订单（买价≥卖价）"""
        return (len(self.buy_orders) > 0 and 
                len(self.sell_orders) > 0 and 
                self.buy_orders[0].price >= self.sell_orders[0].price)

    def _update_fundamental(self):
        """更新基本面价值（几何布朗运动，σ_f=0.001）"""
        self.fundamental_value *= np.exp(np.random.normal(0, 0.001))

    def _update_market_price(self, trades: List[Dict]):
        """更新市场价格（最后成交价或中点价）"""
        if trades:
            self.last_trade_price = trades[-1]['price']
        else:
            best_bid = self.buy_orders[0].price if self.buy_orders else None
            best_ask = self.sell_orders[0].price if self.sell_orders else None
            if best_bid and best_ask:
                self.last_trade_price = (best_bid + best_ask) / 2
        
        # 记录历史价格
        self.trade_history.append({
            'time': len(self.trade_history),
            'price': self.last_trade_price,
            'best_bid': best_bid if 'best_bid' in locals() else None,
            'best_ask': best_ask if 'best_ask' in locals() else None
        })

    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """获取当前最优买卖报价"""
        best_bid = self.buy_orders[0].price if self.buy_orders else None
        best_ask = self.sell_orders[0].price if self.sell_orders else None
        return best_bid, best_ask

    def get_market_price(self) -> float:
        """获取当前市场价格（最后成交价或中点价）"""
        return self.last_trade_price

    def _get_best_opposite_price(self, direction: str) -> Optional['Order']:
        """获取对手方最优价格订单"""
        if direction == 'buy':
            return self.sell_orders[0] if self.sell_orders else None
        else:
            return self.buy_orders[0] if self.buy_orders else None

    def _remove_order(self, order: 'Order'):
        """从订单簿移除指定订单"""
        if order.direction == 'buy':
            self.buy_orders.remove(order)
        else:
            self.sell_orders.remove(order)

    def _insert_remaining_order(self, order: 'Order'):
        """插入未完全成交的剩余订单"""
        if order.quantity > 0:
            self._insert_order(order)