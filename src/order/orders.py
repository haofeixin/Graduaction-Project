from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, ClassVar


class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'


class OrderDirection(Enum):
    BUY = 'buy'
    SELL = 'sell'


class OrderStatus(Enum):
    PENDING = 'pending'
    EXECUTED = 'executed'
    CANCELLED = 'cancelled'


@dataclass
class Order:
    _next_id: ClassVar[int] = 0  # 全局订单编号，用于时间排序

    trader_id: int
    max_wait_time: int
    quantity: int
    timestep: int
    direction: OrderDirection
    order_type: Optional[OrderType] = None
    price: Optional[float] = None
    order_id: int = field(init=False)
    status: OrderStatus = field(default=OrderStatus.PENDING)
    executed_price: Optional[float] = field(default=None, init=False)
    executed_timestep: Optional[int] = field(default=None, init=False)
    cancelled_timestep: Optional[int] = field(default=None, init=False)
    

    def __post_init__(self):
        self.order_id = Order._next_id
        Order._next_id += 1

    def execute(self, executed_price: float, executed_timestep: int, fill_qty: int):
        self.executed_price = executed_price
        self.executed_timestep = executed_timestep
        self.quantity -= fill_qty

        # 如果没剩余了，才标记为完全成交
        if self.quantity <= 0:
            self.status = OrderStatus.EXECUTED


    def check_timeout(self, current_timestep: int):
        """检查订单是否超时"""
        if current_timestep - self.timestep > self.max_wait_time:
            self.status = OrderStatus.CANCELLED
            return True
        return False

    def is_executable(self, best_bid: float, best_ask: float) -> bool:
        if self.order_type == OrderType.MARKET:
            return True
        if self.price is None:
            return False  # 明确检查 None 情况
        if self.direction == OrderDirection.BUY:
            return self.price >= best_ask
        if self.direction == OrderDirection.SELL:
            return self.price <= best_bid
        return False
    
    def __repr__(self):
        price_str = f"{self.price:.2f}" if self.price is not None else "MKT"
        return (f"<Order {self.order_id} | trader={self.trader_id} | "
                f"type={self.order_type.name if self.order_type else 'None'} | "
                f"dir={self.direction.name} | qty={self.quantity} | "
                f"price={price_str} | status={self.status.name} | step={self.timestep}>")

        
