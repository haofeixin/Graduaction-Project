
class Order:
    def __init__(
        self,
        order_id: int,
        agent_id: int,
        price: float,
        quantity: float,
        direction: str,
        timestamp: int,
        tick_size: float = 0.01,      # 默认A股规则
        lot_size: int = 100,          # 默认1手=100股
        price_limit: float = None,
    ):
        self.order_id = order_id
        self.agent_id = agent_id
        self.price = price
        self.quantity = quantity
        self.direction = direction.lower()
        self.timestamp = timestamp
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.price_limit = price_limit
        self._validate_order()

    def _validate_order(self):
        """检查订单参数的合法性（市场规则）"""
        # 方向校验
        assert self.direction in ['buy', 'sell'], "Direction must be 'buy' or 'sell'"
        
        # 数量校验
        assert self.quantity > 0, "Quantity must be positive"
        assert self.quantity % self.lot_size == 0, \
            f"Quantity must be a multiple of {self.lot_size} (1 lot)"
        
        # 价格校验
        assert self.price > 0, "Price must be positive"
        assert round(self.price / self.tick_size) == self.price / self.tick_size, \
            f"Price must be a multiple of {self.tick_size}"
        
        if self.price_limit is not None:
            assert abs(self.price - self.price_limit) < 1e-6, \
                f"Price exceeds limit ({self.price_limit})"

    def __repr__(self):
        return f"Order(id={self.order_id}, agent={self.agent_id}, {self.direction} {self.quantity}@{self.price}, t={self.timestamp})"
    

    