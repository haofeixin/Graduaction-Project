from src.order.orderbooks import OrderBook
from src.traders.retail import RetailTrader
from src.traders.institutional import InstitutionalTrader
from src.market.structure import Market
from src.config.loader import load_config

import warnings
warnings.filterwarnings('ignore')

def create_agents(config):
    agents = []
    total = config["agents"]["count"]
    retail_count = int(total * config["agents"]["types"]["retail_ratio"])
    institutional_count = total - retail_count

    for i in range(retail_count):
        agents.append(RetailTrader(trader_id=i, config=config["agents"], trader_type="retail"))

    for i in range(institutional_count):
        agents.append(InstitutionalTrader(trader_id=retail_count + i, config=config["agents"], trader_type="institutional"))

    return agents

def run():
    config = load_config("config.yaml")
    orderbook = OrderBook()
    market = Market(orderbook=orderbook, max_timesteps=config["market"]["max_timesteps"], config=config)

    agents = create_agents(config)
    
    for agent in agents:
        market.register_agent(agent)
    
    market.run()

    print(f"Simulation complete. Trade log contains {len(orderbook.trade_log)} trades.")
    for trade in orderbook.trade_log[:10]:
        print(trade)

if __name__ == "__main__":
    
    run()

