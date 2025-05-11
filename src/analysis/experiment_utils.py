from src.traders.retail import RetailTrader
from src.traders.institutional import InstitutionalTrader
from src.market.structure import Market
from src.order.orderbooks import OrderBook
from src.social.cyberbullying import CyberbullyingModel
from src.config.loader import load_config

def run_scenario(enable_cyberbullying: bool, config_path="config.yaml", seed=None, max_timesteps=None):
    config = load_config(config_path)
    if max_timesteps is not None:
        config["market"]["max_timesteps"] = max_timesteps
    if seed is not None:
        import numpy as np
        np.random.seed(seed)
    agents = []
    n_retail = int(config["agents"]["count"] * config["agents"]["retail_ratio"])
    n_institutional = config["agents"]["count"] - n_retail
    for i in range(n_retail):
        agents.append(RetailTrader(i, config["agents"], "retail"))
    for i in range(n_institutional):
        agents.append(InstitutionalTrader(n_retail + i, config["agents"], "institutional"))
    orderbook = OrderBook()
    cyberbullying_model = None
    if enable_cyberbullying:
        cyberbullying_model = CyberbullyingModel(config, agents)
        cyberbullying_model.build_network()
    market = Market(orderbook, config["market"]["max_timesteps"], config, cyberbullying_model)
    for agent in agents:
        market.register_agent(agent)
    initial_agents = [agent.copy() for agent in agents]
    initial_price = market.fundamental_price
    market.run()
    final_price = market.price_history[-1] if market.price_history else market.fundamental_price
    return initial_agents, agents, initial_price, final_price, market 