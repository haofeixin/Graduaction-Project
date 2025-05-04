from src.order.orderbooks import OrderBook
from src.traders.retail import RetailTrader
from src.traders.institutional import InstitutionalTrader
from src.market.structure import Market
from src.config.loader import load_config
from src.analysis import WealthAnalyzer, MarketAnalyzer
import numpy as np
from src.social.cyberbullying import CyberbullyingModel
import copy
import pandas as pd
import warnings
from datetime import datetime   
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

def show_trade_records(orderbook):
    """显示所有交易记录"""
    print("\n=== Trade Records ===")
    print("=" * 100)
    print(f"{'Time':<8} {'Buyer':<8} {'Seller':<8} {'Price':<10} {'Quantity':<10} {'Total Value':<12}")
    print("-" * 100)
    
    total_trades = len(orderbook.trade_log)
    total_volume = 0
    total_value = 0
    
    for trade in orderbook.trade_log:
        trade_value = trade['trade_price'] * trade['trade_qty']
        total_volume += trade['trade_qty']
        total_value += trade_value
        
        print(f"{trade['trade_timestamp']:<8} "
              f"{trade['buyer_id']:<8} "
              f"{trade['seller_id']:<8} "
              f"{trade['trade_price']:<10.2f} "
              f"{trade['trade_qty']:<10} "
              f"{trade_value:<12.2f}")
    
    print("=" * 100)
    print(f"Total Trades: {total_trades}")
    print(f"Total Volume: {total_volume}")
    print(f"Total Value: {total_value:.2f}")
    print(f"Average Price: {total_value/total_volume if total_volume > 0 else 0:.2f}")
    print("=" * 100)

def run():
    config = load_config("config.yaml")
    
    # 设置随机种子
    random_seed = config["simulation"]["random_seed"]
    np.random.seed(random_seed)
    print(f"\n🔧 Using random seed: {random_seed}")

    agents = create_agents(config)
    initial_agents = copy.deepcopy(agents)


    orderbook = OrderBook()

    cyber_model = CyberbullyingModel(config, agents)
    cyber_model.build_network()

    market = Market(orderbook=orderbook, max_timesteps=config["market"]["max_timesteps"], config=config, cyberbullying_model=cyber_model)

    for agent in agents:
        market.register_agent(agent)
    
    # 创建分析器
    wealth_analyzer = WealthAnalyzer(agents, config["market"]["fundamental_price"])
    market_analyzer = MarketAnalyzer(market.price_history, market.log_returns, market.fundamental_price_history, cyber_model)
    
    # 获取初始市场价格
    if market.price_history:
        initial_price = market.price_history[0]
    else:
        initial_price = market.fundamental_price
    
    # 运行模拟
    market.run()
    
    # 获取最终市场价格
    if market.price_history:
        final_price = market.price_history[-1]
    else:
        final_price = market.fundamental_price
    
    # 财富分布对比图
    wealth_analyzer.plot_analysis(initial_agents, agents, initial_price, final_price)
    
    
    
    # 计算并展示财富不平等指标
    inequality = wealth_analyzer.calculate_wealth_inequality()
    print("\n=== Wealth Inequality Metrics ===")
    print(f"Gini Coefficient: {inequality['gini_coefficient']:.4f}")
    print(f"Wealth Concentration (Top 20%): {inequality['wealth_concentration']:.2%}")
    
    # 展示市场分析结果
    # print("\n=== Market Analysis ===")
    # market_metrics = market_analyzer.calculate_market_metrics()
    # print(f"Annualized Volatility: {market_metrics['volatility']:.2%}")
    # print(f"Average Price Deviation: {market_metrics['price_deviation']:.2%}")
    # print(f"Returns Autocorrelation: {market_metrics['autocorrelation']:.4f}")
    
    # 绘制市场图表
    market_analyzer.plot_price_evolution("price_evolution.png")
    
    # market_analyzer.plot_volatility_evolution("结果/volatility_evolution.png")
    
    # 显示所有交易记录
    # show_trade_records(orderbook)

    log_experiment(config, initial_agents, agents, initial_price, final_price, wealth_analyzer)

def log_experiment(config, initial_agents, final_agents, initial_price, final_price, wealth_analyzer, save_path='results/experiment_log.xlsx'):
    # 参数
    market_cfg = config['market']
    agent_cfg = config['agents']
    social_cfg = config.get('social', agent_cfg)  # 兼容两种写法
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    max_timesteps = market_cfg.get('max_timesteps')
    agent_count = agent_cfg.get('count')
    retail_ratio = agent_cfg['types'].get('retail_ratio', 0.9)
    enable_cyberbullying = social_cfg.get('enable_cyberbullying', False)
    enable_bully_infect = social_cfg.get('enable_bully_infect', False)
    enable_regulator = social_cfg.get('enable_regulator', False)
    enable_bully_report = social_cfg.get('enable_bully_report', False)

    # 由分析器统一返回实验结果
    summary = wealth_analyzer.get_summary(initial_agents, final_agents, initial_price, final_price)

    row = {
        'timestamp': timestamp,
        'max_timesteps': max_timesteps,
        'agent_count': agent_count,
        'retail_ratio': retail_ratio,
        'enable_cyberbullying': enable_cyberbullying,
        'enable_bully_infect': enable_bully_infect,
        'enable_regulator': enable_regulator,
        'enable_bully_report': enable_bully_report,
        **summary
    }
    # 保存
    try:
        df = pd.read_excel(save_path)
        df = df.append(row, ignore_index=True)
    except Exception:
        df = pd.DataFrame([row])
    df.to_excel(save_path, index=False)
    print(f"实验记录已保存到 {save_path}")

    
if __name__ == "__main__":
    
    run()

