import pandas as pd
from datetime import datetime
from typing import List
from src.traders.base import BaseTrader
from src.analysis.wealth import WealthAnalyzer

def show_trade_records(orderbook):
    """
    显示所有交易记录
    
    Args:
        orderbook: 订单簿对象
    """
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

def log_experiment(config: dict, 
                  initial_agents: List[BaseTrader], 
                  final_agents: List[BaseTrader], 
                  initial_price: float, 
                  final_price: float, 
                  wealth_analyzer: WealthAnalyzer, 
                  save_path: str = 'results/experiment_log.xlsx'):
    """
    记录实验参数和结果
    
    Args:
        config: 配置字典
        initial_agents: 初始时刻的智能体列表
        final_agents: 最终时刻的智能体列表
        initial_price: 初始价格
        final_price: 最终价格
        wealth_analyzer: 财富分析器
        save_path: 保存路径
    """
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