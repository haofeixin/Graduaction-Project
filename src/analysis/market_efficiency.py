import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from src.analysis.experiment_utils import run_scenario
from src.config.loader import load_config
from src.analysis.market import MarketAnalyzer
from matplotlib import pyplot as plt

def run_market_efficiency_analysis_multi(baseline_markets, cyber_markets, save_dir='thesis/image'):
    """运行多次实验的市场效率分析"""
    os.makedirs(save_dir, exist_ok=True)
    results = []

    # 1. 收益波动率分析
    baseline_volatility = []
    cyber_volatility = []
    for base_market, cyber_market in zip(baseline_markets, cyber_markets):
        base_analyzer = MarketAnalyzer(base_market)
        cyber_analyzer = MarketAnalyzer(cyber_market)
        base_metrics = base_analyzer.calculate_market_metrics()
        cyber_metrics = cyber_analyzer.calculate_market_metrics()
        baseline_volatility.append(base_metrics['volatility'])
        cyber_volatility.append(cyber_metrics['volatility'])
    t_stat_vol, p_val_vol = ttest_rel(baseline_volatility, cyber_volatility)
    results.append({
        'metric': '收益波动率',
        't_stat': t_stat_vol,
        'p_val': p_val_vol,
        'baseline_mean': np.mean(baseline_volatility),
        'cyber_mean': np.mean(cyber_volatility)
    })
    print("配对t检验：收益波动率分析完成")

    # 2. 流动性指标分析
    # 2.1 Bid-Ask Spread
    baseline_spread = []
    cyber_spread = []
    for base_market, cyber_market in zip(baseline_markets, cyber_markets):
        base_analyzer = MarketAnalyzer(base_market)
        cyber_analyzer = MarketAnalyzer(cyber_market)
        base_metrics = base_analyzer.calculate_market_metrics()
        cyber_metrics = cyber_analyzer.calculate_market_metrics()
        baseline_spread.append(base_metrics['bid_ask_spread'])
        cyber_spread.append(cyber_metrics['bid_ask_spread'])
    t_stat_spread, p_val_spread = ttest_rel(baseline_spread, cyber_spread)
    results.append({
        'metric': '买卖价差',
        't_stat': t_stat_spread,
        'p_val': p_val_spread,
        'baseline_mean': np.mean(baseline_spread),
        'cyber_mean': np.mean(cyber_spread)
    })

    # 2.2 Market Depth
    baseline_depth = []
    cyber_depth = []
    for base_market, cyber_market in zip(baseline_markets, cyber_markets):
        base_analyzer = MarketAnalyzer(base_market)
        cyber_analyzer = MarketAnalyzer(cyber_market)
        base_metrics = base_analyzer.calculate_market_metrics()
        cyber_metrics = cyber_analyzer.calculate_market_metrics()
        baseline_depth.append(base_metrics['order_depth'])
        cyber_depth.append(cyber_metrics['order_depth'])
    t_stat_depth, p_val_depth = ttest_rel(baseline_depth, cyber_depth)
    results.append({
        'metric': '市场深度',
        't_stat': t_stat_depth,
        'p_val': p_val_depth,
        'baseline_mean': np.mean(baseline_depth),
        'cyber_mean': np.mean(cyber_depth)
    })

    # 2.3 Amihud非流动性指标
    baseline_amihud = []
    cyber_amihud = []
    for base_market, cyber_market in zip(baseline_markets, cyber_markets):
        base_analyzer = MarketAnalyzer(base_market)
        cyber_analyzer = MarketAnalyzer(cyber_market)
        base_metrics = base_analyzer.calculate_market_metrics()
        cyber_metrics = cyber_analyzer.calculate_market_metrics()
        baseline_amihud.append(base_metrics['amihud_illiquidity'])
        cyber_amihud.append(cyber_metrics['amihud_illiquidity'])
    t_stat_amihud, p_val_amihud = ttest_rel(baseline_amihud, cyber_amihud)
    results.append({
        'metric': 'Amihud非流动性',
        't_stat': t_stat_amihud,
        'p_val': p_val_amihud,
        'baseline_mean': np.mean(baseline_amihud),
        'cyber_mean': np.mean(cyber_amihud)
    })
    print("配对t检验：流动性指标分析完成")

    # 3. 价格发现能力分析
    baseline_deviation = []
    cyber_deviation = []
    for base_market, cyber_market in zip(baseline_markets, cyber_markets):
        base_analyzer = MarketAnalyzer(base_market)
        cyber_analyzer = MarketAnalyzer(cyber_market)
        base_metrics = base_analyzer.calculate_market_metrics()
        cyber_metrics = cyber_analyzer.calculate_market_metrics()
        baseline_deviation.append(base_metrics['price_deviation'])
        cyber_deviation.append(cyber_metrics['price_deviation'])
    t_stat_dev, p_val_dev = ttest_rel(baseline_deviation, cyber_deviation)
    results.append({
        'metric': '价格偏离度',
        't_stat': t_stat_dev,
        'p_val': p_val_dev,
        'baseline_mean': np.mean(baseline_deviation),
        'cyber_mean': np.mean(cyber_deviation)
    })
    print("配对t检验：价格发现能力分析完成")

    # 保存t检验结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, 'table4_3_market_efficiency_ttest_results.csv'), index=False)

    # 使用最后一次实验的结果生成可视化图表
    first_baseline = baseline_markets[0]
    first_cyber = cyber_markets[0]
    baseline_analyzer = MarketAnalyzer(first_baseline)
    cyber_analyzer = MarketAnalyzer(first_cyber)

    # 1. 收益波动率对比图
    plt.figure(figsize=(10, 6))
    plt.boxplot([baseline_volatility, cyber_volatility], labels=['Baseline', 'Cyberbullying'])
    plt.ylabel('Annualized Volatility')
    plt.title('收益波动率对比')
    plt.savefig(os.path.join(save_dir, 'fig4_8_volatility_comparison.png'))
    plt.close()

    # 2. 流动性指标对比图
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.boxplot([baseline_spread, cyber_spread], labels=['Baseline', 'Cyberbullying'])
    plt.ylabel('Bid-Ask Spread')
    plt.title('买卖价差对比')

    plt.subplot(132)
    plt.boxplot([baseline_depth, cyber_depth], labels=['Baseline', 'Cyberbullying'])
    plt.ylabel('Market Depth')
    plt.title('市场深度对比')

    plt.subplot(133)
    plt.boxplot([baseline_amihud, cyber_amihud], labels=['Baseline', 'Cyberbullying'])
    plt.ylabel('Amihud Illiquidity')
    plt.title('Amihud非流动性对比')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_9_liquidity_metrics_comparison.png'))
    plt.close()

    # 3. 价格发现能力对比图
    plt.figure(figsize=(10, 6))
    plt.boxplot([baseline_deviation, cyber_deviation], labels=['Baseline', 'Cyberbullying'])
    plt.ylabel('Price Deviation')
    plt.title('价格偏离度对比')
    plt.savefig(os.path.join(save_dir, 'fig4_10_price_deviation_comparison.png'))
    plt.close()

    print(f'4.3多次实验市场效率分析结果已生成，保存在{save_dir}')

def runner_4_3():
    """运行4.3节的市场效率分析"""
    # 4.3 多次实验
    baseline_markets = []
    cyber_markets = []
    config = load_config()
    for seed in range(config['simulation']['n_simulations']):  # 例如20次实验
        _, _, _, _, baseline_market = run_scenario(False, seed=seed)
        _, _, _, _, cyber_market = run_scenario(True, seed=seed)
        baseline_markets.append(baseline_market)
        cyber_markets.append(cyber_market)
        print(f"第{seed}次实验完成")
    run_market_efficiency_analysis_multi(baseline_markets, cyber_markets) 