import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple
from src.traders.base import BaseTrader
from src.traders.retail import RetailTrader
from src.traders.institutional import InstitutionalTrader
from src.market.structure import Market
from src.config.loader import load_config
from src.analysis.logger import log_experiment
from src.analysis.wealth import WealthAnalyzer
from src.analysis.market import MarketAnalyzer
from src.order.orderbooks import OrderBook
from src.social.cyberbullying import CyberbullyingModel
import os

def create_agents(config: Dict) -> List[BaseTrader]:
    """创建agents"""
    agents = []
    
    # 从配置中获取参数
    n_retail = int(config["count"] * config["retail_ratio"])
    n_institutional = config["count"] - n_retail
    
    # 创建散户
    for i in range(n_retail):
        agent = RetailTrader(
            trader_id=i,
            config=config,
            trader_type="retail"
        )
        agents.append(agent)
    
    # 创建机构
    for i in range(n_institutional):
        agent = InstitutionalTrader(
            trader_id=n_retail + i,  # 确保ID不重复
            config=config,
            trader_type="institutional"
        )
        agents.append(agent)
    
    return agents

def run_scenario(enable_cyberbullying: bool, tag: str, seed: int = None) -> Tuple[List[BaseTrader], List[BaseTrader], float, float, Market]:
    """
    运行单个场景的模拟
    
    Args:
        enable_cyberbullying: 是否启用网络欺凌
        tag: 场景标签
        seed: 随机种子，如果为None则使用配置中的种子
    
    Returns:
        Tuple[List[BaseTrader], List[BaseTrader], float, float, Market]: 
        (初始交易者列表, 最终交易者列表, 初始价格, 最终价格, 市场对象)
    """
    # 加载配置
    config = load_config("config.yaml")
    
    base_seed = config["simulation"]["random_seed"]
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(base_seed)
    
    # 创建智能体
    agents = create_agents(config['agents'])
    
    # 创建订单簿
    orderbook = OrderBook()
    
    # 创建网络欺凌模型（如果需要）
    cyberbullying_model = None
    if enable_cyberbullying:
        cyberbullying_model = CyberbullyingModel(config, agents)
        cyberbullying_model.build_network()
    
    # 创建市场
    market = Market(
        orderbook=orderbook,
        max_timesteps=config["market"]["max_timesteps"],
        config=config,
        cyberbullying_model=cyberbullying_model
    )
    
    # 注册智能体
    for agent in agents:
        market.register_agent(agent)
    
    # 记录初始状态
    initial_agents = [agent.copy() for agent in agents]
    initial_price = market.fundamental_price
    
    # 运行模拟
    market.run()
    
    # 获取最终价格
    final_price = market.price_history[-1] if market.price_history else market.fundamental_price
    
    # 记录实验结果
    # log_experiment(
    #     tag=tag,
    #     trade_records=market.orderbook.trade_log,
    #     final_price=final_price
    # )
    
    return initial_agents, agents, initial_price, final_price, market

def run_single_experiment():
    """运行单次实验"""
    # 运行baseline场景
    baseline_initial, baseline_final, baseline_initial_price, baseline_final_price, baseline_market = run_scenario(False, "baseline")
    
    # 运行cyberbullying场景
    cyber_initial, cyber_final, cyber_initial_price, cyber_final_price, cyber_market = run_scenario(True, "cyberbullying")
    
    # 分析baseline结果
    baseline_wealth_analyzer = WealthAnalyzer(baseline_final, baseline_final_price)
    baseline_wealth_changes = baseline_wealth_analyzer.get_summary(baseline_initial, baseline_final, 
                                                                 baseline_initial_price, baseline_final_price)
    
    baseline_market_analyzer = MarketAnalyzer(baseline_market)
    
    # 分析cyberbullying结果
    cyber_wealth_analyzer = WealthAnalyzer(cyber_final, cyber_final_price)
    cyber_wealth_changes = cyber_wealth_analyzer.get_summary(cyber_initial, cyber_final, 
                                                          cyber_initial_price, cyber_final_price)
    
    cyber_market_analyzer = MarketAnalyzer(cyber_market, cyber_market.cyberbullying_model)
    
    # 生成分析报告
    print_single_experiment_results(baseline_wealth_changes, cyber_wealth_changes,
                                  baseline_market_analyzer, cyber_market_analyzer)

def print_single_experiment_results(baseline_wealth: Dict, cyber_wealth: Dict,
                                  baseline_market: MarketAnalyzer, cyber_market: MarketAnalyzer):
    """保存单次实验结果到Excel文件"""
    # 创建结果目录
    os.makedirs("results/market/single_expe", exist_ok=True)
    
    # 1. 财富分析结果
    wealth_results = []
    
    # 散户群体
    retail_baseline_change = (baseline_wealth['final_retail_mean'] - baseline_wealth['initial_retail_mean']) / baseline_wealth['initial_retail_mean'] * 100
    retail_cyber_change = (cyber_wealth['final_retail_mean'] - cyber_wealth['initial_retail_mean']) / cyber_wealth['initial_retail_mean'] * 100
    retail_diff = retail_cyber_change - retail_baseline_change
    
    wealth_results.append({
        'Group': 'Retail',
        'Baseline Change (%)': retail_baseline_change,
        'Cyberbullying Change (%)': retail_cyber_change,
        'Difference (%)': retail_diff
    })
    
    # 机构群体
    inst_baseline_change = (baseline_wealth['final_inst_mean'] - baseline_wealth['initial_inst_mean']) / baseline_wealth['initial_inst_mean'] * 100
    inst_cyber_change = (cyber_wealth['final_inst_mean'] - cyber_wealth['initial_inst_mean']) / cyber_wealth['initial_inst_mean'] * 100
    inst_diff = inst_cyber_change - inst_baseline_change
    
    wealth_results.append({
        'Group': 'Institutional',
        'Baseline Change (%)': inst_baseline_change,
        'Cyberbullying Change (%)': inst_cyber_change,
        'Difference (%)': inst_diff
    })
    
    # 2. 市场分析结果
    baseline_metrics = baseline_market.calculate_market_metrics()
    cyber_metrics = cyber_market.calculate_market_metrics()
    
    market_results = []
    for metric in ['volatility', 'price_deviation', 'autocorrelation', 
                  'skewness', 'kurtosis', 'max_drawdown', 
                  'bid_ask_spread', 'order_depth', 'amihud_illiquidity']:
        market_results.append({
            'Metric': metric,
            'Baseline': baseline_metrics[metric],
            'Cyberbullying': cyber_metrics[metric],
            'Difference': cyber_metrics[metric] - baseline_metrics[metric]
        })
    
    # 创建DataFrame并保存到Excel
    wealth_df = pd.DataFrame(wealth_results)
    market_df = pd.DataFrame(market_results)
    
    # 保存到Excel文件
    with pd.ExcelWriter('results/market/single_expe/single_experiment_results.xlsx') as writer:
        wealth_df.to_excel(writer, sheet_name='Wealth Analysis', index=False)
        market_df.to_excel(writer, sheet_name='Market Analysis', index=False)
    
    # 保存图表
    baseline_market.plot_analysis(save_path="results/market/single_expe/baseline_analysis.png")
    cyber_market.plot_analysis(save_path="results/market/single_expe/cyberbullying_analysis.png")

def run_paired_t_test(n_simulations: int = 1000) -> Dict:
    """
    运行配对T检验实验
    
    Args:
        n_simulations: 模拟次数
    
    Returns:
        Dict: 统计检验结果
    """
    baseline_results = []
    cyberbullying_results = []
    baseline_market_metrics = []
    cyberbullying_market_metrics = []
    
    # 获取基础配置
    config = load_config("config.yaml")
    base_seed = config["simulation"]["random_seed"]  # 直接从配置中读取基础种子
    
    # 设置一个固定的种子序列，确保每次运行配对t检验时使用相同的种子序列
    np.random.seed(base_seed)
    simulation_seeds = np.random.randint(0, 1000000, size=n_simulations)
    
    for i, seed in enumerate(simulation_seeds):
        # 运行baseline场景
        baseline_initial, baseline_final, baseline_initial_price, baseline_final_price, baseline_market = run_scenario(False, f"baseline_{i}", seed)
        baseline_analyzer = WealthAnalyzer(baseline_final, baseline_final_price)
        baseline_wealth_changes = baseline_analyzer.get_summary(baseline_initial, baseline_final, 
                                                              baseline_initial_price, baseline_final_price)
        baseline_results.append(baseline_wealth_changes)
        
        baseline_market_analyzer = MarketAnalyzer(baseline_market)
        baseline_market_metrics.append(baseline_market_analyzer.calculate_market_metrics())
        
        # 运行cyberbullying场景（使用相同的种子）
        cyber_initial, cyber_final, cyber_initial_price, cyber_final_price, cyber_market = run_scenario(True, f"cyberbullying_{i}", seed)
        cyber_analyzer = WealthAnalyzer(cyber_final, cyber_final_price)
        cyber_wealth_changes = cyber_analyzer.get_summary(cyber_initial, cyber_final, 
                                                        cyber_initial_price, cyber_final_price)
        cyberbullying_results.append(cyber_wealth_changes)
        
        cyber_market_analyzer = MarketAnalyzer(cyber_market, cyber_market.cyberbullying_model)
        cyberbullying_market_metrics.append(cyber_market_analyzer.calculate_market_metrics())
    
    # 分析结果
    results = analyze_paired_results(baseline_results, cyberbullying_results,
                                   baseline_market_metrics, cyberbullying_market_metrics)
    print_paired_t_test_results(results)
    return 

def analyze_paired_results(baseline_wealth: List[Dict], cyber_wealth: List[Dict],
                         baseline_market: List[Dict], cyber_market: List[Dict]) -> None:
    """分析配对T检验结果"""
    # 创建结果目录
    os.makedirs("results/market", exist_ok=True)
    
    # 1. 财富变化分析
    wealth_results = []
    for metric in ['mean', 'median', 'std', 'skew', 'kurt']:
        baseline_values = [w[metric] for w in baseline_wealth]
        cyber_values = [w[metric] for w in cyber_wealth]
        
        # 计算差异
        differences = np.array(cyber_values) - np.array(baseline_values)
        
        # 执行配对T检验
        t_stat, p_value = stats.ttest_rel(cyber_values, baseline_values)
        
        # 计算置信区间
        mean_diff = np.mean(differences)
        std_err = np.std(differences, ddof=1) / np.sqrt(len(differences))
        ci_lower = mean_diff - 1.96 * std_err
        ci_upper = mean_diff + 1.96 * std_err
        
        wealth_results.append({
            'Metric': metric,
            'Baseline Mean': np.mean(baseline_values),
            'Cyberbullying Mean': np.mean(cyber_values),
            'Difference': mean_diff,
            'Std Error': std_err,
            'CI Lower': ci_lower,
            'CI Upper': ci_upper,
            't-statistic': t_stat,
            'p-value': p_value
        })
    
    # 2. 市场指标分析
    market_results = []
    for metric in ['volatility', 'price_deviation', 'autocorrelation', 
                  'skewness', 'kurtosis', 'max_drawdown', 
                  'bid_ask_spread', 'order_depth', 'amihud_illiquidity']:
        baseline_values = [m[metric] for m in baseline_market]
        cyber_values = [m[metric] for m in cyber_market]
        
        # 计算差异
        differences = np.array(cyber_values) - np.array(baseline_values)
        
        # 执行配对T检验
        t_stat, p_value = stats.ttest_rel(cyber_values, baseline_values)
        
        # 计算置信区间
        mean_diff = np.mean(differences)
        std_err = np.std(differences, ddof=1) / np.sqrt(len(differences))
        ci_lower = mean_diff - 1.96 * std_err
        ci_upper = mean_diff + 1.96 * std_err
        
        market_results.append({
            'Metric': metric,
            'Baseline Mean': np.mean(baseline_values),
            'Cyberbullying Mean': np.mean(cyber_values),
            'Difference': mean_diff,
            'Std Error': std_err,
            'CI Lower': ci_lower,
            'CI Upper': ci_upper,
            't-statistic': t_stat,
            'p-value': p_value
        })
    
    # 创建DataFrame并保存到Excel
    wealth_df = pd.DataFrame(wealth_results)
    market_df = pd.DataFrame(market_results)
    
    # 保存到Excel文件
    with pd.ExcelWriter('results/market/paired_t_tests.xlsx') as writer:
        wealth_df.to_excel(writer, sheet_name='Wealth Analysis', index=False)
        market_df.to_excel(writer, sheet_name='Market Analysis', index=False)

def print_paired_t_test_results(results: Dict):
    """打印配对t检验结果"""
    print("\n=== 配对t检验结果 ===")
    
    # 创建指标名称映射
    metric_names = {
        'wealth_retail': '散户财富变化',
        'wealth_institutional': '机构财富变化',
        'volatility': '波动率',
        'price_deviation': '价格偏离度',
        'autocorrelation': '自相关性',
        'skewness': '偏度',
        'kurtosis': '峰度',
        'max_drawdown': '最大回撤',
        'bid_ask_spread': '买卖价差',
        'order_depth': '订单深度',
        'amihud_illiquidity': 'Amihud非流动性'
    }
    
    # 创建结果表格
    print("\n{:<20} {:<15} {:<15} {:<15} {:<15} {:<10}".format(
        "指标", "Baseline均值", "Cyber均值", "t统计量", "p值", "显著性"
    ))
    print("-" * 90)
    
    # 打印财富指标
    for trader_type in ['retail', 'institutional']:
        metric = f'wealth_{trader_type}'
        data = results['wealth'][trader_type]
        print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<10}".format(
            metric_names[metric],
            data['baseline_mean'],
            data['cyberbullying_mean'],
            data['t_statistic'],
            data['p_value'],
            "***" if data['significant'] else "ns"
        ))
    
    # 打印市场指标
    for metric in results['market'].keys():
        data = results['market'][metric]
        print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<10}".format(
            metric_names[metric],
            data['baseline_mean'],
            data['cyberbullying_mean'],
            data['t_statistic'],
            data['p_value'],
            "***" if data['significant'] else "ns"
        ))
    
    print("\n注: *** 表示p < 0.05, ns表示不显著") 