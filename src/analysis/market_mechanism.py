import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import acf
from scipy.stats import ttest_ind
from src.analysis.market import MarketAnalyzer
from src.analysis.experiment_utils import run_scenario

def plot_price_evolution_comparison(baseline_market, cyber_market, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_market.price_history, label='Baseline', color='blue')
    plt.plot(cyber_market.price_history, label='Cyberbullying', color='red', alpha=0.7)
    plt.title('模拟市场价格时间序列（baseline vs cyberbullying）')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_1_price_evolution.png'))
    plt.close()

def plot_return_hist_comparison(baseline_market, cyber_market, save_dir):
    plt.figure(figsize=(10, 6))
    returns1 = np.array(baseline_market.log_returns)
    returns2 = np.array(cyber_market.log_returns)
    plt.hist(returns1, bins=60, alpha=0.6, label='Baseline', density=True)
    plt.hist(returns2, bins=60, alpha=0.6, label='Cyberbullying', density=True)
    plt.title('收益率分布直方图')
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_2_return_hist.png'))
    plt.close()

def plot_return_qq_single(market_analyzer, save_path, title):
    plt.figure(figsize=(6, 6))
    qqplot(np.array(market_analyzer.log_returns), line='s', ax=plt.gca())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_return_acf_baseline(baseline_market, save_dir):
    plt.figure(figsize=(8, 5))
    acf_vals = acf(np.abs(baseline_market.log_returns), nlags=40)
    plt.stem(range(len(acf_vals)), acf_vals, use_line_collection=True)
    plt.title('Baseline 收益率绝对值ACF')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_4_return_acf_baseline.png'))
    plt.close()

def generate_return_stats_table(baseline_market, cyber_market, save_dir):
    def stats(arr):
        arr = np.array(arr)
        return {
            '均值': np.mean(arr),
            '标准差': np.std(arr),
            '偏度': pd.Series(arr).skew(),
            '峰度': pd.Series(arr).kurtosis(),
        }
    stats1 = stats(baseline_market.log_returns)
    stats2 = stats(cyber_market.log_returns)
    t_stat, p_val = ttest_ind(baseline_market.log_returns, cyber_market.log_returns, equal_var=False)
    df = pd.DataFrame([stats1, stats2], index=['Baseline', 'Cyberbullying'])
    df['均值t检验p值'] = [p_val, p_val]
    df.to_csv(os.path.join(save_dir, 'table4_1_return_stats.csv'))
    return df

def run_market_mechanism_analysis(baseline_market, cyber_market, save_dir='thesis/image'):
    os.makedirs(save_dir, exist_ok=True)
    plot_price_evolution_comparison(baseline_market, cyber_market, save_dir)
    plot_return_hist_comparison(baseline_market, cyber_market, save_dir)
    plot_return_qq_single(baseline_market, os.path.join(save_dir, 'fig4_3_return_qq_baseline.png'), 'Baseline 收益率QQ图')
    plot_return_qq_single(cyber_market, os.path.join(save_dir, 'fig4_3_return_qq_cyberbullying.png'), 'Cyberbullying 收益率QQ图')
    plot_return_acf_baseline(baseline_market, save_dir)
    generate_return_stats_table(baseline_market, cyber_market, save_dir)
    print(f"4.1市场机制验证图表已生成，保存在{save_dir}") 

def runner_4_1():
    # baseline
    _, _, _, _, baseline_market = run_scenario(False, seed=0)
    # cyberbullying
    _, _, _, _, cyber_market = run_scenario(True, seed=0)
    
    # 4.1分析
    baseline_analyzer = MarketAnalyzer(baseline_market)
    cyber_analyzer = MarketAnalyzer(cyber_market)
    
    run_market_mechanism_analysis(baseline_analyzer, cyber_analyzer)