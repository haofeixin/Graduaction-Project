import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from src.traders.base import BaseTrader
from scipy.interpolate import make_interp_spline

# 关闭交互模式
plt.ioff()

class WealthAnalyzer:
    def __init__(self, traders: List[BaseTrader], fundamental_price: float):
        self.traders = traders
        self.fundamental_price = fundamental_price
        self.initial_wealth = self._calculate_total_wealth()
        
    def _calculate_total_wealth(self) -> float:
        """计算市场总财富"""
        return sum(trader.cash + trader.stock * self.fundamental_price for trader in self.traders)
    
    def show_wealth_distribution(self, title: str = "Wealth Distribution"):
        """展示财富分布"""
        print(f"\n📊 {title}")
        print("=" * 50)
        print(f"{'Trader ID':<10} {'Type':<15} {'Cash':<15} {'Stock':<15} {'Total Wealth':<15} {'Wealth Share (%)':<15}")
        print("-" * 50)
        
        total_wealth = self._calculate_total_wealth()
        for trader in self.traders:
            wealth = trader.cash + trader.stock * self.fundamental_price
            wealth_share = (wealth / total_wealth) * 100
            print(f"{trader.trader_id:<10} {trader.type:<15} {trader.cash:<15.2f} {trader.stock:<15.2f} {wealth:<15.2f} {wealth_share:<15.2f}")
        print("=" * 50)
        
    def plot_wealth_distribution(self, save_path: str = None):
        """绘制财富分布直方图"""
        wealths = [trader.cash + trader.stock * self.fundamental_price for trader in self.traders]
        total_traders = len(self.traders)
        
        plt.figure(figsize=(10, 6))
        
        # 使用Sturges规则计算bin数量: k = 1 + log2(n)
        n_bins = int(1 + np.log2(total_traders))
        
        # 创建直方图，density=True使得纵轴显示密度（占比）
        plt.hist(wealths, bins=n_bins, density=True, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Wealth Distribution Among Traders')
        plt.xlabel('Total Wealth')
        plt.ylabel('Proportion of Traders')
        plt.grid(True, alpha=0.3)
        
        # 添加均值和中位数的垂直线
        mean_wealth = np.mean(wealths)
        median_wealth = np.median(wealths)
        plt.axvline(mean_wealth, color='red', linestyle='--', label=f'Mean: {mean_wealth:.2f}')
        plt.axvline(median_wealth, color='green', linestyle='--', label=f'Median: {median_wealth:.2f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def calculate_wealth_inequality(self) -> Dict[str, float]:
        """计算财富不平等指标"""
        wealths = [trader.cash + trader.stock * self.fundamental_price for trader in self.traders]
        total_wealth = sum(wealths)
        
        # 计算基尼系数
        sorted_wealths = np.sort(wealths)
        n = len(wealths)
        index = np.arange(1, n + 1)
        gini = ((2 * index - n - 1) * sorted_wealths).sum() / (n * total_wealth)
        
        # 计算财富集中度（前20%交易者拥有的财富比例）
        top_20_percent = int(n * 0.2)
        top_wealth = sum(sorted_wealths[-top_20_percent:])
        wealth_concentration = top_wealth / total_wealth
        
        return {
            'gini_coefficient': gini,
            'wealth_concentration': wealth_concentration
        }

    def plot_analysis(self, initial_traders, final_traders, initial_price, final_price, save_prefix: str = "results/wealth/"):
        """
        分别绘制散户和机构的初始/最终财富分布对比图（仅直方图，标注均值和中位数）。
        :param initial_traders: 初始traders列表
        :param final_traders: 最终traders列表
        :param initial_price: 初始市场价格
        :param final_price: 最终市场价格
        :param save_prefix: 保存路径前缀
        """
        def plot_distribution(ax, wealths, color, label, bins=30):
            if len(wealths) == 0:
                return
            min_w, max_w = min(wealths), max(wealths)
            bin_edges = np.linspace(min_w, max_w, bins + 1)
            counts, _ = np.histogram(wealths, bins=bin_edges)
            total = sum(counts)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            proportions = counts / total if total > 0 else np.zeros_like(counts)
            # 只画直方图
            ax.bar(centers, proportions, width=(bin_edges[1]-bin_edges[0])*0.9, color=color, alpha=0.6, label=label)
            # 标注均值和中位数
            mean = np.mean(wealths)
            
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label='Mean')
            
            ax.set_xlabel("Wealth")
            ax.set_ylabel("Proportion")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 散户
        initial_retail = [t for t in initial_traders if t.type == 'retail']
        final_retail = [t for t in final_traders if t.type == 'retail']
        initial_retail_wealth = [t.cash + t.stock * initial_price for t in initial_retail]
        final_retail_wealth = [t.cash + t.stock * final_price for t in final_retail]

        plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(1, 2, 1)
        plot_distribution(ax1, initial_retail_wealth, color='blue', label='Initial')
        plt.title("Retail Traders Initial Wealth Distribution")
        ax2 = plt.subplot(1, 2, 2)
        plot_distribution(ax2, final_retail_wealth, color='green', label='Final')
        plt.title("Retail Traders Final Wealth Distribution")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}retail_wealth_compare.png")
        plt.close()

        # 机构
        initial_inst = [t for t in initial_traders if t.type == 'institutional']
        final_inst = [t for t in final_traders if t.type == 'institutional']
        initial_inst_wealth = [t.cash + t.stock * initial_price for t in initial_inst]
        final_inst_wealth = [t.cash + t.stock * final_price for t in final_inst]

        plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(1, 2, 1)
        plot_distribution(ax1, initial_inst_wealth, color='red', label='Initial')
        plt.title("Institutional Traders Initial Wealth Distribution")
        ax2 = plt.subplot(1, 2, 2)
        plot_distribution(ax2, final_inst_wealth, color='orange', label='Final')
        plt.title("Institutional Traders Final Wealth Distribution")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}institutional_wealth_compare.png")
        plt.close()

    def generate_report(self) -> str:
        """只输出散户和机构的财富统计"""
        # 分离散户和机构
        retail_traders = [t for t in self.traders if t.type == 'retail']
        institutional_traders = [t for t in self.traders if t.type == 'institutional']
        
        retail_wealths = [t.cash + t.stock * self.fundamental_price for t in retail_traders]
        institutional_wealths = [t.cash + t.stock * self.fundamental_price for t in institutional_traders]
        
        report = f"""
Wealth Analysis Report
=====================

1. Retail Traders
----------------
- Number of traders: {len(retail_traders)}
- Average wealth: {np.mean(retail_wealths):.2f} (±{np.std(retail_wealths):.2f})
- Median wealth: {np.median(retail_wealths):.2f}
- Min wealth: {np.min(retail_wealths):.2f}
- Max wealth: {np.max(retail_wealths):.2f}

2. Institutional Traders
-----------------------
- Number of traders: {len(institutional_traders)}
- Average wealth: {np.mean(institutional_wealths):.2f} (±{np.std(institutional_wealths):.2f})
- Median wealth: {np.median(institutional_wealths):.2f}
- Min wealth: {np.min(institutional_wealths):.2f}
- Max wealth: {np.max(institutional_wealths):.2f}
"""
        return report 

    def get_summary(self, initial_agents, final_agents, initial_price, final_price):
        def calc_wealth(agents, price):
            return [t.cash + t.stock * price for t in agents]
        def calc_gini(wealths):
            sorted_wealths = np.sort(wealths)
            n = len(wealths)
            total_wealth = np.sum(sorted_wealths)
            if total_wealth == 0 or n == 0:
                return 0.0
            index = np.arange(1, n + 1)
            gini = ((2 * index - n - 1) * sorted_wealths).sum() / (n * total_wealth)
            return gini

        initial_retail = [t for t in initial_agents if t.type == 'retail']
        final_retail = [t for t in final_agents if t.type == 'retail']
        initial_inst = [t for t in initial_agents if t.type == 'institutional']
        final_inst = [t for t in final_agents if t.type == 'institutional']

        initial_retail_wealth = calc_wealth(initial_retail, initial_price)
        final_retail_wealth = calc_wealth(final_retail, final_price)
        initial_inst_wealth = calc_wealth(initial_inst, initial_price)
        final_inst_wealth = calc_wealth(final_inst, final_price)

        initial_gini = calc_gini(initial_retail_wealth + initial_inst_wealth)
        final_gini = calc_gini(final_retail_wealth + final_inst_wealth)

        return {
            'initial_retail_mean': np.mean(initial_retail_wealth),
            'final_retail_mean': np.mean(final_retail_wealth),
            'initial_inst_mean': np.mean(initial_inst_wealth),
            'final_inst_mean': np.mean(final_inst_wealth),
            'initial_gini': initial_gini,
            'final_gini': final_gini,
        } 