import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from src.social.cyberbullying import CyberbullyingModel
from scipy.stats import gaussian_kde
from src.market.structure import Market
# 关闭交互模式
plt.ioff()

class MarketAnalyzer:
    def __init__(self, market: Market, cyberbullying_model: Optional[CyberbullyingModel] = None):
        self.market = market
        self.orderbook = market.orderbook
        self.cyberbullying_model = cyberbullying_model
        self.price_history = market.price_history
        self.fundamental_price_history = market.fundamental_price_history
        self.log_returns = np.array(market.log_returns)
        self.bid_ask_spreads = market.bid_ask_spreads
        self.order_depths = market.order_depths
        self.amihud_illiquidity = market.amihud_illiquidity

    def _calculate_rolling_volatility(self, window: int = 20) -> np.ndarray:
        """计算滚动波动率"""
        log_returns = pd.Series(self.log_returns)
        return log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)

    def _plot_time_series(self, data: np.ndarray, title: str, xlabel: str, ylabel: str, 
                         color: str = 'blue', label: str = None, grid: bool = True) -> None:
        """绘制时间序列图"""
        plt.plot(data, color=color, label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if label:
            plt.legend()
        if grid:
            plt.grid(True, alpha=0.3)

    def _plot_histogram_with_kde(self, data: np.ndarray, title: str, xlabel: str, ylabel: str,
                                bins: int = 60, color: str = 'orange', alpha: float = 0.7) -> None:
        """绘制直方图和核密度估计"""
        mean = np.mean(data)
        std = np.std(data)
        range_min = mean - 2 * std
        range_max = mean + 2 * std
        filtered_data = data[(data >= range_min) & (data <= range_max)]
        
        plt.hist(filtered_data, bins=bins, density=True, color=color, alpha=alpha, label='Histogram')
        if len(filtered_data) > 1:
            kde = gaussian_kde(filtered_data)
            x_grid = np.linspace(range_min, range_max, 500)
            plt.plot(x_grid, kde(x_grid), color='red', lw=2, label='KDE')
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(range_min, range_max)
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot_price_evolution(self, save_path: str = None):
        """绘制价格演化图"""
        plt.figure(figsize=(12, 6))
        self._plot_time_series(self.price_history, 'Price Evolution', 'Time Step', 'Price', 
                             color='blue', label='Market Price')
        self._plot_time_series(self.fundamental_price_history, 'Price Evolution', 'Time Step', 'Price',
                             color='red', label='Fundamental Price', grid=False)
        plt.legend()
        
        if save_path:
            plt.savefig(f"results/market/{save_path}")
        plt.close()
        
    def plot_returns_distribution(self, save_path: str = None):
        """绘制收益率时间序列图"""
        plt.figure(figsize=(12, 6))
        self._plot_time_series(self.log_returns, 'Log Returns Over Time', 'Time Step', 'Log Return')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # 添加统计信息
        mean_return = np.mean(self.log_returns)
        std_return = np.std(self.log_returns)
        plt.axhline(y=mean_return, color='red', linestyle='--', label=f'Mean: {mean_return:.4f}')
        plt.axhline(y=mean_return + std_return, color='green', linestyle=':', label=f'±1 Std: {std_return:.4f}')
        plt.axhline(y=mean_return - std_return, color='green', linestyle=':')
        plt.legend()
        
        if save_path:
            plt.savefig(f"results/market/{save_path}")
        plt.close()
        
    def calculate_market_metrics(self) -> Dict[str, float]:
        """计算市场指标"""
        # 确保价格历史长度一致
        min_length = min(len(self.price_history), len(self.fundamental_price_history))
        price_history = np.array(self.price_history[:min_length], dtype=float)
        fundamental_price_history = np.array(self.fundamental_price_history[:min_length], dtype=float)
        log_returns = np.array(self.log_returns, dtype=float)
        
        # 计算波动率
        volatility = np.std(log_returns) * np.sqrt(252)  # 年化波动率
        
        # 计算价格偏离
        price_deviation = np.mean(np.abs(price_history - fundamental_price_history) / fundamental_price_history)
        
        # 计算自相关性
        autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
        
        # 计算偏度和峰度
        skewness = pd.Series(log_returns).skew()
        kurtosis = pd.Series(log_returns).kurtosis()
        
        # 计算最大回撤
        cumulative_returns = np.cumprod(1 + log_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)

        # 获取流动性指标
        liquidity_metrics = self.market.get_liquidity_metrics()
        
        return {
            'volatility': volatility,
            'price_deviation': price_deviation,
            'autocorrelation': autocorr,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'max_drawdown': max_drawdown,
            'bid_ask_spread': liquidity_metrics['bid_ask_spread'],
            'order_depth': liquidity_metrics['order_depth'],
            'amihud_illiquidity': liquidity_metrics['amihud_illiquidity']
        }
        
    def plot_volatility_evolution(self, window: int = 20, save_path: str = None):
        """绘制波动率演化图"""
        rolling_vol = self._calculate_rolling_volatility(window)
        
        plt.figure(figsize=(12, 6))
        self._plot_time_series(rolling_vol, f'Rolling Volatility ({window}-day window)', 
                             'Time Step', 'Annualized Volatility', color='green')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_analysis(self, save_path: str = None, window: int = 200):
        """绘制综合分析图"""
        plt.figure(figsize=(15, 10))

        # 1. 价格走势
        plt.subplot(2, 2, 1)
        self._plot_time_series(self.price_history, "Price Evolution", "Time", "Price", 
                             color='blue', label="Market Price")
        self._plot_time_series(self.fundamental_price_history, "Price Evolution", "Time", "Price",
                             color='red', label="Fundamental Price", grid=False)
        plt.legend()

        # 2. 收益率变化（时间序列）
        plt.subplot(2, 2, 2)
        self._plot_time_series(self.log_returns, "Log Returns Over Time", "Time", "Log Return",
                             color='purple')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ymin, ymax = self.log_returns.min(), self.log_returns.max()
        yrange = ymax - ymin
        plt.ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)

        # 3. 收益率分布
        plt.subplot(2, 2, 3)
        self._plot_histogram_with_kde(self.log_returns, "Return Distribution", 
                                    "Log Return", "Probability Density")

        # 4. 波动率集聚
        plt.subplot(2, 2, 4)
        rolling_vol = self._calculate_rolling_volatility(window)
        self._plot_time_series(rolling_vol, f"Rolling Volatility ({window}-step)", 
                             "Time", "Annualized Volatility", color='green')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def generate_report(self) -> str:
        """生成市场分析报告"""
        metrics = self.calculate_market_metrics()
        
        report = f"""
Market Analysis Report
=====================

1. Market Statistics
-------------------
- Volatility (annualized): {metrics['volatility']:.4f}
- Price Deviation: {metrics['price_deviation']:.4f}
- Return Autocorrelation: {metrics['autocorrelation']:.4f}
- Return Skewness: {metrics['skewness']:.4f}
- Return Kurtosis: {metrics['kurtosis']:.4f}
- Maximum Drawdown: {metrics['max_drawdown']:.4f}

2. Price Analysis
----------------
- Initial Price: {self.price_history[0]:.2f}
- Final Price: {self.price_history[-1]:.2f}
- Total Return: {(self.price_history[-1] / self.price_history[0] - 1) * 100:.2f}%
- Average Daily Return: {np.mean(self.log_returns) * 100:.4f}%
- Return Volatility: {np.std(self.log_returns) * 100:.4f}%

"""
        return report 