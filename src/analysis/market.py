import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from src.social.cyberbullying import CyberbullyingModel
# 关闭交互模式
plt.ioff()

class MarketAnalyzer:
    def __init__(self, price_history: List[float], log_returns: List[float], fundamental_price_history: List[float], cyberbullying_model: CyberbullyingModel):
        self.price_history = price_history
        self.log_returns = log_returns
        self.fundamental_price_history = fundamental_price_history
        self.cyberbullying_model = cyberbullying_model

    def plot_price_evolution(self, save_path: str = None):
        """绘制价格演化图"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.price_history, label='Market Price', color='blue')
        plt.plot(self.fundamental_price_history, label='Fundamental Price', color='red', linestyle='--')
        plt.title('Price Evolution')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(f"results/market/{save_path}")
        plt.close()
        
    def plot_returns_distribution(self, save_path: str = None):
        """绘制收益率时间序列图"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.log_returns, color='blue', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.title('Log Returns Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Log Return')
        plt.grid(True, alpha=0.3)
        
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
        returns = np.array(self.log_returns)
        price = np.array(self.price_history)
        fundamental = np.array(self.fundamental_price_history)
        
        # 计算波动率
        volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
        
        # 计算价格偏离
        price_deviation = np.mean(np.abs(price - fundamental) / fundamental)
        
        # 计算自相关性
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        
        return {
            'volatility': volatility,
            'price_deviation': price_deviation,
            'autocorrelation': autocorr
        }
        
    def plot_volatility_evolution(self, window: int = 20, save_path: str = None):
        """绘制波动率演化图"""
        # 确保所有返回值都是浮点数
        returns = np.array(self.log_returns, dtype=float)
        rolling_vol = np.array([np.std(returns[max(0, i-window):i+1]) * np.sqrt(252) 
                              for i in range(len(returns))])
        
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_vol)
        plt.title('Rolling Volatility (20-day window)')
        plt.xlabel('Time Step')
        plt.ylabel('Annualized Volatility')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_analysis(self, save_path: str = None):
        """绘制分析结果"""
        plt.figure(figsize=(15, 10))
        
        # 1. 价格演化
        plt.subplot(2, 2, 1)
        price_history = self.price_history
        fundamental_price = self.fundamental_price_history[-1]
        plt.plot(price_history, label="Market Price")
        plt.axhline(y=fundamental_price, color='r', linestyle='--', label="Fundamental Price")
        plt.title("Price Evolution")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        
        # 2. 收益率分布
        plt.subplot(2, 2, 2)
        returns = self.log_returns
        plt.hist(returns, bins=50, density=True)
        plt.title("Return Distribution")
        plt.xlabel("Return")
        plt.ylabel("Density")
        
        # 3. 波动率
        plt.subplot(2, 2, 3)
        volatility = self.calculate_market_metrics()['volatility']
        plt.plot(volatility)
        plt.title("Volatility")
        plt.xlabel("Time")
        plt.ylabel("Volatility")
        
        # 4. 市场深度
        plt.subplot(2, 2, 4)
        depth = self.calculate_market_metrics()['price_deviation']
        plt.plot(depth)
        plt.title("Market Depth")
        plt.xlabel("Time")
        plt.ylabel("Depth")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"results/market/{save_path}")
        plt.close() 