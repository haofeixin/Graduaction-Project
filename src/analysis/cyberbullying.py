import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from src.traders.base import BaseTrader
from src.market.structure import Market

class CyberbullyingAnalyzer:
    def __init__(self, traders: List[BaseTrader], market: Market):
        self.traders = traders
        self.market = market
        self.retail_traders = [t for t in traders if not t.is_institutional]
        
    def analyze_emotion_impact(self) -> Dict:
        """分析网暴对散户情绪的影响"""
        bullied_traders = [t for t in self.retail_traders if t.is_bullied]
        non_bullied_traders = [t for t in self.retail_traders if not t.is_bullied]
        
        return {
            "bullied_emotion_mean": np.mean([t.emotion_bias for t in bullied_traders]) if bullied_traders else 0,
            "non_bullied_emotion_mean": np.mean([t.emotion_bias for t in non_bullied_traders]) if non_bullied_traders else 0,
            "bullied_emotion_std": np.std([t.emotion_bias for t in bullied_traders]) if bullied_traders else 0,
            "non_bullied_emotion_std": np.std([t.emotion_bias for t in non_bullied_traders]) if non_bullied_traders else 0,
            "bullied_count": len(bullied_traders),
            "total_retail_count": len(self.retail_traders)
        }
    
    def analyze_trading_activity(self, orders: List) -> Dict:
        """分析网暴对交易活跃度的影响"""
        bullied_orders = [o for o in orders if o.trader.is_bullied]
        non_bullied_orders = [o for o in orders if not o.trader.is_bullied]
        
        return {
            "bullied_order_count": len(bullied_orders),
            "non_bullied_order_count": len(non_bullied_orders),
            "bullied_order_ratio": len(bullied_orders) / len(orders) if orders else 0,
            "bullied_traders_active_ratio": len(set(o.trader for o in bullied_orders)) / len([t for t in self.retail_traders if t.is_bullied]) if bullied_orders else 0
        }
    
    def analyze_market_efficiency(self) -> Dict:
        """分析网暴对市场效率的影响"""
        price_history = self.market.get_price_history()
        fundamental_price = self.market.get_fundamental_price()
        
        if len(price_history) < 2:
            return {}
            
        # 计算价格偏离度
        price_deviations = [(p - fundamental_price) / fundamental_price for p in price_history]
        
        # 计算波动率
        returns = np.diff(np.log(price_history))
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        return {
            "avg_price_deviation": np.mean(price_deviations),
            "max_price_deviation": np.max(np.abs(price_deviations)),
            "volatility": volatility
        }
    
    def analyze_wealth_distribution(self) -> Dict:
        """分析网暴对财富分布的影响"""
        bullied_wealth = [t.wealth for t in self.retail_traders if t.is_bullied]
        non_bullied_wealth = [t.wealth for t in self.retail_traders if not t.is_bullied]
        
        return {
            "bullied_wealth_mean": np.mean(bullied_wealth) if bullied_wealth else 0,
            "non_bullied_wealth_mean": np.mean(non_bullied_wealth) if non_bullied_wealth else 0,
            "bullied_wealth_std": np.std(bullied_wealth) if bullied_wealth else 0,
            "non_bullied_wealth_std": np.std(non_bullied_wealth) if non_bullied_wealth else 0
        }
    
    def plot_analysis(self, save_path: str = None):
        """绘制分析结果"""
        plt.figure(figsize=(15, 10))
        
        # 1. 情绪影响
        plt.subplot(2, 2, 1)
        emotion_data = self.analyze_emotion_impact()
        plt.bar(["Bullied", "Non-Bullied"], 
                [emotion_data["bullied_emotion_mean"], emotion_data["non_bullied_emotion_mean"]],
                yerr=[emotion_data["bullied_emotion_std"], emotion_data["non_bullied_emotion_std"]])
        plt.title("Emotion Bias Comparison")
        plt.ylabel("Emotion Bias")
        
        # 2. 交易活跃度
        plt.subplot(2, 2, 2)
        activity_data = self.analyze_trading_activity(self.market.get_all_orders())
        plt.bar(["Bullied", "Non-Bullied"], 
                [activity_data["bullied_order_count"], activity_data["non_bullied_order_count"]])
        plt.title("Trading Activity Comparison")
        plt.ylabel("Number of Orders")
        
        # 3. 市场效率
        plt.subplot(2, 2, 3)
        efficiency_data = self.analyze_market_efficiency()
        price_history = self.market.get_price_history()
        fundamental_price = self.market.get_fundamental_price()
        plt.plot(price_history, label="Market Price")
        plt.axhline(y=fundamental_price, color='r', linestyle='--', label="Fundamental Price")
        plt.title("Price Evolution")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        
        # 4. 财富分布
        plt.subplot(2, 2, 4)
        wealth_data = self.analyze_wealth_distribution()
        plt.bar(["Bullied", "Non-Bullied"], 
                [wealth_data["bullied_wealth_mean"], wealth_data["non_bullied_wealth_mean"]],
                yerr=[wealth_data["bullied_wealth_std"], wealth_data["non_bullied_wealth_std"]])
        plt.title("Wealth Distribution")
        plt.ylabel("Average Wealth")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"results/cyberbullying/{save_path}")
        plt.close()
    
    def generate_report(self) -> str:
        """生成分析报告"""
        emotion_data = self.analyze_emotion_impact()
        activity_data = self.analyze_trading_activity(self.market.get_all_orders())
        efficiency_data = self.analyze_market_efficiency()
        wealth_data = self.analyze_wealth_distribution()
        
        report = f"""
Cyberbullying Impact Analysis Report
===================================

1. Emotion Impact
----------------
- Bullied traders: {emotion_data['bullied_count']}/{emotion_data['total_retail_count']} ({emotion_data['bullied_count']/emotion_data['total_retail_count']*100:.1f}%)
- Average emotion bias:
  * Bullied traders: {emotion_data['bullied_emotion_mean']:.3f} (±{emotion_data['bullied_emotion_std']:.3f})
  * Non-bullied traders: {emotion_data['non_bullied_emotion_mean']:.3f} (±{emotion_data['non_bullied_emotion_std']:.3f})

2. Trading Activity
------------------
- Orders from bullied traders: {activity_data['bullied_order_count']} ({activity_data['bullied_order_ratio']*100:.1f}% of total)
- Active bullied traders: {activity_data['bullied_traders_active_ratio']*100:.1f}% of bullied traders

3. Market Efficiency
-------------------
- Average price deviation: {efficiency_data['avg_price_deviation']*100:.2f}%
- Maximum price deviation: {efficiency_data['max_price_deviation']*100:.2f}%
- Market volatility: {efficiency_data['volatility']*100:.2f}%

4. Wealth Distribution
---------------------
- Average wealth:
  * Bullied traders: {wealth_data['bullied_wealth_mean']:.2f} (±{wealth_data['bullied_wealth_std']:.2f})
  * Non-bullied traders: {wealth_data['non_bullied_wealth_mean']:.2f} (±{wealth_data['non_bullied_wealth_std']:.2f})
"""
        return report
