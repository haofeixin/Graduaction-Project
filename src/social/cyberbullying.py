import networkx as nx
import random
import numpy as np
import pandas as pd
from datetime import datetime

class CyberbullyingModel:
    def __init__(self, config, traders):
        self.config = config
        self.network = None
        self.bully_count = {}  # 记录每个trader被网暴次数
        self.attack_log = []   # 记录本步所有攻击 (attacker_id, victim_id, strength, timestep)
        self.timestep = 0
        self.traders = traders

    def build_network(self):
        # 只处理散户
        retail_traders = [t for t in self.traders if t.type == 'retail']
        n = len(retail_traders)
        avg_k = self.config["social"].get("average_degree", 4)
        net_type = self.config["social"].get("network_type", "small_world")
        seed = self.config["simulation"].get("random_seed", 42)
        p_born_bully = self.config["social"].get("born_bully_ratio", 0.1)

        random.seed(seed)
        np.random.seed(seed)

        if net_type == "small_world":
            G = nx.watts_strogatz_graph(n, k=avg_k, p=0.3)
        elif net_type == "random":
            G = nx.erdos_renyi_graph(n, p=avg_k / (n - 1))
        else:
            raise ValueError("Unsupported network_type")

        self.network = G

        for i, trader in enumerate(retail_traders):
            trader.neighbors = [retail_traders[j] for j in G.neighbors(i)]
            trader.exposure = 0.0
            trader.resilience = 0.0
            trader.is_bullied = False
            trader.bully_cooldown = 0
            trader.suppression = 0.0
            trader.last_attack_strength = 0.0
            trader.last_been_attacked = False
            trader.last_attackers = []
            self.bully_count[trader.trader_id] = 0
            
            # 初始化少数"天生喷子"
            if np.random.rand() < p_born_bully:
                trader.is_attacker = True
                sign = 1 if np.random.rand() < 0.5 else -1
                trader.emotion_bias = sign * np.random.uniform(0.1, 0.2)
            else:
                trader.is_attacker = False

    def propagate(self):
        if not self.config["social"].get("enable_cyberbullying", False):
            return
        self.timestep += 1
        exposure_threshold = self.config["social"].get("exposure_threshold", 0.01)
        cooldown_duration = self.config["social"].get("cooldown_duration", 3)
        suppression_base = self.config["social"].get("suppression_prob", 0.5)
        resilience_growth = self.config["social"].get("resilience_growth", 0.01)
        recovery_shrink = self.config["social"].get("exposure_shrink_factor", 0.9)
        # 正反馈参数
        enable_bully_infect = self.config["social"].get("enable_bully_infect", False)
        bully_infect_prob = self.config["social"].get("bully_infect_prob", 0.01)
        # 负反馈参数
        enable_regulator = self.config["social"].get("enable_regulator", False)
        regulator_interval = self.config["social"].get("regulator_interval", 10)
        regulator_cooldown = self.config["social"].get("regulator_cooldown", 10)
        enable_bully_report = self.config["social"].get("enable_bully_report", False)
        regulator_report_prob = self.config["social"].get("regulator_report_prob", 0.1)
        regulator_report_cooldown = self.config["social"].get("regulator_report_cooldown", 8)
        enable_bully_resilience = self.config["social"].get("enable_bully_resilience", True)
        bully_amplify = self.config["social"].get("bully_amplify", 1.5)
        self.attack_log = []
        # 先清空所有trader的攻击记录
        for trader in self.traders:
            trader.last_attack_strength = 0.0
            trader.last_been_attacked = False
            trader.last_attackers = []

        any_attack = False
        any_bullied = False

        for trader in self.traders:
            if not hasattr(trader, "neighbors"):
                continue
            # 冷却自动减1
            if hasattr(trader, "bully_cooldown"):
                trader.bully_cooldown = max(0, trader.bully_cooldown - 1)
            

            # 计算被攻击的强度
            attack_sum = 0
            attackers = []
            for neighbor in trader.neighbors:
                if (
                    getattr(neighbor, "is_attacker", False) and
                    getattr(neighbor, "bully_cooldown", 0) == 0 and
                    np.sign(neighbor.emotion_bias) != np.sign(trader.emotion_bias) and
                    abs(neighbor.emotion_bias) > 1e-3
                ):
                    opinion_diff = abs(neighbor.emotion_bias - trader.emotion_bias)
                    attack_strength = opinion_diff
                    attack_sum += attack_strength
                    neighbor.bully_cooldown = cooldown_duration
                    neighbor.last_attack_strength = attack_strength
                    # self.attack_log.append((neighbor.trader_id, trader.trader_id, attack_strength, self.timestep))
                    # attackers.append(neighbor.trader_id)
                    any_attack = True

            # if attack_sum == 0:
            #     trader.exposure *= recovery_shrink

            trader.last_been_attacked = attack_sum > 0
            trader.last_attackers = attackers
            # 累计攻击强度
            effective_exposure = attack_sum * (1.0 - trader.resilience)
            trader.exposure += effective_exposure
            # 受网暴影响
            if trader.exposure >= exposure_threshold:
                if not trader.is_bullied:
                    self.bully_count[trader.trader_id] += 1
                trader.is_bullied = True
                any_bullied = True
                # suppression等逻辑不变
                k = self.config["social"].get("sigmoid_k", 1.0)
                max_suppression = self.config["social"].get("max_suppression", 0.95)
                base_suppression = suppression_base
                x = trader.exposure - exposure_threshold
                sigmoid = 1 / (1 + np.exp(-k * x))
                trader.suppression = base_suppression + (max_suppression - base_suppression) * sigmoid
                shrink_factor = self.config["social"].get("emotion_shrink_factor", 0.8)
                trader.emotion_bias *= shrink_factor
                # 正反馈：网暴感染
                if enable_bully_infect and not trader.is_attacker:
                    if np.random.rand() < bully_infect_prob:
                        trader.is_attacker = True
                        trader.emotion_bias *= bully_amplify
            else:
                trader.is_bullied = False
                trader.suppression = 0.0

            # 负反馈3：心理承受能力提升
            if enable_bully_resilience and trader.is_bullied:
                trader.resilience = min(1.0, trader.resilience + resilience_growth)
                
        # 负反馈1：监管者巡视
        if enable_regulator and self.timestep % regulator_interval == 0:
            for attacker_id, victim_id, strength, timestep in self.attack_log:
                attacker = next((t for t in self.traders if t.trader_id == attacker_id), None)
                if attacker:
                    attacker.bully_cooldown += int(regulator_cooldown * strength)

        # 负反馈2：举报
        if enable_bully_report:
            for trader in self.traders:
                if trader.is_bullied and trader.last_attackers:
                    for attacker_id in trader.last_attackers:
                        if np.random.rand() < regulator_report_prob:
                            attacker = next((t for t in self.traders if t.trader_id == attacker_id), None)
                            if attacker:
                                attacker.bully_cooldown += regulator_report_cooldown

        # print(f"[Cyberbullying] Any attack: {any_attack}, Any bullied: {any_bullied}, Timestep: {self.timestep}")

    def export_bully_count(self, path="results/cyberbullying/网暴次数.xlsx"):
        df = pd.DataFrame(list(self.bully_count.items()), columns=["trader_id", "bully_count"])
        df = df.sort_values("trader_id")
        df.to_excel(path, index=False)


