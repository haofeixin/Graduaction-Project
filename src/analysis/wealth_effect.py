import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel
from src.analysis.experiment_utils import run_scenario
from src.config.loader import load_config
def get_final_wealth(agent):
    return agent.cash + agent.stock * agent.config['fundamental_price']

def get_group_wealth(agents, group):
    return [get_final_wealth(a) for a in agents if a.type == group]

def gini_coefficient(wealths):
    wealths = np.sort(wealths)
    n = len(wealths)
    index = np.arange(1, n + 1)
    return ((2 * index - n - 1) * wealths).sum() / (n * wealths.sum())

def run_wealth_effect_analysis_multi(baseline_agents_runs, cyber_agents_runs, save_dir='thesis/image'):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    # 1. 网暴对散户财富影响是否显著
    retail_baseline = []
    retail_cyber = []
    for base_agents, cyber_agents in zip(baseline_agents_runs, cyber_agents_runs):
        retail_baseline.append(np.mean([get_final_wealth(a) for a in base_agents if a.type == 'retail']))
        retail_cyber.append(np.mean([get_final_wealth(a) for a in cyber_agents if a.type == 'retail']))
    t_stat_retail, p_val_retail = ttest_rel(retail_baseline, retail_cyber)
    results.append({
        'group': '散户均值', 
        't_stat': t_stat_retail, 
        'p_val': p_val_retail,
        'baseline_mean': np.mean(retail_baseline),
        'cyber_mean': np.mean(retail_cyber)
    })
    print("配对t检验：散户财富影响分析完成")

    # 2. 网暴对机构财富影响是否显著
    inst_baseline = []
    inst_cyber = []
    for base_agents, cyber_agents in zip(baseline_agents_runs, cyber_agents_runs):
        inst_baseline.append(np.mean([get_final_wealth(a) for a in base_agents if a.type == 'institutional']))
        inst_cyber.append(np.mean([get_final_wealth(a) for a in cyber_agents if a.type == 'institutional']))
    t_stat_inst, p_val_inst = ttest_rel(inst_baseline, inst_cyber)
    results.append({
        'group': '机构均值', 
        't_stat': t_stat_inst, 
        'p_val': p_val_inst,
        'baseline_mean': np.mean(inst_baseline),
        'cyber_mean': np.mean(inst_cyber)
    })
    print("配对t检验：机构财富影响分析完成")

    # 3. 被攻击/未被攻击散户
    attacked_baseline, attacked_cyber = [], []
    not_attacked_baseline, not_attacked_cyber = [], []
    for base_agents, cyber_agents in zip(baseline_agents_runs, cyber_agents_runs):
        attacked_cyber.append(np.mean([get_final_wealth(a) for a in cyber_agents if a.type == 'retail' and getattr(a, 'is_bullied', False)]))
        not_attacked_baseline.append(np.mean([get_final_wealth(a) for a in base_agents if a.type == 'retail' and not getattr(a, 'is_bullied', False)]))
        not_attacked_cyber.append(np.mean([get_final_wealth(a) for a in cyber_agents if a.type == 'retail' and not getattr(a, 'is_bullied', False)]))
    t_stat_attacked, p_val_attacked = ttest_rel(not_attacked_baseline, attacked_cyber)
    t_stat_not_attacked, p_val_not_attacked = ttest_rel(not_attacked_baseline, not_attacked_cyber)
    results.append({
        'group': '被攻击散户均值', 
        't_stat': t_stat_attacked, 
        'p_val': p_val_attacked,
        'baseline_mean': np.mean(attacked_baseline),
        'cyber_mean': np.mean(attacked_cyber)
    })
    results.append({
        'group': '未被攻击散户均值', 
        't_stat': t_stat_not_attacked, 
        'p_val': p_val_not_attacked,
        'baseline_mean': np.mean(not_attacked_baseline),
        'cyber_mean': np.mean(not_attacked_cyber)
    })
    print("配对t检验：被攻击/未被攻击散户财富影响分析完成")

    # 4. 散户基尼系数
    gini_baseline, gini_cyber = [], []
    for base_agents, cyber_agents in zip(baseline_agents_runs, cyber_agents_runs):
        gini_baseline.append(gini_coefficient([get_final_wealth(a) for a in base_agents if a.type == 'retail']))
        gini_cyber.append(gini_coefficient([get_final_wealth(a) for a in cyber_agents if a.type == 'retail']))
    t_stat_gini, p_val_gini = ttest_rel(gini_baseline, gini_cyber)
    results.append({
        'group': '散户基尼系数', 
        't_stat': t_stat_gini, 
        'p_val': p_val_gini,
        'baseline_mean': np.mean(gini_baseline),
        'cyber_mean': np.mean(gini_cyber)
    })
    print("配对t检验：散户基尼系数分析完成")

    # 保存t检验结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, 'table4_2_wealth_ttest_results.csv'), index=False)

    # 使用最后一次实验的结果生成可视化图表
    first_baseline = baseline_agents_runs[0]
    first_cyber = cyber_agents_runs[0]

    # 1. 散户财富分布对比图
    plt.figure(figsize=(10, 6))
    retail_baseline_wealth = [get_final_wealth(a) for a in first_baseline if a.type == 'retail']
    retail_cyber_wealth = [get_final_wealth(a) for a in first_cyber if a.type == 'retail']
    plt.boxplot([retail_baseline_wealth, retail_cyber_wealth], labels=['Baseline-散户', 'Cyber-散户'])
    plt.ylabel('Final Wealth')
    plt.title('散户终期财富分布对比')
    plt.savefig(os.path.join(save_dir, 'fig4_5_final_wealth_boxplot_retail.png'))
    plt.close()

    # 2. 机构财富分布对比图
    plt.figure(figsize=(10, 6))
    inst_baseline_wealth = [get_final_wealth(a) for a in first_baseline if a.type == 'institutional']
    inst_cyber_wealth = [get_final_wealth(a) for a in first_cyber if a.type == 'institutional']
    plt.boxplot([inst_baseline_wealth, inst_cyber_wealth], labels=['Baseline-机构', 'Cyber-机构'])
    plt.ylabel('Final Wealth')
    plt.title('机构终期财富分布对比')
    plt.savefig(os.path.join(save_dir, 'fig4_5_final_wealth_boxplot_institutional.png'))
    plt.close()

    # 3. 被攻击/未被攻击散户财富分布对比图
    plt.figure(figsize=(10, 6))
    baseline_retail = [get_final_wealth(a) for a in first_baseline if a.type == 'retail']
    attacked_cyber = [get_final_wealth(a) for a in first_cyber if a.type == 'retail' and getattr(a, 'is_bullied', False)]
    not_attacked_cyber = [get_final_wealth(a) for a in first_cyber if a.type == 'retail' and not getattr(a, 'is_bullied', False)]
    plt.boxplot([baseline_retail, attacked_cyber, not_attacked_cyber],
                labels=['Baseline-散户', 'Cyber-被攻击散户', 'Cyber-未被攻击散户'])
    plt.ylabel('Final Wealth')
    plt.title('散户财富分布对比（Baseline vs Cyber-被攻击/未被攻击）')
    plt.savefig(os.path.join(save_dir, 'fig4_6_attacked_vs_not_boxplot.png'))
    plt.close()

    # 4. 基尼系数对比图
    plt.figure(figsize=(8, 6))
    plt.boxplot([gini_baseline, gini_cyber], labels=['Baseline', 'Cyberbullying'])
    plt.ylabel('Gini Coefficient')
    plt.title('散户内部不平等性（基尼系数）多次实验分布')
    plt.savefig(os.path.join(save_dir, 'fig4_7_gini_comparison_boxplot.png'))
    plt.close()

    print(f'4.2多次实验财富影响分析结果已生成，保存在{save_dir}')

def runner_4_2():
    # 4.2 多次实验
    baseline_agents_runs = []
    cyber_agents_runs = []
    config = load_config()
    for seed in range(config['simulation']['n_simulations']):  # 例如20次实验
        baseline_agents0, baseline_agents1, _, _, _ = run_scenario(False, seed=seed)
        cyber_agents0, cyber_agents1, _, _, _ = run_scenario(True, seed=seed)
        baseline_agents_runs.append(baseline_agents1)
        cyber_agents_runs.append(cyber_agents1)
        print(f"第{seed}次实验完成")
    run_wealth_effect_analysis_multi(baseline_agents_runs, cyber_agents_runs)