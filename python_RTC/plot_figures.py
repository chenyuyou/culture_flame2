# plot_figures.py

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # 用于 linspace 获取参数值

# --- 配置 ---
RESULTS_DIR = "ensemble_results_cultural_pd" # 必须与 Ensemble.py 中的目录匹配
NUM_RUNS = 11        # 必须与 Ensemble.py 中的 ENSEMBLE_RUNS 匹配

# 参数范围（用于标签和绘图轴） - 必须与 Ensemble.py 中 define_runs 的设置匹配
T_PAYOFF_MIN = 1.0
T_PAYOFF_MAX = 2.0

# 生成 T_PAYOFF 值列表，对应于每个运行
# 使用 linspace 来精确匹配 LerpRangeFloat 的行为
T_PAYOFF_VALUES = np.linspace(T_PAYOFF_MIN, T_PAYOFF_MAX, NUM_RUNS)

OUTPUT_PLOT_DIR = "plots_cultural_pd" # 保存绘图的目录
if not os.path.exists(OUTPUT_PLOT_DIR):
    os.makedirs(OUTPUT_PLOT_DIR)

# --- 数据加载函数 ---
def load_run_data(run_index: int) -> pd.DataFrame:
    """加载指定运行索引的步骤日志数据"""
    # 格式化运行目录名称 (假设 FLAME GPU 使用 8 位零填充)
    run_dir_name = f"run_{run_index:08d}"
    run_path = os.path.join(RESULTS_DIR, run_dir_name)
    log_file = os.path.join(run_path, "log.json")

    if not os.path.exists(log_file):
        print(f"Warning: Log file not found for run {run_index} at {log_file}")
        return None

    try:
        with open(log_file, 'r') as f:
            data = json.load(f)

        # 将 JSON 数据转换为 Pandas DataFrame
        # 提取 'steps' 列表
        steps_data = data.get("steps", [])
        if not steps_data:
             print(f"Warning: No step data found in log file for run {run_index}")
             return None

        df = pd.json_normalize(steps_data) # 自动处理嵌套的 'environment'
        # 重命名列以简化访问（去掉 'environment.' 前缀）
        df.columns = [col.replace("environment.", "") for col in df.columns]
        # 确保核心列存在
        required_cols = ['step', 'average_C', 'cooperation_rate']
        if not all(col in df.columns for col in required_cols):
             print(f"Warning: Missing required columns in log file for run {run_index}")
             print(f"  Available columns: {df.columns.tolist()}")
             return None

        # 添加 T_PAYOFF 值到 DataFrame 中，以便后续分组或标记
        if run_index < len(T_PAYOFF_VALUES):
             df['T_PAYOFF'] = T_PAYOFF_VALUES[run_index]
        else:
             print(f"Warning: Run index {run_index} out of bounds for T_PAYOFF_VALUES.")
             df['T_PAYOFF'] = np.nan # 或者其他标记

        return df

    except Exception as e:
        print(f"Error loading or parsing log file for run {run_index}: {e}")
        return None

# --- 绘图函数 ---

def plot_time_evolution(all_dataframes):
    """绘制所有运行的 average_C 和 cooperation_rate 随时间的变化"""
    plt.figure(figsize=(12, 6))

    # 子图 1: Average C vs Step
    plt.subplot(1, 2, 1)
    for i, df in enumerate(all_dataframes):
        if df is not None and not df.empty:
            t_payoff = df['T_PAYOFF'].iloc[0] # 获取该运行的 T_PAYOFF
            plt.plot(df['step'], df['average_C'], label=f'T={t_payoff:.2f}', alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Average C Value")
    plt.title("Time Evolution of Average C")
    plt.grid(True)
    plt.legend(fontsize='small', ncol=2) # 调整图例

    # 子图 2: Cooperation Rate vs Step
    plt.subplot(1, 2, 2)
    for i, df in enumerate(all_dataframes):
        if df is not None and not df.empty:
            t_payoff = df['T_PAYOFF'].iloc[0]
            plt.plot(df['step'], df['cooperation_rate'], label=f'T={t_payoff:.2f}', alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Cooperation Rate")
    plt.title("Time Evolution of Cooperation Rate")
    plt.grid(True)
    # plt.legend(fontsize='small', ncol=2) # 图例已在左侧显示

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "time_evolution.png"))
    # plt.show() # 取消注释以在运行时显示绘图

def plot_final_state_vs_parameter(all_dataframes):
    """绘制最终状态 (average_C, cooperation_rate) 与 T_PAYOFF 的关系"""
    final_avg_C = []
    final_coop_rate = []
    valid_t_payoffs = []

    for i, df in enumerate(all_dataframes):
        if df is not None and not df.empty:
            # 获取最后一步的数据
            last_step_data = df.iloc[-1]
            final_avg_C.append(last_step_data['average_C'])
            final_coop_rate.append(last_step_data['cooperation_rate'])
            valid_t_payoffs.append(last_step_data['T_PAYOFF'])

    if not valid_t_payoffs:
        print("No valid final state data found to plot.")
        return

    plt.figure(figsize=(10, 5))

    # 子图 1: Final Average C vs T_PAYOFF
    plt.subplot(1, 2, 1)
    plt.plot(valid_t_payoffs, final_avg_C, marker='o', linestyle='-')
    plt.xlabel("Temptation Payoff (T_PAYOFF)")
    plt.ylabel("Final Average C Value")
    plt.title("Final Average C vs Temptation")
    plt.grid(True)

    # 子图 2: Final Cooperation Rate vs T_PAYOFF
    plt.subplot(1, 2, 2)
    plt.plot(valid_t_payoffs, final_coop_rate, marker='s', linestyle='--')
    plt.xlabel("Temptation Payoff (T_PAYOFF)")
    plt.ylabel("Final Cooperation Rate")
    plt.title("Final Cooperation Rate vs Temptation")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "final_state_vs_T_PAYOFF.png"))
    # plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Loading data from: {RESULTS_DIR}")
    all_run_data = []
    for i in range(NUM_RUNS):
        df = load_run_data(i)
        if df is not None:
            all_run_data.append(df)
        else:
             # 可以选择添加一个 None 占位符，或者跳过
             # all_run_data.append(None)
             pass # 跳过加载失败的运行

    if not all_run_data:
        print("Error: No data loaded successfully. Plots cannot be generated.")
        sys.exit(1)

    print(f"Successfully loaded data for {len(all_run_data)} runs.")

    print("Generating plots...")
    plot_time_evolution(all_run_data)
    plot_final_state_vs_parameter(all_run_data)
    print(f"Plots saved in: {OUTPUT_PLOT_DIR}")

