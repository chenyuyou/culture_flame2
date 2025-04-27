# Ensemble.py

import pyflamegpu
import sys
import time
import os
import math # 可能需要 math.ceil

# --- 导入模型定义 ---
try:
    # model_definition 现在定义了包含 RTC 函数和新增环境属性的模型结构
    from model_definition import define_model, MAX_NEIGHBORS
except ImportError as e:
    print(f"Error importing from model_definition: {e}")
    print("Ensure model_definition.py and cuda_kernels.py are accessible.")
    sys.exit(1)

# --- 模拟参数 ---
# 这些可以在 define_runs 中覆盖或用作默认值
SIMULATION_STEPS = 5000  # 每次运行的步数 (根据需要调整)
LOGGING_FREQUENCY = 50   # 日志记录频率
RANDOM_SEED_START = 1234 # 随机种子起始值
ENSEMBLE_RUNS = 11       # 要运行的模拟总次数 (例如，改变 T_PAYOFF 11 次)

# --- 输出配置 ---
RESULTS_DIR = "ensemble_results_cultural_pd" # 输出目录

# --- Host Function for Step Logging Aggregation ---
# 这个函数在每个(或指定的)仿真步骤结束时在主机端执行
class AggregateStepFunction(pyflamegpu.HostFunction):
    """
    计算并更新环境属性中的 average_C 和 cooperation_rate 以供日志记录。
    """
    def run(self, FLAMEGPU):
        # 获取 CulturalAgent 代理状态列表
        # 注意: 这可能会有性能开销，特别是在大量 Agent 时。
        # 仅在需要记录的步骤运行此函数是理想的（通过日志频率控制间接实现）
        agent_pop = FLAMEGPU.agent("CulturalAgent")

        # 获取 Agent 总数
        agent_count = len(agent_pop)
        FLAMEGPU.environment.setPropertyUInt32("agent_count", agent_count)

        if agent_count == 0:
            FLAMEGPU.environment.setPropertyFloat("average_C", 0.0)
            FLAMEGPU.environment.setPropertyFloat("cooperation_rate", 0.0)
            return

        # 计算 C 值总和与合作者数量
        sum_C = 0.0
        cooperator_count = 0
        for agent_state in agent_pop: # 遍历代理状态
            sum_C += agent_state.getVariableFloat("C")
            cooperator_count += agent_state.getVariableInt32("strategy") # strategy=1 为合作

        # 计算平均值和比率
        average_C = sum_C / agent_count
        cooperation_rate = float(cooperator_count) / agent_count

        # 更新环境属性
        FLAMEGPU.environment.setPropertyFloat("average_C", average_C)
        FLAMEGPU.environment.setPropertyFloat("cooperation_rate", cooperation_rate)

# --- Ensemble Configuration Functions ---

def define_output_config(ensemble: pyflamegpu.CUDAEnsemble):
    """配置 Ensemble 的输出设置"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    config = ensemble.Config()
    config.out_directory = RESULTS_DIR
    config.out_format = "json"  # JSON 通常比 CSV 更容易解析复杂结构
    config.concurrent_runs = 4  # 根据 GPU 能力调整并发运行数
    config.timing = True        # 记录运行时间
    config.truncate_log_files = True # 覆盖旧日志
    # config.error_level = pyflamegpu.CUDAEnsembleConfig.Fast # 更快的错误检查（可能牺牲一些调试信息）
    # config.devices = pyflamegpu.IntSet([0]) # 指定使用的 GPU 设备 ID

def define_run_plans(model: pyflamegpu.ModelDescription) -> pyflamegpu.RunPlanVector:
    """定义要执行的运行计划，改变 T_PAYOFF"""
    runs = pyflamegpu.RunPlanVector(model, ENSEMBLE_RUNS)
    runs.setSteps(SIMULATION_STEPS)
    # 为每次运行设置不同的随机种子
    runs.setRandomSimulationSeed(RANDOM_SEED_START, 1) # 种子从 START 开始，每次递增 1

    # 定义 T_PAYOFF (b) 的变化范围
    t_payoff_min = 1.0
    t_payoff_max = 2.0 # Prisoner's Dilemma 要求 T > R (R=1.0)
    # 使用 LerpRange (线性插值) 来设置每次运行的 T_PAYOFF 值
    runs.setPropertyLerpRangeFloat("T_PAYOFF", t_payoff_min, t_payoff_max)

    # --- (可选) 改变其他参数 ---
    # 例如，同时改变 K 值 (创建嵌套循环或更复杂的 RunPlan)
    # k_values = [0.01, 0.1, 0.5]
    # total_runs = ENSEMBLE_RUNS * len(k_values)
    # runs = pyflamegpu.RunPlanVector(model, total_runs)
    # runs.setSteps(SIMULATION_STEPS)
    # run_index = 0
    # for i in range(ENSEMBLE_RUNS):
    #     t_payoff = t_payoff_min + (t_payoff_max - t_payoff_min) * i / (ENSEMBLE_RUNS - 1)
    #     for k_val in k_values:
    #         run = runs[run_index]
    #         run.setRandomSimulationSeed(RANDOM_SEED_START + run_index)
    #         run.setPropertyFloat("T_PAYOFF", t_payoff)
    #         run.setPropertyFloat("K", k_val)
    #         # run.setPropertyFloat("K_C", k_val) # 如果 K 和 K_C 一起变
    #         run_index += 1
    # ---

    print(f"Created {len(runs)} run plans.")
    # 打印第一个和最后一个运行的 T_PAYOFF 值以验证范围
    if len(runs) > 0:
         print(f"  Run 0 T_PAYOFF: {runs[0].getPropertyFloat('T_PAYOFF'):.4f}")
    if len(runs) > 1:
         print(f"  Run {len(runs)-1} T_PAYOFF: {runs[len(runs)-1].getPropertyFloat('T_PAYOFF'):.4f}")

    return runs

def define_step_logging(model: pyflamegpu.ModelDescription) -> pyflamegpu.StepLoggingConfig:
    """配置步骤日志，记录聚合的环境属性"""
    log_config = pyflamegpu.StepLoggingConfig(model)
    log_config.setFrequency(LOGGING_FREQUENCY)

    # 记录我们通过 AggregateStepFunction 更新的环境属性
    log_config.logEnvironment("average_C")
    log_config.logEnvironment("cooperation_rate")
    log_config.logEnvironment("agent_count") # 记录 Agent 数量以供参考

    # (可选) 记录被改变的参数，以便在日志文件中直接看到
    log_config.logEnvironment("T_PAYOFF")
    log_config.logEnvironment("K") # 如果也改变了 K

    return log_config

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    print("1. Defining Model...")
    # 获取基础模型描述 (不含 step/init functions)
    model = define_model()

    print("2. Defining Ensemble Runs...")
    run_plans = define_run_plans(model)

    # --- 添加 Host Functions 到模型描述 ---
    # AggregateStepFunction 需要在 simulate 之前添加到模型中
    # 注意：如果模型本身不需要 InitFunction，则无需添加
    print("3. Adding Host Step Function for Logging...")
    model.addStepFunction(AggregateStepFunction())
    # model.addInitFunction(...) # 如果有初始化 Host Function

    print("4. Creating CUDAEnsemble...")
    # 将包含 Host Function 的模型传递给 Ensemble
    ensemble = pyflamegpu.CUDAEnsemble(model)

    print("5. Configuring Output...")
    define_output_config(ensemble)

    print("6. Configuring Step Logging...")
    log_config = define_step_logging(model)
    ensemble.setStepLog(log_config)

    # --- 运行 Ensemble ---
    print(f"7. Starting Ensemble Simulation ({len(run_plans)} runs)...")
    ensemble.simulate(run_plans)
    print("--- Ensemble Simulation Finished ---")

    # --- 清理 (可选) ---
    # pyflamegpu.cleanup() # 通常自动处理

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    print(f"Results saved in: {os.path.abspath(RESULTS_DIR)}")
    exit()
