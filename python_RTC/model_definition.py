# model_definition.py (修正版)

import pyflamegpu

# --- Import C++ Kernels ---
# cuda_kernels.py should be in the same directory or Python path
try:
    from cuda_kernels import (
        output_state_cpp,
        calculate_utility_cpp,
        output_utility_cpp,
        process_neighbor_utility_cpp,
        decide_strategy_update_cpp,
        decide_culture_update_cpp,
        mutate_cpp,
        advance_cpp
    )
except ImportError as e:
    print(f"Error importing C++ kernels from cuda_kernels.py: {e}")
    print("Ensure cuda_kernels.py exists and is accessible.")
    import sys
    sys.exit(1)


# -------------------------------------
# 环境属性 (Environment Properties)
# -------------------------------------
# Payoff Matrix Values (Defaults, can be overridden in run_simulation)
R_PAYOFF = 1.0
S_PAYOFF = 0.0
# P_PAYOFF = 0.0 (Defined below)

# Maximum number of neighbors (Moore neighborhood = 8)
# This MUST match the value used/assumed in the C++ kernels if hardcoded there
MAX_NEIGHBORS = 8

def define_environment(model: pyflamegpu.ModelDescription):
    """向模型描述中添加环境属性定义"""
    env = model.Environment()
        # 新增: 用于日志记录的聚合环境属性
    env.newPropertyFloat("average_C", 0.0)         # 存储当前步骤的平均 C 值
    env.newPropertyFloat("cooperation_rate", 0.0) # 存储当前步骤的合作率 (strategy=1 的比例)
    env.newPropertyUInt32("agent_count", 0)       # 存储当前步骤的 Agent 总数 (用于验证)

    # Payoff Matrix Parameters
    # Default values, can be overridden by run script
    env.newPropertyFloat("T_PAYOFF", 1.5) # Temptation 'b'
    env.newPropertyFloat("R_PAYOFF", R_PAYOFF)
    env.newPropertyFloat("S_PAYOFF", S_PAYOFF)
    env.newPropertyFloat("P_PAYOFF", 0.0) # Punishment

    # Fermi Rule Parameters (Noise)
    env.newPropertyFloat("K", 0.1)        # Strategy noise
    env.newPropertyFloat("K_C", 0.01)      # Culture noise

    # Update and Mutation Probabilities
    env.newPropertyFloat("p_update_C", 0.1) # P以概率p_update_C决定是否尝试更新
    # Renamed mutation rates for clarity to match C++ kernel usage
    env.newPropertyFloat("mutation_rate_strategy", 0.01) # 以p_mut_culture的概率对文化值C进行随机突变。
    env.newPropertyFloat("mutation_rate_C", 0.001)     # 以p_mut_strategy的概率对策略进行随机突变。

    # Grid size (Optional, if needed by functions, e.g., for visualization bounds)
    env.newPropertyUInt32("grid_width", 50)
    env.newPropertyUInt32("grid_height", 50)

    # Interaction radius (already defined in message)
    # env.newPropertyFloat("interaction_radius", 1.5)

# -------------------------------------
# 代理定义 (Agent Definition)
# -------------------------------------
def define_cultural_agent(model: pyflamegpu.ModelDescription):
    """向模型描述中添加 CulturalAgent 的定义，并附加 C++ RTC 函数"""
    agent = model.newAgent("CulturalAgent")

    # --- Agent Variables ---
    # Core State Variables
    agent.newVariableInt32("strategy")      # Current strategy (0=Defect, 1=Cooperate)
    agent.newVariableFloat("C")             # Cultural parameter [0, 1]
    agent.newVariableFloat("utility")       # Calculated utility in current step

    # Temporary Variables for Staged Updates
    agent.newVariableInt32("next_strategy") # Strategy chosen for the *next* step
    agent.newVariableFloat("next_C")        # C value chosen for the *next* step

    # Variables to Store Neighbor Information (for decision making)
    agent.newVariableUInt32("neighbor_count") # Actual number of neighbors processed
    # Arrays store data from up to MAX_NEIGHBORS neighbors
    agent.newVariableArrayInt32("neighbor_strategies", MAX_NEIGHBORS)  # 邻居的策略
    agent.newVariableArrayFloat("neighbor_Cs", MAX_NEIGHBORS)           # 邻居的文化
    agent.newVariableArrayFloat("neighbor_utilities", MAX_NEIGHBORS)    # 邻居的效用

    # Position Variables (Implicitly used by Spatial Messages/CUDA simulation)
    # Define them if explicit access/setting is needed or for visualization
    agent.newVariableFloat("x")                 # 所在空间坐标x轴
    agent.newVariableFloat("y")                 # 所在空间坐标y轴

    # --- Attach C++ Agent Functions (RTC - Run-Time Compiled) ---
    # These names ("output_state", "calculate_utility", etc.) will be used
    # in the layer definitions in run_simulation.py
    # The second argument is the C++ source code string from cuda_kernels.py
    agent.newRTCFunction("output_state", output_state_cpp).setMessageOutput("StateMessage")
    agent.newRTCFunction("calculate_utility", calculate_utility_cpp).setMessageInput("StateMessage")
    agent.newRTCFunction("output_utility", output_utility_cpp).setMessageOutput("UtilityMessage")
    agent.newRTCFunction("process_neighbor_utility", process_neighbor_utility_cpp).setMessageInput("UtilityMessage")
    agent.newRTCFunction("decide_strategy_update", decide_strategy_update_cpp)
    agent.newRTCFunction("decide_culture_update", decide_culture_update_cpp)
    agent.newRTCFunction("mutate", mutate_cpp)
    agent.newRTCFunction("advance", advance_cpp)

# -------------------------------------
# 消息定义 (Message Definitions)
# -------------------------------------
def define_messages(model: pyflamegpu.ModelDescription):
    """向模型描述中添加消息定义"""

    # Interaction radius - crucial for Spatial messages
    interaction_radius = 1.5 # sqrt(1^2 + 1^2) = sqrt(2) = 1.414. 1.5 ensures corners are included.

    # Message for broadcasting state (strategy, C) for utility calculation
    msg_state = model.newMessageSpatial2D("StateMessage")
    msg_state.newVariableID("id") # Sender's ID (avoids self-interaction calculation)
    msg_state.newVariableInt32("strategy")
    msg_state.newVariableFloat("C")
    msg_state.setRadius(interaction_radius)
    # Optional: Set environment bounds if not using wrap-around (torus)
    # min_bound = 0.0
    # max_bound_x = model.Environment().getPropertyFloat("grid_width") # Needs environment defined first
    # max_bound_y = model.Environment().getPropertyFloat("grid_height")
    # msg_state.setMin(min_bound, min_bound)
    # msg_state.setMax(max_bound_x, max_bound_y)

    # Message for broadcasting calculated utility + state for decision making
    msg_utility = model.newMessageSpatial2D("UtilityMessage")
    msg_utility.newVariableID("id") # Sender's ID (useful for debugging, avoids self-comparison)
    msg_utility.newVariableInt32("strategy")
    msg_utility.newVariableFloat("C")
    msg_utility.newVariableFloat("utility")
    msg_utility.setRadius(interaction_radius)
    # msg_utility.setMin(min_bound, min_bound)
    # msg_utility.setMax(max_bound_x, max_bound_y)

# -------------------------------------
# Convenience function to build the model
# -------------------------------------
def define_model() -> pyflamegpu.ModelDescription:
    """创建并配置完整的 FLAME GPU 模型描述"""
    model = pyflamegpu.ModelDescription("Cultural_Prisoners_Dilemma_RTC")
    define_environment(model)
    define_messages(model) # Messages need to be defined before agent functions that use them
    define_cultural_agent(model) # Agent definition now includes RTC function bindings
    # No need to define execution order here, done in run_simulation.py
    return model
