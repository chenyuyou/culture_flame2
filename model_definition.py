# agent_config.py (或者放在主脚本的开头部分)

import pyflamegpu

# -------------------------------------
# 环境属性 (Environment Properties)
# -------------------------------------
# 将 MESA 的 model 参数映射到 FlameGPU environment properties

# Payoff Matrix Values (R, S, T, P for Prisoner's Dilemma)
# CC: R=1, CD: S=0, DC: T=b, DD: P=0
R_PAYOFF = 1.0
S_PAYOFF = 0.0
# T_PAYOFF (b) 和 P_PAYOFF (0) 会作为可变参数传入
# P_PAYOFF = 0.0 # P 通常为 0

# Fermi Rule Parameters
# K: Strategy update noise
# K_C: Culture update noise

# Update Probabilities
# p_update_C: Probability to attempt cultural update
# p_mut_culture: Probability of random cultural mutation
# p_mut_strategy: Probability of random strategy mutation

# Utility Calculation Parameter C is agent-specific, not environment

# Helper Function to add environment properties to the model description
def define_environment(model: pyflamegpu.ModelDescription):
    """向模型描述中添加环境属性定义"""
    env = model.Environment()
    # Payoffs (T is the variable 'b')
    env.newProperty("T_PAYOFF", pyflamegpu.float_, 1.5) # Example default for 'b'
    env.newProperty("R_PAYOFF", pyflamegpu.float_, R_PAYOFF)
    env.newProperty("S_PAYOFF", pyflamegpu.float_, S_PAYOFF)
    env.newProperty("P_PAYOFF", pyflamegpu.float_, 0.0) # Explicitly define P=0

    # Noise parameters
    env.newProperty("K", pyflamegpu.float_, 0.1)        # Strategy noise
    env.newProperty("K_C", pyflamegpu.float_, 0.1)       # Cultural noise

    # Probabilities
    env.newProperty("p_update_C", pyflamegpu.float_, 0.1) # Prob. attempt culture update
    env.newProperty("p_mut_culture", pyflamegpu.float_, 0.01) # Prob. culture mutation
    env.newProperty("p_mut_strategy", pyflamegpu.float_, 0.001)# Prob. strategy mutation

    # Grid size (useful for potential boundary checks, though torus is default)
    env.newProperty("grid_size", pyflamegpu.uint32, 50) # Example grid size L

    # Threshold for neighbor random selection (if needed)
    # env.newProperty("selection_threshold", pyflamegpu.float_, 1.0 / 8.0) # Example if selecting one

    print("Environment properties defined.")


# -------------------------------------
# 代理变量 (Agent Variables)
# -------------------------------------

# Maximum number of neighbors (Moore neighborhood = 8)
MAX_NEIGHBORS = 8

def define_cultural_agent(model: pyflamegpu.ModelDescription):
    """向模型描述中添加 CulturalAgent 的定义"""
    agent = model.newAgent("CulturalAgent")

    # Core State Variables (与 MESA 对应)
    agent.newVariable("id", pyflamegpu.id_t) # Agent ID (handled by FlameGPU)
    agent.newVariable("strategy", pyflamegpu.int32) # 0 (Defect) or 1 (Cooperate)
    agent.newVariable("C", pyflamegpu.float_)       # Cultural parameter [0, 1]
    agent.newVariable("utility", pyflamegpu.float_) # Current utility, calculated each step

    # Temporary Variables for Staged Updates (类似 MESA 的 next_*)
    agent.newVariable("next_strategy", pyflamegpu.int32)
    agent.newVariable("next_C", pyflamegpu.float_)

    # Variables to Store Neighbor Information (用于决策)
    # We need to store info about neighbors received via messages to make decisions
    agent.newVariable("neighbor_count", pyflamegpu.uint32) # Number of neighbors found in the last message iteration
    # Arrays to store data from *up to* MAX_NEIGHBORS neighbors
    # Note: FlameGPU Array Variables require fixed sizes defined at compile time.
    agent.newVariableArray("neighbor_strategies", MAX_NEIGHBORS, pyflamegpu.int32)
    agent.newVariableArray("neighbor_Cs", MAX_NEIGHBORS, pyflamegpu.float_)
    agent.newVariableArray("neighbor_utilities", MAX_NEIGHBORS, pyflamegpu.float_)

    # Internal position variables (FlameGPU often manages these, but we can declare if needed)
    # agent.newVariable("x", pyflamegpu.float_)
    # agent.newVariable("y", pyflamegpu.float_)
    # Let's assume FlameGPU's default spatial handling works for now via messaging radius.

    print("Agent 'CulturalAgent' defined with variables.")

# -------------------------------------
# 消息定义 (Message Definitions)
# -------------------------------------
# Define messages agents will use to communicate state and utility

def define_messages(model: pyflamegpu.ModelDescription):
    """向模型描述中添加消息定义"""

    # Message to broadcast basic state (Strategy and C) for utility calculation
    # We don't strictly need 'id' here if we only use neighbor state for *our* utility
    msg_state = model.newMessageSpatial2D("StateMessage")
    msg_state.newVariable("id", pyflamegpu.id_t) # Sender's ID (useful for debugging/tracking)
    msg_state.newVariable("strategy", pyflamegpu.int32)
    msg_state.newVariable("C", pyflamegpu.float_)
    # Define interaction radius (e.g., sqrt(2) for Moore neighborhood on integer grid)
    msg_state.setRadius(1.5) # A bit more than sqrt(2) to ensure corners are included
    # msg_state.setMinLocation([0.0, 0.0]) # Define grid boundaries if not infinite
    # msg_state.setMaxLocation([50.0, 50.0]) # Match environment grid_size

    # Message to broadcast calculated utility along with state for decision making
    msg_utility = model.newMessageSpatial2D("UtilityMessage")
    msg_utility.newVariable("id", pyflamegpu.id_t)
    msg_utility.newVariable("strategy", pyflamegpu.int32)
    msg_utility.newVariable("C", pyflamegpu.float_)
    msg_utility.newVariable("utility", pyflamegpu.float_)
    msg_utility.setRadius(1.5) # Same radius
    # msg_utility.setMinLocation([0.0, 0.0])
    # msg_utility.setMaxLocation([50.0, 50.0])

    print("Messages 'StateMessage' and 'UtilityMessage' defined.")

# Example Usage (how you would use these functions)
# model_desc = pyflamegpu.ModelDescription("Cultural_Prisoners_Dilemma")
# define_environment(model_desc)
# define_cultural_agent(model_desc)
# define_messages(model_desc)
