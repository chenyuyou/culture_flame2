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
def define_environment(model, pyflamegpu.ModelDescription):
    """向模型描述中添加环境属性定义"""
    env = model.Environment()
    # 收益矩阵参数
    env.newPropertyFloat("T_PAYOFF", 1.5) # Example default for 'b'
    env.newPropertyFloat("R_PAYOFF", R_PAYOFF) # 修正 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html&quot;,&quot;title&quot;:&quot;Defining Agent Functions — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;Agent Functions can be specified as C++ functions, built at compile time when using the C++ API, or they can be specified as Run-Time Compiled (RTC) function strings when using the both the C++ and Py&quot;}'>1</sup>](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html)
    env.newPropertyFloat("S_PAYOFF", S_PAYOFF) # 修正 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html&quot;,&quot;title&quot;:&quot;Defining Agent Functions — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;Agent Functions can be specified as C++ functions, built at compile time when using the C++ API, or they can be specified as Run-Time Compiled (RTC) function strings when using the both the C++ and Py&quot;}'>1</sup>](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html)
    env.newPropertyFloat("P_PAYOFF", 0.0) # Explicitly define P=0 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html&quot;,&quot;title&quot;:&quot;Defining Agent Functions — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;Agent Functions can be specified as C++ functions, built at compile time when using the C++ API, or they can be specified as Run-Time Compiled (RTC) function strings when using the both the C++ and Py&quot;}'>1</sup>](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html)

    # 噪声参数
    env.newPropertyFloat("K", 0.1)        # 策略噪声 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html&quot;,&quot;title&quot;:&quot;Defining Agent Functions — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;Agent Functions can be specified as C++ functions, built at compile time when using the C++ API, or they can be specified as Run-Time Compiled (RTC) function strings when using the both the C++ and Py&quot;}'>1</sup>](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html)
    env.newPropertyFloat("K_C", 0.1)      # 文化噪声 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html&quot;,&quot;title&quot;:&quot;Defining Agent Functions — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;Agent Functions can be specified as C++ functions, built at compile time when using the C++ API, or they can be specified as Run-Time Compiled (RTC) function strings when using the both the C++ and Py&quot;}'>1</sup>](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html)

    # Probabilities更新概率
    env.newPropertyFloat("p_update_C", 0.1) # 以概率p_update_C决定是否尝试更新 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html&quot;,&quot;title&quot;:&quot;Defining Agent Functions — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;Agent Functions can be specified as C++ functions, built at compile time when using the C++ API, or they can be specified as Run-Time Compiled (RTC) function strings when using the both the C++ and Py&quot;}'>1</sup>](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html)
    env.newPropertyFloat("p_mut_culture", 0.01) # 以p_mut_culture的概率对文化值C进行随机突变。 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html&quot;,&quot;title&quot;:&quot;Defining Agent Functions — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;Agent Functions can be specified as C++ functions, built at compile time when using the C++ API, or they can be specified as Run-Time Compiled (RTC) function strings when using the both the C++ and Py&quot;}'>1</sup>](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html)
    env.newPropertyFloat("p_mut_strategy", 0.001)# 以p_mut_strategy的概率对策略进行随机突变。 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html&quot;,&quot;title&quot;:&quot;Defining Agent Functions — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;Agent Functions can be specified as C++ functions, built at compile time when using the C++ API, or they can be specified as Run-Time Compiled (RTC) function strings when using the both the C++ and Py&quot;}'>1</sup>](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html)

    # Grid size (useful for potential boundary checks, though torus is default)
    # 修正: 使用 newPropertyUint32 定义 uint32 类型的环境属性 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html&quot;,&quot;title&quot;:&quot;Defining Agent Functions — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;Agent Functions can be specified as C++ functions, built at compile time when using the C++ API, or they can be specified as Run-Time Compiled (RTC) function strings when using the both the C++ and Py&quot;}'>1</sup>](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html)
    env.newPropertyUint32("grid_size", 50) # Grid 边长

    # Threshold for neighbor random selection (if needed)
    # env.newPropertyFloat("selection_threshold", 1.0 / 8.0) # Example if selecting one [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html&quot;,&quot;title&quot;:&quot;Defining Agent Functions — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;Agent Functions can be specified as C++ functions, built at compile time when using the C++ API, or they can be specified as Run-Time Compiled (RTC) function strings when using the both the C++ and Py&quot;}'>1</sup>](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html)

# -------------------------------------
# 代理变量 (Agent Variables)
# -------------------------------------

# Maximum number of neighbors (Moore neighborhood = 8)
MAX_NEIGHBORS = 8

def define_cultural_agent(model: pyflamegpu.ModelDescription):
    """向模型描述中添加 CulturalAgent 的定义"""
    agent = model.newAgent("CulturalAgent")

    # Core State Variables (与 MESA 对应)
    # Agent ID 是内置的，不需要作为 agent variable 定义 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/tutorial/index.html&quot;,&quot;title&quot;:&quot;Tutorial — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;This tutorial presents a brief overview of FLAME GPU 2 and a worked example demonstrating how to implement the [Circles example](https://github.com/FLAMEGPU/FLAMEGPU2/tree/master/examples/circles_spat&quot;}'>5</sup>](https://docs.flamegpu.com/tutorial/index.html)
    # agent.newVariable("id", pyflamegpu.id_t) # Agent ID (handled by FlameGPU)
    agent.newVariableUInt("strategy") # 当前策略，0表示背叛，1表示合作 # 修正: 使用 newVariableInt32 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    agent.newVariableFloat("C")       # 文化参数，取值范围是[0, 1]任意实数 # 修正: 使用 newVariableFloat [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    agent.newVariableFloat("utility") # Current utility, calculated each step # 修正: 使用 newVariableFloat [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)

    # Temporary Variables for Staged Updates (类似 MESA 的 next_*)
    agent.newVariableUInt("next_strategy") # 下一期策略，0表示背叛，1表示合作  # 修正: 使用 newVariableInt32 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    agent.newVariableFloat("next_C") # 修正: 使用 newVariableFloat [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)

    # Variables to Store Neighbor Information (用于决策)
    # We need to store info about neighbors received via messages to make decisions
    agent.newVariableUInt("neighbor_count") # 邻居的个数 # 修正: 使用 newVariableUint32 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    # Arrays to store data from *up to* MAX_NEIGHBORS neighbors
    # Note: FlameGPU Array Variables require fixed sizes defined at compile time.
    agent.newVariableArrayInt32("neighbor_strategies", MAX_NEIGHBORS) # 修正: 使用 newVariableArrayInt32 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    agent.newVariableArrayFloat("neighbor_Cs", MAX_NEIGHBORS) # 修正: 使用 newVariableArrayFloat [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    agent.newVariableArrayFloat("neighbor_utilities", MAX_NEIGHBORS) # 修正: 使用 newVariableArrayFloat [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)

    # Internal position variables (FlameGPU often manages these, but we can declare if needed)
    # agent.newVariableFloat("x") # 修正: 使用 newVariableFloat [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    # agent.newVariableFloat("y") # 修正: 使用 newVariableFloat [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    # Let's assume FlameGPU's default spatial handling works for now via messaging radius.

# -------------------------------------
# 消息定义 (Message Definitions)
# -------------------------------------
# Define messages agents will use to communicate state and utility

def define_messages(model: pyflamegpu.ModelDescription):
    """向模型描述中添加消息定义"""

    # Message to broadcast basic state (Strategy and C) for utility calculation
    # We don't strictly need 'id' here if we only use neighbor state for *our* utility
    msg_state = model.newMessageSpatial2D("StateMessage")
    msg_state.newVariableID("id") # Sender's ID (useful for debugging/tracking) # 修正: 使用 newVariableID [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    msg_state.newVariableInt32("strategy") # 修正: 使用 newVariableInt32 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    msg_state.newVariableFloat("C") # 修正: 使用 newVariableFloat [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    # Define interaction radius (e.g., sqrt(2) for Moore neighborhood on integer grid)
    msg_state.setRadius(1.5) # A bit more than sqrt(2) to ensure corners are included # 此行代码本身是正确的用法 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/defining-messages-communication/index.html&quot;,&quot;title&quot;:&quot;Defining Messages (Communication) — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;Communication between agents in FLAME GPU is handled through messages. Messages contain variables which are used to transmit information between agents. Agents may output a single message from an agen&quot;}'>12</sup>](https://docs.flamegpu.com/guide/defining-messages-communication/index.html)
    # msg_state.setMinLocation([0.0, 0.0]) # Define grid boundaries if not infinite
    # msg_state.setMaxLocation([50.0, 50.0]) # Match environment grid_size

    # Message to broadcast calculated utility along with state for decision making
    msg_utility = model.newMessageSpatial2D("UtilityMessage")
    msg_utility.newVariableID("id") # 修正: 使用 newVariableID [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    msg_utility.newVariableInt32("strategy") # 修正: 使用 newVariableInt32 [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    msg_utility.newVariableFloat("C") # 修正: 使用 newVariableFloat [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    msg_utility.newVariableFloat("utility") # 修正: 使用 newVariableFloat [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/creating-a-model/index.html&quot;,&quot;title&quot;:&quot;Creating a Model — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;In order to create a FLAME GPU 2 simulation, you must first describe the model.\n\nWhat is a Model?[](#what-is-a-model \&quot;Permalink to this heading\&quot;)\n-----------------------------------------------------&quot;}'>2</sup>](https://docs.flamegpu.com/guide/creating-a-model/index.html)
    msg_utility.setRadius(1.5) # Same radius [<sup data-citation='{&quot;url&quot;:&quot;https://docs.flamegpu.com/guide/defining-messages-communication/index.html&quot;,&quot;title&quot;:&quot;Defining Messages (Communication) — FLAME GPU 2 0 documentation&quot;,&quot;content&quot;:&quot;Communication between agents in FLAME GPU is handled through messages. Messages contain variables which are used to transmit information between agents. Agents may output a single message from an agen&quot;}'>12</sup>](https://docs.flamegpu.com/guide/defining-messages-communication/index.html)
    # msg_utility.setMinLocation([0.0, 0.0])
    # msg_utility.setMaxLocation([50.0, 50.0])

#    print("Messages 'StateMessage' and 'UtilityMessage' defined.")

# Example Usage (how you would use these functions)
# model_desc = pyflamegpu.ModelDescription("Cultural_Prisoners_Dilemma")
# define_environment(model_desc)
# define_cultural_agent(model_desc)
# define_messages(model_desc)
