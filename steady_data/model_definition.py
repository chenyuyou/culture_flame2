# model_definition.py

import pyflamegpu
import numpy as np

# Import from the NEW CUDA file with STAGED functions
# Assuming cultural_game_cuda_array.py exists and contains the CUDA functions
from cultural_game_cuda_array import (
    output_state_func,
    calculate_utility_func,
    calculate_local_metrics_func,
    decide_updates_func,
    mutate_func,
    advance_func
)


def create_model(model_name="CulturalGameArrayStaged"):
    """Creates the FLAMEGPU ModelDescription."""
    model = pyflamegpu.ModelDescription(model_name)
    return model


def define_environment(model, params):
    """Defines environment properties based on the provided parameters."""
    env = model.Environment()
    # These properties are accessed by Agent Functions (CUDA code) or StepFunction
    env.newPropertyFloat("b", params["b"])
    env.newPropertyFloat("K", params["K"])
    env.newPropertyFloat("K_C", params["K_C"])
    env.newPropertyFloat("p_update_C", params["p_update_C"])
    env.newPropertyFloat("p_mut_culture", params["p_mut_culture"])
    env.newPropertyFloat("p_mut_strategy", params["p_mut_strategy"])
    env.newMacroPropertyFloat("payoff", 2, 2)
    env.newPropertyUInt("GRID_DIM_L", params["L"])

    # Reporting properties (updated by StepFunction)
    env.newPropertyUInt("agent_count", params["L"] * params["L"])
    env.newPropertyUInt("cooperators", 0)
    env.newPropertyUInt("defectors", 0)
    # Add properties for cooperation and defection rates
    env.newPropertyFloat("coop_rate", 0.0)
    env.newPropertyFloat("defection_rate", 0.0)
    env.newPropertyFloat("std_coop_rate", 0.0)
    env.newPropertyFloat("std_defection_rate", 0.0)
    env.newPropertyFloat("avg_C", 1.0)
    env.newPropertyFloat("std_C", 1.0)

    # New reporting properties
    env.newPropertyFloat("segregation_index_strategy", 0.0) # Segregation based on strategy
    env.newPropertyFloat("segregation_index_type", 0.0)     # Segregation based on cultural type (A/B)
    env.newPropertyFloat("boundary_fraction", 0.0)
    env.newPropertyFloat("boundary_coop_rate", 0.0)
    env.newPropertyFloat("bulk_coop_rate", 0.0)
    env.newPropertyUInt("total_boundary_agents", 0) # Helper for boundary_fraction
    env.newPropertyUInt("total_boundary_cooperators", 0) # Helper for boundary_coop_rate
    env.newPropertyUInt("total_bulk_cooperators", 0)     # Helper for bulk_coop_rate
    env.newPropertyUInt("total_bulk_agents", 0)      # Helper for bulk_coop_rate
    # **IMPORTANT:** Declare the "random_seed" environment property
    # This is required because InitFunction accesses it via FLAMEGPU.environment.getPropertyUInt("random_seed")
    env.newPropertyUInt("random_seed", 0) # Initial value doesn't matter, it's set by RunPlanVector


    # Note: initial_coop_ratio, C_dist, mu, sigma are used by InitFunction
    # but do not need to be defined as environment properties here.


def define_messages(model, params):
    """Defines messages using MessageArray2D."""
    L = params["L"]

    # Message for sharing state (strategy, C) from t-1
    msg_state = model.newMessageArray2D("agent_state_message")
    msg_state.newVariableID("id")
    msg_state.newVariableUInt("strategy")
    msg_state.newVariableFloat("C")
    msg_state.setDimensions(L, L)
    # msg_state.setBoundaryWrap(True) # Explicitly enable wrapping - already handled by grid/message array


    # Message for sharing calculated utility and state needed for adoption
    msg_utility = model.newMessageArray2D("utility_message")
    msg_utility.newVariableID("id")
    msg_utility.newVariableFloat("utility")  # Calculated utility @ t
    msg_utility.newVariableUInt("strategy")  # Strategy @ t-1 (for adoption)
    msg_utility.newVariableFloat("C")  # C @ t-1 (for adoption)
    msg_utility.setDimensions(L, L)
    # msg_utility.setBoundaryWrap(True) # Explicitly enable wrapping


def define_agents(model):
    """Defines the 'CulturalAgent' agent type."""
    agent = model.newAgent("CulturalAgent")

    # Agent variables
    agent.newVariableInt("ix")  # Grid index X
    agent.newVariableInt("iy")  # Grid index Y
    agent.newVariableUInt("strategy")  # Current strategy
    agent.newVariableFloat("C")  # Current C
    agent.newVariableUInt("next_strategy")  # Buffer for next strategy
    agent.newVariableFloat("next_C")  # Buffer for next C
    # Variable to store calculated utility
    agent.newVariableFloat("current_utility", 0.0)

    # New variables for data collection - ENSURE THESE ARE PRESENT
    agent.newVariableUInt("same_strategy_neighbors", 0) # Count of neighbors with same strategy
    agent.newVariableUInt("same_type_neighbors", 0)     # Count of neighbors with same cultural type (A/B)
    agent.newVariableUInt("total_neighbors_count", 0)   # Total number of neighbors found
    agent.newVariableUInt("is_boundary_agent", 0)       # 1 if boundary, 0 if bulk
    agent.newVariableUInt("is_boundary_cooperator", 0)  # 1 if boundary and cooperator
    agent.newVariableUInt("is_bulk_cooperator", 0)      # 1 if bulk and cooperator
    agent.newVariableUInt("agent_type", 0)              # 1 for Type A, 2 for Type B (based on C threshold)


    # Link agent functions to the C++ code strings
    fn = agent.newRTCFunction("output_state_func", output_state_func)
    fn.setMessageOutput("agent_state_message")  # Outputs state

    fn = agent.newRTCFunction("calculate_utility_func", calculate_utility_func)
    fn.setMessageInput("agent_state_message")  # Reads state @ t-1
    fn.setMessageOutput("utility_message")  # Outputs utility @ t and state

    fn = agent.newRTCFunction("calculate_local_metrics_func", calculate_local_metrics_func)
    fn.setMessageInput("agent_state_message") # Reads state from t-1


    fn = agent.newRTCFunction("decide_updates_func", decide_updates_func)
    fn.setMessageInput("utility_message")  # Reads utility @ t and state

    fn = agent.newRTCFunction("mutate_func", mutate_func)
    # No message input/output

    fn = agent.newRTCFunction("advance_func", advance_func)
    # No message input/output


def define_execution_order(model):
    """Defines the layers and execution order for STAGED logic."""
    # Layer 1: Output state from t-1
    layer1 = model.newLayer("Output State")
    layer1.addAgentFunction("CulturalAgent", "output_state_func")

    # Layer 2: Calculate utility based on t-1 state, store it, and output for neighbors
    layer2 = model.newLayer("Calculate Utility")
    layer2.addAgentFunction("CulturalAgent", "calculate_utility_func")

    # Layer 3: Calculate local metrics based on t-1 state
    layer3 = model.newLayer("Calculate Local Metrics")
    layer3.addAgentFunction("CulturalAgent", "calculate_local_metrics_func")

    # Layer 4: Decide next strategy/C based on utilities calculated in Layer 2
    # Note: decide_updates_func uses utility_message, which is output by Layer 2
    layer4 = model.newLayer("Decide Updates")
    layer4.addAgentFunction("CulturalAgent", "decide_updates_func")

    # Layer 5: Apply mutation to the decided next state
    layer5 = model.newLayer("Mutate")
    layer5.addAgentFunction("CulturalAgent", "mutate_func")

    # Layer 6: Advance state for the next step
    layer6 = model.newLayer("Advance State")
    layer6.addAgentFunction("CulturalAgent", "advance_func")

