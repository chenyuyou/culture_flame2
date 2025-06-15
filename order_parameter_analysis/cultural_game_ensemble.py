# cultural_game_ensemble.py

import pyflamegpu
import sys
import random
import math
import time
import numpy as np
import os
import pandas as pd
import pickle
import json
from tqdm import tqdm
import itertools # Import itertools for parameter sweeps
import argparse # Import argparse for command line arguments
# Import from the NEW CUDA file with STAGED functions
from cultural_game_cuda_array import (
    output_state_func,
    calculate_utility_func,
    calculate_local_metrics_func,
    decide_updates_func,
    mutate_func,
    advance_func
)


# ==============================================================================
# SIMULATION PARAMETERS - CONFIGURE YOUR EXPERIMENTS HERE
# ==============================================================================
# Define a dictionary to hold all simulation parameters
# You can copy and modify this dictionary for different simulation runs
SIMULATION_PARAMETERS = {
    "L": 50,  # Grid size (L x L)
    "initial_coop_ratio": 0.5,  # Initial fraction of cooperators
    "b": 1.5,  # Benefit of cooperation (for the recipient)
    "K": 0.1,  # Selection strength for strategy update (Fermi rule)

    # Cultural Parameters (Initial Distribution)
    "C_dist": "uniform",  # Options: "uniform", "normal", "bimodal", "fixed"
    # Meaning depends on C_dist (e.g., mean for normal, fixed value, p(C=1) for bimodal)
    "mu": 0.5,
    "sigma": 0.1,  # Std dev for normal distribution

    # Cultural Evolution Parameters
    "K_C": 0.1,  # Noise in cultural update rule (Fermi rule)
    "p_update_C": 0.1,  # Probability to attempt cultural update per step
    "p_mut_culture": 0.01,  # Probability of random cultural mutation per step
    "p_mut_strategy": 0.001,  # Probability of random strategy mutation per step

    "steps": 2000,  # Number of simulation steps to run
    "seed": None,  # Random seed for reproducibility (None lets FLAMEGPU generate)

    # Ensemble/Batch Run Parameters
    "num_runs": 20, # Number of independent simulation runs for this parameter set
    "output_directory": "results_flamegpu_array_staged", # Directory to save log files
    "steady_state_window": 500 # Add steady state window parameter
}
# ==============================================================================
# END OF PARAMETER CONFIGURATION
# ==============================================================================


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

    # **NEW** Properties for group-specific metrics
    env.newPropertyUInt("total_cooperators_A", 0)
    env.newPropertyUInt("total_cooperators_B", 0)
    env.newPropertyUInt("total_count_type_A", 0)
    env.newPropertyUInt("total_count_type_B", 0)
    env.newPropertyUInt("total_same_type_neighbors_A", 0)
    env.newPropertyUInt("total_same_type_neighbors_B", 0)
    env.newPropertyUInt("total_neighbors_count_A", 0)
    env.newPropertyUInt("total_neighbors_count_B", 0)

    env.newPropertyFloat("coop_rate_A", 0.0)
    env.newPropertyFloat("coop_rate_B", 0.0)
    env.newPropertyFloat("segregation_index_type_A", 0.0) # Segregation for Type A
    env.newPropertyFloat("segregation_index_type_B", 0.0) # Segregation for Type B


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

    # **NEW** Agent variables for group-specific metrics (for reduction)
    agent.newVariableUInt("cooperator_A", 0)
    agent.newVariableUInt("cooperator_B", 0)
    agent.newVariableUInt("count_type_A", 0)
    agent.newVariableUInt("count_type_B", 0)
    agent.newVariableUInt("same_type_neighbors_A", 0)
    agent.newVariableUInt("same_type_neighbors_B", 0)
    agent.newVariableUInt("total_neighbors_count_A", 0)
    agent.newVariableUInt("total_neighbors_count_B", 0)


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

    # Layer 3: Calculate local metrics based on t-1 state (including group-specific sums)
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



# --- Host Functions (InitFunction, StepFunction) ---

class InitFunction(pyflamegpu.HostFunction):
    """Host function run once at the beginning."""

    def __init__(self, params):
        super().__init__()
        # Store the parameters passed during instantiation
        self.params = params

    def run(self, FLAMEGPU):
        # Access parameters directly from the stored params dictionary
        L = self.params["L"]
        agent_count = L * L
        b = self.params["b"] # This is also in environment, but can be accessed here too
        initial_coop_ratio = self.params["initial_coop_ratio"]
        C_dist = self.params["C_dist"]
        mu = self.params["mu"]
        sigma = self.params["sigma"]

        # Set environment properties that are needed by Agent Functions or StepFunction
        FLAMEGPU.environment.setPropertyUInt("agent_count", agent_count)
        FLAMEGPU.environment.setPropertyUInt("GRID_DIM_L", L)
        payoff = FLAMEGPU.environment.getMacroPropertyFloat("payoff")
        # Payoff: [My Strategy][Neighbor Strategy] -> My Payoff
        # P(C,C)=1, P(C,D)=0, P(D,C)=b, P(D,D)=0
        # Row Player (Me), Column Player (Neighbor)
        payoff[1][1] = 1.0  # Me C, Neighbor C -> My Payoff = 1
        payoff[1][0] = 0.0  # Me C, Neighbor D -> My Payoff = 0
        payoff[0][1] = b    # Me D, Neighbor C -> My Payoff = b
        payoff[0][0] = 0.0  # Me D, Neighbor D -> My Payoff = 0

        agents = FLAMEGPU.agent("CulturalAgent")
        # Use the simulation seed for initial agent state randomness
        # Get the simulation seed from the environment (set by RunPlanVector)
        # This requires "random_seed" to be defined as an environment property
        sim_seed = FLAMEGPU.environment.getPropertyUInt("random_seed")
        rng = random.Random(sim_seed)

        for i in range(agent_count):
            agent = agents.newAgent()
            ix = i % L
            iy = i // L
            agent.setVariableInt("ix", ix)
            agent.setVariableInt("iy", iy)

            # Initial strategy (random based on rate)
            strategy = 1 if rng.random() < initial_coop_ratio else 0
            agent.setVariableUInt("strategy", strategy)
            agent.setVariableUInt("next_strategy", strategy)  # Initialize buffer

            # Initial Culture (C) - same logic as before
            C_value = 0.0
            if C_dist == "uniform":
                C_value = rng.uniform(0, 1)
            elif C_dist == "bimodal":
                C_value = 0.0 if rng.random() < (1.0 - mu) else 1.0
            elif C_dist == "normal":
                # Use numpy random for normal distribution, seeded from Python random
                np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
                C_value = np.clip(np_rng.normal(mu, sigma), 0, 1)
            elif C_dist == "fixed":
                C_value = mu
            else:
                C_value = rng.uniform(0, 1)  # Default to uniform if unknown
            agent.setVariableFloat("C", C_value)
            agent.setVariableFloat("next_C", C_value)  # Initialize buffer

            # Initialize the new utility variable
            agent.setVariableFloat("current_utility", 0.0)

            # Initialize new group-specific variables
            agent.setVariableUInt("cooperator_A", 0)
            agent.setVariableUInt("cooperator_B", 0)
            agent.setVariableUInt("count_type_A", 0)
            agent.setVariableUInt("count_type_B", 0)
            agent.setVariableUInt("same_type_neighbors_A", 0)
            agent.setVariableUInt("same_type_neighbors_B", 0)
            agent.setVariableUInt("total_neighbors_count_A", 0)
            agent.setVariableUInt("total_neighbors_count_B", 0)


class StepFunction(pyflamegpu.HostFunction):
    """Host function run after each GPU step."""

    def run(self, FLAMEGPU):
        agent_count_gpu = FLAMEGPU.agent("CulturalAgent").count()

        # --- Calculate Overall Metrics ---
        avg_coop_rate, std_coop_rate = FLAMEGPU.agent("CulturalAgent").meanStandardDeviationUInt("strategy")
        FLAMEGPU.environment.setPropertyFloat("coop_rate", avg_coop_rate)
        FLAMEGPU.environment.setPropertyFloat("std_coop_rate", std_coop_rate)
        FLAMEGPU.environment.setPropertyFloat("defection_rate", 1.0 - avg_coop_rate)
        FLAMEGPU.environment.setPropertyFloat("std_defection_rate", std_coop_rate)

        avg_C, std_C = FLAMEGPU.agent("CulturalAgent").meanStandardDeviationFloat("C")
        FLAMEGPU.environment.setPropertyFloat("avg_C", avg_C)
        FLAMEGPU.environment.setPropertyFloat("std_C", std_C)

        total_same_strategy_neighbors = FLAMEGPU.agent("CulturalAgent").sumUInt("same_strategy_neighbors")
        total_neighbors_sum = FLAMEGPU.agent("CulturalAgent").sumUInt("total_neighbors_count")

        # Calculate Overall Strategy Segregation Index (using the original formula)
        if total_neighbors_sum > 0:
            seg_index_strategy = (2.0 * total_same_strategy_neighbors - total_neighbors_sum) / total_neighbors_sum
            FLAMEGPU.environment.setPropertyFloat("segregation_index_strategy", seg_index_strategy)
        else:
            FLAMEGPU.environment.setPropertyFloat("segregation_index_strategy", 0.0) # Or np.nan

        # Calculate Overall Cultural Segregation Index as AVERAGE SAME TYPE NEIGHBOR PROPORTION
        total_same_type_neighbors = FLAMEGPU.agent("CulturalAgent").sumUInt("same_type_neighbors")
        if total_neighbors_sum > 0:
            avg_same_type_neighbor_proportion = float(total_same_type_neighbors) / total_neighbors_sum
            FLAMEGPU.environment.setPropertyFloat("segregation_index_type", avg_same_type_neighbor_proportion)
        else:
            FLAMEGPU.environment.setPropertyFloat("segregation_index_type", 0.0)     # Or np.nan

        # --- Calculate Boundary and Bulk Cooperation Rates ---
        total_boundary_agents = FLAMEGPU.agent("CulturalAgent").sumUInt("is_boundary_agent")
        FLAMEGPU.environment.setPropertyUInt("total_boundary_agents", total_boundary_agents)
        boundary_frac = float(total_boundary_agents) / agent_count_gpu if agent_count_gpu > 0 else 0.0
        FLAMEGPU.environment.setPropertyFloat("boundary_fraction", boundary_frac)

        total_boundary_cooperators = FLAMEGPU.agent("CulturalAgent").sumUInt("is_boundary_cooperator")
        total_bulk_cooperators = FLAMEGPU.agent("CulturalAgent").sumUInt("is_bulk_cooperator")
        total_bulk_agents = agent_count_gpu - total_boundary_agents
        FLAMEGPU.environment.setPropertyUInt("total_boundary_cooperators", total_boundary_cooperators)
        FLAMEGPU.environment.setPropertyUInt("total_bulk_cooperators", total_bulk_cooperators)
        FLAMEGPU.environment.setPropertyUInt("total_bulk_agents", total_bulk_agents)

        if total_boundary_agents > 0:
            boundary_coop_rate = float(total_boundary_cooperators) / total_boundary_agents
            FLAMEGPU.environment.setPropertyFloat("boundary_coop_rate", boundary_coop_rate)
        else:
            FLAMEGPU.environment.setPropertyFloat("boundary_coop_rate", 0.0)

        if total_bulk_agents > 0:
            bulk_coop_rate = float(total_bulk_cooperators) / total_bulk_agents
            FLAMEGPU.environment.setPropertyFloat("bulk_coop_rate", bulk_coop_rate)
        else:
            FLAMEGPU.environment.setPropertyFloat("bulk_coop_rate", 0.0)

        # --- Calculate Group-Specific Metrics ---
        total_cooperators_A = FLAMEGPU.agent("CulturalAgent").sumUInt("cooperator_A")
        total_cooperators_B = FLAMEGPU.agent("CulturalAgent").sumUInt("cooperator_B")
        total_count_type_A = FLAMEGPU.agent("CulturalAgent").sumUInt("count_type_A")
        total_count_type_B = FLAMEGPU.agent("CulturalAgent").sumUInt("count_type_B")
        total_same_type_neighbors_A = FLAMEGPU.agent("CulturalAgent").sumUInt("same_type_neighbors_A")
        total_same_type_neighbors_B = FLAMEGPU.agent("CulturalAgent").sumUInt("same_type_neighbors_B")
        total_neighbors_count_A = FLAMEGPU.agent("CulturalAgent").sumUInt("total_neighbors_count_A")
        total_neighbors_count_B = FLAMEGPU.agent("CulturalAgent").sumUInt("total_neighbors_count_B")

        FLAMEGPU.environment.setPropertyUInt("total_cooperators_A", total_cooperators_A)
        FLAMEGPU.environment.setPropertyUInt("total_cooperators_B", total_cooperators_B)
        FLAMEGPU.environment.setPropertyUInt("total_count_type_A", total_count_type_A)
        FLAMEGPU.environment.setPropertyUInt("total_count_type_B", total_count_type_B)
        FLAMEGPU.environment.setPropertyUInt("total_same_type_neighbors_A", total_same_type_neighbors_A)
        FLAMEGPU.environment.setPropertyUInt("total_same_type_neighbors_B", total_same_type_neighbors_B)
        FLAMEGPU.environment.setPropertyUInt("total_neighbors_count_A", total_neighbors_count_A)
        FLAMEGPU.environment.setPropertyUInt("total_neighbors_count_B", total_neighbors_count_B)


        # Calculate Cooperation Rate for Type A
        if total_count_type_A > 0:
            coop_rate_A = float(total_cooperators_A) / total_count_type_A
            FLAMEGPU.environment.setPropertyFloat("coop_rate_A", coop_rate_A)
        else:
            FLAMEGPU.environment.setPropertyFloat("coop_rate_A", 0.0) # Or np.nan

        # Calculate Cooperation Rate for Type B
        if total_count_type_B > 0:
            coop_rate_B = float(total_cooperators_B) / total_count_type_B
            FLAMEGPU.environment.setPropertyFloat("coop_rate_B", coop_rate_B)
        else:
            FLAMEGPU.environment.setPropertyFloat("coop_rate_B", 0.0) # Or np.nan

        # Calculate Cultural Segregation Index (Average Same Type Neighbor Proportion) for Type A
        if total_neighbors_count_A > 0:
            seg_index_type_A = float(total_same_type_neighbors_A) / total_neighbors_count_A
            FLAMEGPU.environment.setPropertyFloat("segregation_index_type_A", seg_index_type_A)
        else:
            FLAMEGPU.environment.setPropertyFloat("segregation_index_type_A", 0.0) # Or np.nan

        # Calculate Cultural Segregation Index (Average Same Type Neighbor Proportion) for Type B
        if total_neighbors_count_B > 0:
            seg_index_type_B = float(total_same_type_neighbors_B) / total_neighbors_count_B
            FLAMEGPU.environment.setPropertyFloat("segregation_index_type_B", seg_index_type_B)
        else:
            FLAMEGPU.environment.setPropertyFloat("segregation_index_type_B", 0.0) # Or np.nan


# --- Logging, Output, Runs Configuration ---

def define_logs(model):
    """Defines logging."""
    log_cfg = pyflamegpu.StepLoggingConfig(model)
    log_cfg.setFrequency(1)  # Log every step
    # Log environment properties updated in StepFunction
    log_cfg.logEnvironment("coop_rate")
    log_cfg.logEnvironment("defection_rate")
    log_cfg.logEnvironment("std_coop_rate")
    log_cfg.logEnvironment("std_defection_rate")
    log_cfg.logEnvironment("avg_C")
    log_cfg.logEnvironment("std_C")

    # Log new overall metrics
    log_cfg.logEnvironment("segregation_index_strategy")
    log_cfg.logEnvironment("segregation_index_type")
    log_cfg.logEnvironment("boundary_fraction")
    log_cfg.logEnvironment("boundary_coop_rate")
    log_cfg.logEnvironment("bulk_coop_rate")

    # **NEW** Log group-specific metrics
    log_cfg.logEnvironment("coop_rate_A")
    log_cfg.logEnvironment("coop_rate_B")
    log_cfg.logEnvironment("segregation_index_type_A")
    log_cfg.logEnvironment("segregation_index_type_B")

    # Optionally log helper counts if needed for debugging
    # log_cfg.logEnvironment("total_boundary_agents")
    # log_cfg.logEnvironment("total_boundary_cooperators")
    # log_cfg.logEnvironment("total_bulk_cooperators")
    # log_cfg.logEnvironment("total_bulk_agents")
    # log_cfg.logEnvironment("total_cooperators_A")
    # log_cfg.logEnvironment("total_cooperators_B")
    # log_cfg.logEnvironment("total_count_type_A")
    # log_cfg.logEnvironment("total_count_type_B")
    # log_cfg.logEnvironment("total_same_type_neighbors_A")
    # log_cfg.logEnvironment("total_same_type_neighbors_B")
    # log_cfg.logEnvironment("total_neighbors_count_A")
    # log_cfg.logEnvironment("total_neighbors_count_B")


    return log_cfg


def define_output(ensemble, output_directory):
    """Configures simulation output."""
    os.makedirs(output_directory, exist_ok=True)  # Ensure directory exists
    ensemble.Config().out_directory = output_directory
    ensemble.Config().out_format = "json" # Ensure JSON format for easy reading
    ensemble.Config().timing = True
    ensemble.Config().truncate_log_files = True
    # Normal error level is fine
    ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Fast
    ensemble.Config().devices = pyflamegpu.IntSet([0])  # Use GPU 0


def define_run_plans(model, params):
    """Creates RunPlanVector for simulations based on parameters."""
    num_runs = params.get("num_runs", 1) # Get num_runs from params, default to 1
    run_plans = pyflamegpu.RunPlanVector(model, num_runs)
    run_plans.setSteps(params["steps"])

    # Use simulation seed for agent initialization randomness within InitFunction
    # Use step seed for randomness within agent functions (like Fermi choice)
    # If seed is None, pyflamegpu will generate unique seeds for each run
    if params.get("seed") is not None:
         # If a specific seed is provided, use it as the base for all runs
         # FLAMEGPU will add run index to this base seed for each run
         run_plans.setRandomSimulationSeed(params["seed"], num_runs)
    else:
         # If seed is None, let FLAMEGPU generate unique seeds for each run
         # Use a different base seed for each RunPlanVector creation if you run multiple sets
         # For simplicity here, we use time, but for sweeps, a systematic approach is better
         run_plans.setRandomSimulationSeed(int(time.time()), num_runs)


    # Set environment properties for the runs that are accessed by Agent Functions or StepFunction
    run_plans.setPropertyFloat("b", params["b"])
    run_plans.setPropertyFloat("K", params["K"])
    run_plans.setPropertyFloat("K_C", params["K_C"])
    run_plans.setPropertyFloat("p_update_C", params["p_update_C"])
    run_plans.setPropertyFloat("p_mut_culture", params["p_mut_culture"])
    run_plans.setPropertyFloat("p_mut_strategy", params["p_mut_strategy"])
    run_plans.setPropertyUInt("GRID_DIM_L", params["L"])

    # Note: initial_coop_ratio, C_dist, mu, sigma are passed to InitFunction via its constructor
    # and do not need to be set as environment properties here for RunPlanVector.


    return run_plans

# --- Main Simulation Execution ---
def run_simulation(params):
    """Sets up and runs the STAGED FLAMEGPU simulation with given parameters."""
    print("--- Setting up STAGED Array-Based Cultural Game ---")
    print(f"Parameters: {params}")

    model = create_model()
    # Define environment properties that are accessed by Agent Functions or StepFunction
    define_environment(model, params)

    # Note: initial_coop_ratio, C_dist, mu, sigma are used by InitFunction
    # and are passed to its constructor, not set as environment properties here.

    define_messages(model, params)  # Defines both state and utility messages
    define_agents(model)  # Defines agent with utility variable and staged functions
    define_execution_order(model)  # Defines the 5 layers

    # Instantiate and add Host Functions
    # Pass the parameters dictionary to the InitFunction constructor
    init_func_instance = InitFunction(params)
    step_func_instance = StepFunction()
    model.addInitFunction(init_func_instance)
    model.addStepFunction(step_func_instance)  # Runs after all GPU layers

    run_plans = define_run_plans(model, params)
    log_config = define_logs(model)

    print("Initializing CUDA Ensemble...")
    # Use pyflamegpu.CUDAEnsemble for running on GPU
    ensemble_sim = pyflamegpu.CUDAEnsemble(model)
    define_output(ensemble_sim, params["output_directory"])  # Pass output directory
    ensemble_sim.setStepLog(log_config)

    # Get the number of runs from the RunPlanVector using len()
    num_runs = len(run_plans)

    print(
        f"--- Starting FLAMEGPU Simulation ({num_runs} runs, {params['steps']} steps) ---")
    start_time = time.time()
    ensemble_sim.simulate(run_plans)
    end_time = time.time()
    print(f"--- Simulation Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Return the output directory and parameters for post-processing
    return params["output_directory"], params



# --- Modified Parameter Sweep Execution ---
def run_parameter_sweep(base_params, sweep_params):
    """
    Runs simulations for all combinations of sweep_params, keeping base_params fixed.
    Saves raw JSON logs to unique subdirectories. Does NOT process logs.
    Returns the base output directory.
    """
    print("\n--- Starting FLAMEGPU2 Parameter Sweep (Simulation Run Only) ---")
    sweep_keys = list(sweep_params.keys())
    sweep_values = list(sweep_params.values())
    value_combinations = list(itertools.product(*sweep_values))
    total_combinations = len(value_combinations)
    print(f"Total parameter combinations to run: {total_combinations}")
    # Determine the base output directory for this sweep
    base_output_dir = base_params.get("output_directory", "results_flamegpu_sweep")
    os.makedirs(base_output_dir, exist_ok=True) # Ensure base directory exists
    for i, combo in enumerate(tqdm(value_combinations, desc="Running Sweep Combinations")):
        current_params = base_params.copy()
        combo_id_parts = []
        for j, key in enumerate(sweep_keys):
            value = combo[j]
            current_params[key] = value
            # Create a meaningful ID part (handle floats carefully)
            id_part = f"{key}{value:.4g}" if isinstance(
                value, float) else f"{key}{value}"
            combo_id_parts.append(id_part.replace('.', 'p').replace('-', 'm')) # Replace . and - for directory names
        # Generate a unique output subdirectory for this parameter combination
        combo_output_subdir = "_".join(combo_id_parts)
        current_params["output_directory"] = os.path.join(base_output_dir, combo_output_subdir)
        os.makedirs(current_params["output_directory"], exist_ok=True) # Ensure subdirectory exists
        print(f"\nRunning combination {i+1}/{total_combinations}: {combo_output_subdir}")
        # --- Setup and Run Simulation for this combination ---
        try:
            model = create_model()
            define_environment(model, current_params)
            define_messages(model, current_params)
            define_agents(model)
            define_execution_order(model)
            init_func_instance = InitFunction(current_params)
            step_func_instance = StepFunction()
            model.addInitFunction(init_func_instance)
            model.addStepFunction(step_func_instance)
            run_plans = define_run_plans(model, current_params)
            log_config = define_logs(model)
            print("Initializing CUDA Ensemble...")
            ensemble_sim = pyflamegpu.CUDAEnsemble(model)
            define_output(ensemble_sim, current_params["output_directory"])
            ensemble_sim.setStepLog(log_config)
            num_runs = len(run_plans)
            print(
                f"--- Starting FLAMEGPU Simulation ({num_runs} runs, {current_params['steps']} steps) ---")
            start_time = time.time()
            ensemble_sim.simulate(run_plans)
            end_time = time.time()
            print(f"--- Simulation Finished for {combo_output_subdir} ---")
            print(f"Total execution time: {end_time - start_time:.2f} seconds")
            print(f"Raw logs saved to: {current_params['output_directory']}")
        except Exception as e:
            print(f"Error running simulation for combination {combo_output_subdir}: {e}")
            import traceback
            traceback.print_exc()
            # Continue to the next combination even if one fails
    print("\n--- All FLAMEGPU2 Simulations for Sweep Finished ---")
    return base_output_dir # Return the root directory where all subdirectories were created
# ==========================================================================
# MAIN EXECUTION BLOCK FOR RUNNING SIMULATIONS
# ==========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLAMEGPU2 Cultural Game Simulations.")
    parser.add_argument("--output_dir", type=str, default=SIMULATION_PARAMETERS["output_directory"],
                        help=f"Base directory to save simulation logs. Default: {SIMULATION_PARAMETERS['output_directory']}")
    parser.add_argument("--steps", type=int, default=SIMULATION_PARAMETERS["steps"],
                        help=f"Number of simulation steps per run. Default: {SIMULATION_PARAMETERS['steps']}")
    parser.add_argument("--num_runs", type=int, default=SIMULATION_PARAMETERS["num_runs"],
                        help=f"Number of independent runs per parameter set. Default: {SIMULATION_PARAMETERS['num_runs']}")
    # Add other base parameters if you want to control them from command line
    # parser.add_argument("--L", type=int, default=SIMULATION_PARAMETERS["L"], help="Grid size L")
    # parser.add_argument("--b", type=float, default=SIMULATION_PARAMETERS["b"], help="Benefit of cooperation b")
    # ...
    args = parser.parse_args()
    # Update base parameters from command line arguments
    base_params_main = SIMULATION_PARAMETERS.copy()
    base_params_main["output_directory"] = args.output_dir
    base_params_main["steps"] = args.steps
    base_params_main["num_runs"] = args.num_runs
    # Update other params if added to parser
    # Define the parameter sweep configuration
    # Remove parameters that will be swept from the base params for clarity
    base_params_main["p_mut_culture"] = 0.005  
    base_params_main["p_mut_strategy"] = 0.001  
    # Update other params if added to parser
    values_part1 = np.arange(1.1, 3.6 + 0.1, 0.5) # 从 1.1 到 3.6，步长 0.1，包含 3.6
# 生成第二个范围的数值
    values_part2 = np.arange(3.7, 100 + 0.5, 0.5) # 从 3.7 到 8.7，步长 0.5，包含 8.7

    # Define the parameter sweep configuration for the main scan
    # Remove parameters that will be swept from the base params for clarity
#    values_part1 = np.arange(1.1, 3.7, 0.2) # 从 1.1 到 3.6，步长 0.1，包含 3.6
# 生成第二个范围的数值
#    values_part2 = np.arange(3.7, 8.7 + 0.5, 0.5)
# 合并所有值并去重

    sweep_params_config_main = {
        'L': [50],
        'b': np.unique(np.concatenate((values_part1, values_part2)))
    }
    # Ensure sweep parameters are not in base_params_main if they are swept
    for key in sweep_params_config_main.keys():
        if key in base_params_main:
            del base_params_main[key]
    print("\n--- Running Main Scan Simulations ---")
    main_scan_log_dir = run_parameter_sweep(base_params_main, sweep_params_config_main)
    print(f"\nMain scan raw logs saved to base directory: '{main_scan_log_dir}'")
    print(f"You can now run 'python process_logs.py --log_dir {main_scan_log_dir}' to process the logs.")
#     --- Example for Phase Diagram Scan Simulations ---
#     You can uncomment and configure this section to run the PD scan
