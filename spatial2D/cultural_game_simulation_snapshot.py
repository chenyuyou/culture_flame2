# cultural_game_simulation_snapshot.py

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
from tqdm import tqdm # Keep tqdm for potential future use or if you add progress bars elsewhere
import itertools
import argparse
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
# SIMULATION PARAMETERS - CONFIGURE YOUR SINGLE RUN HERE
# ==============================================================================
# Define a dictionary to hold all simulation parameters for a single run
SINGLE_RUN_PARAMETERS = {
    "L": 50,  # Grid size (L x L)
    "initial_coop_ratio": 0.5,  # Initial fraction of cooperators
    "b": 1.6,  # Benefit of cooperation (for the recipient)
    "K": 0.01,  # Selection strength for strategy update (Fermi rule)

    # Cultural Parameters (Initial Distribution)
    "C_dist": "uniform",  # Options: "uniform", "normal", "bimodal", "fixed"
    # Meaning depends on C_dist (e.g., mean for normal, fixed value, p(C=1) for bimodal)
    "mu": 0.5,
    "sigma": 0.1,  # Std dev for normal distribution

    # Cultural Evolution Parameters
    "K_C": 0.01,  # Noise in cultural update rule (Fermi rule)
    "p_update_C": 0.01,  # Probability to attempt cultural update per step
    "p_mut_culture": 0.001,  # Probability of random cultural mutation per step
    "p_mut_strategy": 0.001,  # Probability of random strategy mutation per step

    "steps": 2000,  # Number of simulation steps to run
    "seed": None,  # Random seed for reproducibility (None lets FLAMEGPU generate)

    "output_directory": "snapshot_output", # Directory to save the snapshot file
    "snapshot_filename_pattern": "snapshot_step_{}.json", # Filename pattern for snapshots
    "snapshot_steps": [], # List of steps at which to save snapshots (empty means only final)
    "snapshot_interval": 0 # Save snapshot every N steps (0 means no interval saving)
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
    env.newPropertyFloat("b", params["b"])
    env.newPropertyFloat("K", params["K"])
    env.newPropertyFloat("K_C", params["K_C"])
    env.newPropertyFloat("p_update_C", params["p_update_C"])
    env.newPropertyFloat("p_mut_culture", params["p_mut_culture"])
    env.newPropertyFloat("p_mut_strategy", params["p_mut_strategy"])
    env.newMacroPropertyFloat("payoff", 2, 2)
    env.newPropertyUInt("GRID_DIM_L", params["L"])

    # Reporting properties - Keep these as they might be useful in snapshot
    env.newPropertyUInt("agent_count", params["L"] * params["L"])
    env.newPropertyUInt("cooperators", 0)
    env.newPropertyUInt("defectors", 0)
    env.newPropertyFloat("coop_rate", 0.0)
    env.newPropertyFloat("defection_rate", 0.0)
    env.newPropertyFloat("std_coop_rate", 0.0)
    env.newPropertyFloat("std_defection_rate", 0.0)
    env.newPropertyFloat("avg_C", 1.0)
    env.newPropertyFloat("std_C", 1.0)

    # New reporting properties - Keep these as they might be useful in snapshot
    env.newPropertyFloat("segregation_index_strategy", 0.0)
    env.newPropertyFloat("segregation_index_type", 0.0)
    env.newPropertyFloat("boundary_fraction", 0.0)
    env.newPropertyFloat("boundary_coop_rate", 0.0)
    env.newPropertyFloat("bulk_coop_rate", 0.0)
    env.newPropertyUInt("total_boundary_agents", 0)
    env.newPropertyUInt("total_boundary_cooperators", 0)
    env.newPropertyUInt("total_bulk_cooperators", 0)
    env.newPropertyUInt("total_bulk_agents", 0)
    env.newPropertyUInt("random_seed", 0)


def define_messages(model, params):
    """Defines messages using MessageArray2D."""
    L = params["L"]

    msg_state = model.newMessageArray2D("agent_state_message")
    msg_state.newVariableID("id")
    msg_state.newVariableUInt("strategy")
    msg_state.newVariableFloat("C")
    msg_state.setDimensions(L, L)

    msg_utility = model.newMessageArray2D("utility_message")
    msg_utility.newVariableID("id")
    msg_utility.newVariableFloat("utility")
    msg_utility.newVariableUInt("strategy")
    msg_utility.newVariableFloat("C")
    msg_utility.setDimensions(L, L)


def define_agents(model):
    """Defines the 'CulturalAgent' agent type."""
    agent = model.newAgent("CulturalAgent")

    agent.newVariableInt("ix")
    agent.newVariableInt("iy")
    agent.newVariableUInt("strategy")
    agent.newVariableFloat("C")
    agent.newVariableUInt("next_strategy")
    agent.newVariableFloat("next_C")
    agent.newVariableFloat("current_utility", 0.0)

    agent.newVariableUInt("same_strategy_neighbors", 0)
    agent.newVariableUInt("same_type_neighbors", 0)
    agent.newVariableUInt("total_neighbors_count", 0)
    agent.newVariableUInt("is_boundary_agent", 0)
    agent.newVariableUInt("is_boundary_cooperator", 0)
    agent.newVariableUInt("is_bulk_cooperator", 0)
    agent.newVariableUInt("agent_type", 0)

    fn = agent.newRTCFunction("output_state_func", output_state_func)
    fn.setMessageOutput("agent_state_message")

    fn = agent.newRTCFunction("calculate_utility_func", calculate_utility_func)
    fn.setMessageInput("agent_state_message")
    fn.setMessageOutput("utility_message")

    fn = agent.newRTCFunction("calculate_local_metrics_func", calculate_local_metrics_func)
    fn.setMessageInput("agent_state_message")

    fn = agent.newRTCFunction("decide_updates_func", decide_updates_func)
    fn.setMessageInput("utility_message")

    fn = agent.newRTCFunction("mutate_func", mutate_func)

    fn = agent.newRTCFunction("advance_func", advance_func)


def define_execution_order(model):
    """Defines the layers and execution order for STAGED logic."""
    layer1 = model.newLayer("Output State")
    layer1.addAgentFunction("CulturalAgent", "output_state_func")

    layer2 = model.newLayer("Calculate Utility")
    layer2.addAgentFunction("CulturalAgent", "calculate_utility_func")

    layer3 = model.newLayer("Calculate Local Metrics")
    layer3.addAgentFunction("CulturalAgent", "calculate_local_metrics_func")

    layer4 = model.newLayer("Decide Updates")
    layer4.addAgentFunction("CulturalAgent", "decide_updates_func")

    layer5 = model.newLayer("Mutate")
    layer5.addAgentFunction("CulturalAgent", "mutate_func")

    layer6 = model.newLayer("Advance State")
    layer6.addAgentFunction("CulturalAgent", "advance_func")


# --- Host Functions (StepFunction) ---

class StepFunction(pyflamegpu.HostFunction):
    """Host function run after each GPU step."""

    def run(self, FLAMEGPU):
        agent_count_gpu = FLAMEGPU.agent("CulturalAgent").count()
        if agent_count_gpu > 0:
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

            if total_neighbors_sum > 0:
                seg_index_strategy = (2.0 * total_same_strategy_neighbors - total_neighbors_sum) / total_neighbors_sum
                FLAMEGPU.environment.setPropertyFloat("segregation_index_strategy", seg_index_strategy)
                total_same_type_neighbors = FLAMEGPU.agent("CulturalAgent").sumUInt("same_type_neighbors")
                seg_index_type = float(total_same_type_neighbors) / total_neighbors_sum
                FLAMEGPU.environment.setPropertyFloat("segregation_index_type", seg_index_type)
            else:
                FLAMEGPU.environment.setPropertyFloat("segregation_index_strategy", 0.0)
                FLAMEGPU.environment.setPropertyFloat("segregation_index_type", 0.0)

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


# --- Main Simulation Execution and Snapshot Export ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single FLAMEGPU2 Cultural Game Simulation and export snapshots at specified steps.")
    parser.add_argument("--output_dir", type=str, default=SINGLE_RUN_PARAMETERS["output_directory"],
                        help=f"Directory to save the snapshot files. Default: {SINGLE_RUN_PARAMETERS['output_directory']}")
    parser.add_argument("--snapshot_filename_pattern", type=str, default=SINGLE_RUN_PARAMETERS["snapshot_filename_pattern"],
                        help=f"Filename pattern for the snapshot JSONs (use {{}} for step number). Default: {SINGLE_RUN_PARAMETERS['snapshot_filename_pattern']}")
    parser.add_argument("--steps", type=int, default=SINGLE_RUN_PARAMETERS["steps"],
                        help=f"Number of simulation steps to run. Default: {SINGLE_RUN_PARAMETERS['steps']}")
    parser.add_argument("--seed", type=int, default=SINGLE_RUN_PARAMETERS["seed"],
                        help=f"Random seed for the simulation. Default: {SINGLE_RUN_PARAMETERS['seed']} (None means time-based seed)")
    parser.add_argument("--snapshot_steps", type=int, nargs='*', default=SINGLE_RUN_PARAMETERS["snapshot_steps"],
                        help="List of specific steps at which to save snapshots (e.g., --snapshot_steps 100 500 1000).")
    parser.add_argument("--snapshot_interval", type=int, default=SINGLE_RUN_PARAMETERS["snapshot_interval"],
                        help="Save snapshot every N steps (e.g., --snapshot_interval 100). 0 means no interval saving.")

    # Add other parameters you want to control from command line for the single run
    parser.add_argument("--L", type=int, default=SINGLE_RUN_PARAMETERS["L"], help=f"Grid size L. Default: {SINGLE_RUN_PARAMETERS['L']}")
    parser.add_argument("--b", type=float, default=SINGLE_RUN_PARAMETERS["b"], help=f"Benefit of cooperation b. Default: {SINGLE_RUN_PARAMETERS['b']}")
    parser.add_argument("--K", type=float, default=SINGLE_RUN_PARAMETERS["K"], help=f"Selection strength K. Default: {SINGLE_RUN_PARAMETERS['K']}")
    parser.add_argument("--K_C", type=float, default=SINGLE_RUN_PARAMETERS["K_C"], help=f"Cultural noise K_C. Default: {SINGLE_RUN_PARAMETERS['K_C']}")
    parser.add_argument("--p_update_C", type=float, default=SINGLE_RUN_PARAMETERS["p_update_C"], help=f"Cultural update probability p_update_C. Default: {SINGLE_RUN_PARAMETERS['p_update_C']}")
    parser.add_argument("--p_mut_culture", type=float, default=SINGLE_RUN_PARAMETERS["p_mut_culture"], help=f"Cultural mutation probability p_mut_culture. Default: {SINGLE_RUN_PARAMETERS['p_mut_culture']}")
    parser.add_argument("--p_mut_strategy", type=float, default=SINGLE_RUN_PARAMETERS["p_mut_strategy"], help=f"Strategy mutation probability p_mut_strategy. Default: {SINGLE_RUN_PARAMETERS['p_mut_strategy']}")
    parser.add_argument("--initial_coop_ratio", type=float, default=SINGLE_RUN_PARAMETERS["initial_coop_ratio"], help=f"Initial cooperation ratio. Default: {SINGLE_RUN_PARAMETERS['initial_coop_ratio']}")
    parser.add_argument("--C_dist", type=str, default=SINGLE_RUN_PARAMETERS["C_dist"], help=f"Initial C distribution. Default: {SINGLE_RUN_PARAMETERS['C_dist']}")
    parser.add_argument("--mu", type=float, default=SINGLE_RUN_PARAMETERS["mu"], help=f"Mean/parameter for C distribution. Default: {SINGLE_RUN_PARAMETERS['mu']}")
    parser.add_argument("--sigma", type=float, default=SINGLE_RUN_PARAMETERS["sigma"], help=f"Std dev for normal C distribution. Default: {SINGLE_RUN_PARAMETERS['sigma']}")


    args = parser.parse_args()

    # Update parameters for the single run from command line arguments
    current_params = SINGLE_RUN_PARAMETERS.copy()
    current_params["output_directory"] = args.output_dir
    current_params["snapshot_filename_pattern"] = args.snapshot_filename_pattern
    current_params["steps"] = args.steps
    current_params["seed"] = args.seed
    current_params["snapshot_steps"] = args.snapshot_steps
    current_params["snapshot_interval"] = args.snapshot_interval

    current_params["L"] = args.L
    current_params["b"] = args.b
    current_params["K"] = args.K
    current_params["K_C"] = args.K_C
    current_params["p_update_C"] = args.p_update_C
    current_params["p_mut_culture"] = args.p_mut_culture
    current_params["p_mut_strategy"] = args.p_mut_strategy
    current_params["initial_coop_ratio"] = args.initial_coop_ratio
    current_params["C_dist"] = args.C_dist
    current_params["mu"] = args.mu
    current_params["sigma"] = args.sigma


    print("--- Setting up STAGED Array-Based Cultural Game for Single Run ---")
    print(f"Parameters: {current_params}")

    model = create_model()
    env = define_environment(model, current_params) # Pass params to define_environment
    define_messages(model, current_params) # Pass params to define_messages
    define_agents(model)
    define_execution_order(model)

    # StepFunction is still needed to update environment properties during simulation
    step_func_instance = StepFunction()
    model.addStepFunction(step_func_instance)

    # --- Initialize CUDASimulation ---
    cuda_sim = pyflamegpu.CUDASimulation(model)

    # Initialize the simulation with command line arguments and seed
    # This also sets the random seed for the simulation
    cuda_sim.initialise(sys.argv)

    # --- Initialise Agent Population (Logic moved from InitFunction) ---
    L = current_params["L"]
    agent_count = L * L
    initial_coop_ratio = current_params["initial_coop_ratio"]
    C_dist = current_params["C_dist"]
    mu = current_params["mu"]
    sigma = current_params["sigma"]

    # Correct way to create and populate AgentVector
    init_pop = pyflamegpu.AgentVector(model.Agent("CulturalAgent"), agent_count)

    # Use the simulation seed for initial agent state randomness
    sim_seed = cuda_sim.SimulationConfig().random_seed
    rng = random.Random(sim_seed)

    for i in range(agent_count):
        # Access agent instance by index
        agent = init_pop[i]
        ix = i % L
        iy = i // L
        agent.setVariableInt("ix", ix)
        agent.setVariableInt("iy", iy)

        # Initial strategy (random based on rate)
        strategy = 1 if rng.random() < initial_coop_ratio else 0
        agent.setVariableUInt("strategy", strategy)
        agent.setVariableUInt("next_strategy", strategy)  # Initialize buffer

        # Initial Culture (C)
        C_value = 0.0
        if C_dist == "uniform":
            C_value = rng.uniform(0, 1)
        elif C_dist == "bimodal":
            C_value = 0.0 if rng.random() < (1.0 - mu) else 1.0
        elif C_dist == "normal":
            np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
            C_value = np.clip(np_rng.normal(mu, sigma), 0, 1)
        elif C_dist == "fixed":
            C_value = mu
        else:
            C_value = rng.uniform(0, 1)  # Default to uniform if unknown
        agent.setVariableFloat("C", C_value)
        agent.setVariableFloat("next_C", C_value)  # Initialize buffer

        agent.setVariableFloat("current_utility", 0.0)

    # Set the initial population data for the simulation
    cuda_sim.setPopulationData(init_pop)

    # --- Create RunPlan for single simulation ---
    # We will manually loop through steps to allow for intermediate snapshots
    # run_plan = pyflamegpu.RunPlan(model)
    # run_plan.setSteps(current_params["steps"])
    # Use the simulation seed obtained after cuda_sim.initialise()
    # run_plan.setRandomSimulationSeed(cuda_sim.SimulationConfig().random_seed)

    # --- Run Simulation with Intermediate Snapshot Export ---
    print(
        f"--- Starting FLAMEGPU Simulation ({current_params['steps']} steps) ---")
    start_time = time.time()

    output_directory = current_params["output_directory"]
    snapshot_filename_pattern = current_params["snapshot_filename_pattern"]
    snapshot_steps = set(current_params["snapshot_steps"]) # Use a set for efficient lookup
    snapshot_interval = current_params["snapshot_interval"]

    os.makedirs(output_directory, exist_ok=True) # Ensure output directory exists

    # Manually loop through simulation steps
    for step in range(current_params["steps"]):
        # Check if current step is in the list of specified steps or matches the interval
        if (step in snapshot_steps) or (snapshot_interval > 0 and step % snapshot_interval == 0):
            snapshot_path = os.path.join(output_directory, snapshot_filename_pattern.format(step))
            print(f"Exporting snapshot at step {step} to: {snapshot_path}")
            try:
                cuda_sim.exportData(snapshot_path)
                print(f"Snapshot at step {step} exported successfully.")
            except Exception as e:
                print(f"Error exporting snapshot at step {step}: {e}")
                import traceback
                traceback.print_exc()

        # Advance the simulation by one step
        cuda_sim.step()

    # Export the final snapshot if it wasn't already exported by interval/list
    if current_params["steps"] not in snapshot_steps and (snapshot_interval == 0 or (current_params["steps"] - 1) % snapshot_interval != 0):
         snapshot_path = os.path.join(output_directory, snapshot_filename_pattern.format(current_params["steps"]))
         print(f"Exporting final snapshot at step {current_params['steps']} to: {snapshot_path}")
         try:
             cuda_sim.exportData(snapshot_path)
             print(f"Final snapshot at step {current_params['steps']} exported successfully.")
         except Exception as e:
             print(f"Error exporting final snapshot: {e}")
             import traceback
             traceback.print_exc()


    end_time = time.time()
    print(f"--- Simulation Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


    # Ensure profiling / memcheck work correctly
    pyflamegpu.cleanup()

    print("\nSingle simulation run and snapshot export complete.")
