# cultural_game_simulation_snapshot.py (Modified for single run with corrected snapshot logic)

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
import itertools
import argparse
import shutil
import traceback

# Import from the NEW CUDA file with STAGED functions
try:
    from cultural_game_cuda_array import (
        output_state_func,
        calculate_utility_func,
        calculate_local_metrics_func,
        decide_updates_func,
        mutate_func,
        advance_func
    )
except ImportError:
    print("Error: cultural_game_cuda_array.py not found. Please ensure it's in the same directory.")
    sys.exit(1)


# ==============================================================================
# SIMULATION PARAMETERS - CONFIGURE YOUR SINGLE RUN DEFAULTS HERE
# These will be used if not overridden by command line arguments
# ==============================================================================
DEFAULT_PARAMETERS = {
    "L": 50,  # Grid size (L x L)
    "initial_coop_ratio": 0.5,  # Initial fraction of cooperators
    "b": 1.6,  # Benefit of cooperation (for the recipient) - Default value
    "K": 0.01,  # Selection strength for strategy update (Fermi rule)

    # Cultural Parameters (Initial Distribution)
    "C_dist": "uniform",  # Options: "uniform", "normal", "bimodal", "fixed"
    # Meaning depends on C_dist (e.g., mean for normal, fixed value, p(C=1) for bimodal)
    "mu": 0.5,
    "sigma": 0.1,  # Std dev for normal distribution

    # Cultural Evolution Parameters
    "K_C": 0.1,  # Noise in cultural update rule (Fermi rule)
    "p_update_C": 0.1,  # Probability to attempt cultural update per step
    "p_mut_culture": 0.005,  # Probability of random cultural mutation per step
    "p_mut_strategy": 0.001,  # Probability of random strategy mutation per step

    "steps": 2000,  # Number of simulation steps to run
    "seed": None,  # Random seed for reproducibility (None lets FLAMEGPU generate, but we'll override this)

    "output_directory": "snapshot_output", # Base directory to save the snapshot files
    "snapshot_filename_pattern": "snapshot_step_{}.json", # Filename pattern for snapshots
    "snapshot_steps": [], # List of steps at which to save snapshots (empty means only final)
    "snapshot_interval": 0 # Save snapshot every N steps (0 means no interval saving)
}
# ==============================================================================
# END OF DEFAULT PARAMETER CONFIGURATION
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

    # Reporting properties
    env.newPropertyUInt("agent_count", params["L"] * params["L"])
    env.newPropertyUInt("cooperators", 0)
    env.newPropertyUInt("defectors", 0)
    env.newPropertyFloat("coop_rate", 0.0)
    env.newPropertyFloat("defection_rate", 0.0)
    env.newPropertyFloat("std_coop_rate", 0.0)
    env.newPropertyFloat("std_defection_rate", 0.0)
    env.newPropertyFloat("avg_C", 1.0)
    env.newPropertyFloat("std_C", 1.0)

    # New reporting properties
    env.newPropertyFloat("segregation_index_strategy", 0.0)
    env.newPropertyFloat("segregation_index_type", 0.0)
    env.newPropertyFloat("boundary_fraction", 0.0)
    env.newPropertyFloat("boundary_coop_rate", 0.0)
    env.newPropertyFloat("bulk_coop_rate", 0.0)
    env.newPropertyUInt("total_boundary_agents", 0)
    env.newPropertyUInt("total_boundary_cooperators", 0)
    env.newPropertyUInt("total_bulk_cooperators", 0)
    env.newPropertyUInt("total_bulk_agents", 0)
    env.newPropertyUInt("random_seed", 0) # Define the random_seed property

    return env


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
            # Ensure these operations are safe if agent_count_gpu is 0
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
                bulk_coop_rate = 0.0
                FLAMEGPU.environment.setPropertyFloat("bulk_coop_rate", bulk_coop_rate)


# --- Main Simulation Execution and Snapshot Export ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single FLAMEGPU2 Cultural Game Simulation and export snapshots.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_PARAMETERS["output_directory"],
                        help=f"Base directory to save the snapshot files. Default: {DEFAULT_PARAMETERS['output_directory']}")
    parser.add_argument("--snapshot_filename_pattern", type=str, default=DEFAULT_PARAMETERS["snapshot_filename_pattern"],
                        help=f"Filename pattern for the snapshot JSONs (use {{}} for step number). Default: {DEFAULT_PARAMETERS['snapshot_filename_pattern']}")
    parser.add_argument("--steps", type=int, default=DEFAULT_PARAMETERS["steps"],
                        help=f"Number of simulation steps to run. Default: {DEFAULT_PARAMETERS['steps']}")
    # Keep the seed argument, but we will generate a unique one if None
    parser.add_argument("--seed", type=int, default=DEFAULT_PARAMETERS["seed"],
                        help=f"Random seed for the simulation. If None, a unique time-based seed is generated. Default: {DEFAULT_PARAMETERS['seed']}")
    parser.add_argument("--snapshot_steps", type=int, nargs='*', default=DEFAULT_PARAMETERS["snapshot_steps"],
                        help="List of specific steps at which to save snapshots (e.g., --snapshot_steps 100 500 1000). Includes step 0 and the final step.")
    parser.add_argument("--snapshot_interval", type=int, default=DEFAULT_PARAMETERS["snapshot_interval"],
                        help="Save snapshot every N steps (e.g., --snapshot_interval 100). 0 means no interval saving. Includes step 0 and the final step if they are multiples of N.")

    # Add other parameters you want to control from command line
    parser.add_argument("--L", type=int, default=DEFAULT_PARAMETERS["L"], help=f"Grid size L. Default: {DEFAULT_PARAMETERS['L']}")
    # This script now requires a single --b value
    parser.add_argument("--b", type=float, required=True, help="Benefit of cooperation b for this specific run.")
    parser.add_argument("--K", type=float, default=DEFAULT_PARAMETERS["K"], help=f"Selection strength K. Default: {DEFAULT_PARAMETERS['K']}")
    parser.add_argument("--K_C", type=float, default=DEFAULT_PARAMETERS["K_C"], help=f"Cultural noise K_C. Default: {DEFAULT_PARAMETERS['K_C']}")
    parser.add_argument("--p_update_C", type=float, default=DEFAULT_PARAMETERS["p_update_C"], help=f"Cultural update probability p_update_C. Default: {DEFAULT_PARAMETERS['p_update_C']}")

    parser.add_argument("--p_mut_culture", type=float, default=DEFAULT_PARAMETERS["p_mut_culture"], help=f"Cultural mutation probability p_mut_culture. Default: {DEFAULT_PARAMETERS['p_mut_culture']}")
    parser.add_argument("--p_mut_strategy", type=float, default=DEFAULT_PARAMETERS["p_mut_strategy"], help=f"Strategy mutation probability p_mut_strategy. Default: {DEFAULT_PARAMETERS['p_mut_strategy']}")
    parser.add_argument("--initial_coop_ratio", type=float, default=DEFAULT_PARAMETERS["initial_coop_ratio"], help=f"Initial cooperation ratio. Default: {DEFAULT_PARAMETERS['initial_coop_ratio']}")
    parser.add_argument("--C_dist", type=str, default=DEFAULT_PARAMETERS["C_dist"], help=f"Initial C distribution. Default: {DEFAULT_PARAMETERS['C_dist']}")
    parser.add_argument("--mu", type=float, default=DEFAULT_PARAMETERS["mu"], help=f"Mean/parameter for C distribution. Default: {DEFAULT_PARAMETERS['mu']}")
    parser.add_argument("--sigma", type=float, default=DEFAULT_PARAMETERS["sigma"], help=f"Std dev for normal C distribution. Default: {DEFAULT_PARAMETERS['sigma']}")

    args = parser.parse_args() # Parse arguments for this single run

    # --- Start of try block for cleanup ---
    try:
        current_params = DEFAULT_PARAMETERS.copy()
        # Update parameters for the current run from command line arguments
        current_params["output_directory"] = args.output_dir
        current_params["snapshot_filename_pattern"] = args.snapshot_filename_pattern
        current_params["steps"] = args.steps
        # Use the seed from args if provided, otherwise generate a unique one
        if args.seed is not None:
            run_seed = args.seed # Use the fixed seed if provided
            print(f"Using fixed seed {run_seed} for b = {args.b}")
        else:
            # Generate a unique seed for this specific run using time
            # Using getrandbits(32) provides a 32-bit unsigned integer seed
            # Seed Python's main RNG once using time before generating the seed
            random.seed(time.time())
            run_seed = random.getrandbits(32)
            print(f"Generated unique seed {run_seed} for b = {args.b}")

        current_params["seed"] = run_seed # Store the actual seed used in params
        # current_params["snapshot_steps"] = args.snapshot_steps # Will process below
        current_params["snapshot_interval"] = args.snapshot_interval
        current_params["L"] = args.L
        current_params["b"] = args.b  # Set the current b value for this run
        current_params["K"] = args.K
        current_params["K_C"] = args.K_C
        current_params["p_update_C"] = args.p_update_C
        current_params["p_mut_culture"] = args.p_mut_culture
        current_params["p_mut_strategy"] = args.p_mut_strategy
        current_params["initial_coop_ratio"] = args.initial_coop_ratio
        current_params["C_dist"] = args.C_dist
        current_params["mu"] = args.mu
        current_params["sigma"] = args.sigma

        # Create a unique output directory for this b value run
        run_output_directory = os.path.join(current_params["output_directory"], f"b_{args.b}")

        # --- Add directory cleanup here ---
        if os.path.exists(run_output_directory):
            print(f"Cleaning existing output directory: {run_output_directory}")
            try:
                shutil.rmtree(run_output_directory)
            except OSError as e:
                print(f"Error: Could not remove directory {run_output_directory} : {e.strerror}")
                # Decide if you want to exit or continue - exiting is safer
                sys.exit(1)
        # --- End of cleanup ---

        os.makedirs(run_output_directory, exist_ok=True) # Recreate the directory

        print(f"\n--- Setting up STAGED Array-Based Cultural Game for b = {args.b} ---")
        print(f"Parameters: {current_params}")
        print(f"Saving snapshots to: {run_output_directory}")

        # Create model and simulation for this run
        model = create_model()
        env = define_environment(model, current_params)
        define_messages(model, current_params)
        define_agents(model)
        define_execution_order(model)

        step_func_instance = StepFunction()
        model.addStepFunction(step_func_instance)

        # Create CUDASimulation
        cuda_sim = pyflamegpu.CUDASimulation(model)

        # --- Set the explicitly generated unique seed ---
        cuda_sim.SimulationConfig().random_seed = run_seed
        # --- End of setting seed ---

        # Get the simulation seed (should be the one we just set)
        sim_seed = cuda_sim.SimulationConfig().random_seed
        # Store seed in environment for reporting/debugging
        env.setPropertyUInt("random_seed", sim_seed)

        # Initialize Python's RNGs using this *same* seed
        # These are only used for initial agent population setup
        rng = random.Random(sim_seed)
        np_rng = np.random.default_rng(sim_seed)


        # --- Initialise Agent Population ---
        L = current_params["L"]
        agent_count = L * L
        initial_coop_ratio = current_params["initial_coop_ratio"]
        C_dist = current_params["C_dist"]
        mu = current_params["mu"]
        sigma = current_params["sigma"]

        init_pop = pyflamegpu.AgentVector(model.Agent("CulturalAgent"), agent_count)

        for j in range(agent_count): # Use 'j' for agent index
            agent = init_pop[j]
            ix = j % L
            iy = j // L
            agent.setVariableInt("ix", ix)
            agent.setVariableInt("iy", iy)
            # Use the Python RNGs initialized with the FLAMEGPU seed
            strategy = 1 if rng.random() < initial_coop_ratio else 0
            agent.setVariableUInt("strategy", strategy)
            agent.setVariableUInt("next_strategy", strategy)

            C_value = 0.0
            if C_dist == "uniform":
                C_value = rng.uniform(0, 1)
            elif C_dist == "bimodal":
                C_value = 0.0 if rng.random() < (1.0 - mu) else 1.0
            elif C_dist == "normal":
                # Use numpy RNG for normal distribution
                C_value = np.clip(np_rng.normal(mu, sigma), 0, 1)
            elif C_dist == "fixed":
                C_value = mu
            else:
                # Default to uniform if C_dist is unknown
                C_value = rng.uniform(0, 1)

            agent.setVariableFloat("C", C_value)
            agent.setVariableFloat("next_C", C_value)
            agent.setVariableFloat("current_utility", 0.0)

        cuda_sim.setPopulationData(init_pop)

        # --- Determine which steps to save snapshots ---
        steps_to_save = set(args.snapshot_steps) # Start with explicitly listed steps
        if args.snapshot_interval > 0:
            # Add steps based on interval, including 0 and final step if they align
            for step in range(0, args.steps + 1, args.snapshot_interval):
                 steps_to_save.add(step)

        # If no specific steps or interval are given, default to saving only the final step
        if not args.snapshot_steps and args.snapshot_interval == 0:
             steps_to_save.add(args.steps)

        # Ensure step 0 is included if steps > 0 and it's requested (or default)
        # This is handled by the interval logic if interval > 0 and includes 0
        # If interval is 0 and snapshot_steps is empty, we add args.steps, not 0.
        # If interval is 0 and snapshot_steps is not empty, user must explicitly add 0.
        # Let's make it explicit: if steps > 0 and 0 is in steps_to_save, save initial state.
        # If steps == 0, only save step 0 if requested.

        print(f"Calculated steps to save snapshots: {sorted(list(steps_to_save))}")


        # --- Run Simulation with Intermediate Snapshot Export ---
        print(
            f"--- Starting FLAMEGPU Simulation for b = {args.b} ({current_params['steps']} steps) ---")
        start_time = time.time()

        snapshot_filename_pattern = current_params["snapshot_filename_pattern"]

        # Save initial state (step 0) if requested
        if 0 in steps_to_save:
             snapshot_path = os.path.join(run_output_directory, snapshot_filename_pattern.format(0))
             print(f"Exporting snapshot at step {0} to: {snapshot_path}")
             try:
                 cuda_sim.exportData(snapshot_path)
                 print(f"Snapshot at step {0} for b = {args.b} exported successfully.")
             except Exception as e:
                 print(f"Error exporting snapshot at step {0} for b = {args.b}: {e}")
                 traceback.print_exc()


        # Run simulation steps
        for step in range(args.steps): # This loop runs for step = 0, 1, ..., args.steps - 1
            cuda_sim.step() # Advance the simulation by one step

            # After step 'step' is completed, the simulation is at state 'step + 1'
            current_sim_step = step + 1

            # Check if the current simulation step (after the step() call) should be saved
            if current_sim_step in steps_to_save:
                snapshot_path = os.path.join(run_output_directory, snapshot_filename_pattern.format(current_sim_step))
                print(f"Exporting snapshot at step {current_sim_step} to: {snapshot_path}")
                try:
                    cuda_sim.exportData(snapshot_path)
                    print(f"Snapshot at step {current_sim_step} for b = {args.b} exported successfully.")
                except Exception as e:
                    print(f"Error exporting snapshot at step {current_sim_step} for b = {args.b}: {e}")
                    traceback.print_exc()


        end_time = time.time()
        print(f"--- Simulation for b = {args.b} Finished ---")
        print(f"Total execution time for b = {args.b}: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"An unexpected error occurred during simulation setup or run for b = {args.b}: {e}")
        traceback.print_exc()
        sys.exit(1) # Exit with a non-zero code to indicate failure
    finally:
        # --- Call cleanup after the single simulation ---
        print(f"Cleaning up FLAMEGPU context for b = {args.b}")
        pyflamegpu.cleanup()
        # --- End of cleanup ---

    sys.exit(0) # Exit with 0 code to indicate success

