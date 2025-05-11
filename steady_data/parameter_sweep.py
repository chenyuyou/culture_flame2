# parameter_sweep.py

import pyflamegpu
import time
import numpy as np
import os
import itertools
from tqdm import tqdm
import traceback

# Import model definition and host functions
from model_definition import (
    create_model,
    define_environment,
    define_messages,
    define_agents,
    define_execution_order
)
from host_functions import InitFunction, StepFunction

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
    "K_C": 0.1,  # Selection strength for cultural update (Fermi rule)
    "p_update_C": 0.1,  # Probability to attempt cultural update per step
    "p_mut_culture": 0.01,  # Probability of random cultural mutation per step
    "p_mut_strategy": 0.001,  # Probability of random strategy mutation per step

    "steps": 2000,  # Number of simulation steps to run
    "seed": None,  # Random seed for reproducibility (None lets FLAMEGPU generate)

    # Ensemble/Batch Run Parameters
    "num_runs": 20, # Number of independent simulation runs for this parameter set
    "output_directory": "results_flamegpu_array_staged", # Directory to save log files
    "steady_state_window": 500 # Add steady state window parameter (used by process_logs.py)
}
# ==============================================================================
# END OF PARAMETER CONFIGURATION
# ==============================================================================


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

    # Log new metrics
    log_cfg.logEnvironment("segregation_index_strategy")
    log_cfg.logEnvironment("segregation_index_type")
    log_cfg.logEnvironment("boundary_fraction")
    log_cfg.logEnvironment("boundary_coop_rate")
    log_cfg.logEnvironment("bulk_coop_rate")

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

            # --- Modified logic for generating ID part ---
            # Handle K_C specifically to avoid splitting
            if key == 'K_C':
                id_part = f"K_C{value:.4g}".replace('.', 'p').replace('-', 'm')
            else:
                # Create a meaningful ID part for other parameters
                id_part = f"{key}{value:.4g}" if isinstance(
                    value, float) else f"{key}{value}"
                id_part = id_part.replace('.', 'p').replace('-', 'm') # Apply replacements

            combo_id_parts.append(id_part)
            # --- End of modified logic ---

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
            traceback.print_exc()
            # Continue to the next combination even if one fails

    print("\n--- All FLAMEGPU2 Simulations for Sweep Finished ---")
    return base_output_dir # Return the root directory where all subdirectories were created

