# run_phasediagram_scan.py

import argparse
import numpy as np
import os
from parameter_sweep import run_parameter_sweep, SIMULATION_PARAMETERS

# ==========================================================================
# MAIN EXECUTION BLOCK FOR RUNNING PHASE DIAGRAM SCAN SIMULATIONS
# ==========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLAMEGPU2 Cultural Game Phase Diagram Scan Simulations.")
    parser.add_argument("--output_dir", type=str, default="results_flamegpu_phasediagram_ensemble_raw",
                        help="Base directory to save phase diagram simulation logs. Default: results_flamegpu_phasediagram_ensemble_raw")
    parser.add_argument("--steps", type=int, default=500, # Default steps for PD scan
                        help=f"Number of simulation steps per run. Default: 3000")
    parser.add_argument("--num_runs", type=int, default=1000, # Default runs for PD scan
                        help=f"Number of independent runs per parameter set. Default: 5")
    # Add other base parameters if you want to control them from command line
    # parser.add_argument("--L", type=int, default=40, help="Grid size L for PD scan") # Fixed L for PD
    # ...
    args = parser.parse_args()

    # Update base parameters from command line arguments
    base_params_phasediagram = SIMULATION_PARAMETERS.copy()
    base_params_phasediagram["output_directory"] = args.output_dir
    base_params_phasediagram["steps"] = args.steps
    base_params_phasediagram["num_runs"] = args.num_runs
    base_params_phasediagram["L"] = 50 # Fixed L for PD scan (can make this a command line arg if needed)
    base_params_phasediagram["p_mut_culture"] = 0.005  
    base_params_phasediagram["p_mut_strategy"] = 0.001  
    # Update other params if added to parser

    # Define the parameter sweep configuration for the phase diagram scan
    # Remove parameters that will be swept from the base params for clarity
    phasediagram_param1_name = 'b'
    phasediagram_param2_name = 'K_C'
    sweep_params_config_phasediagram = {
        phasediagram_param1_name: np.arange(1.1, 8.6 + 0.25, 0.25),
        phasediagram_param2_name: np.geomspace(0.0001, 1.0, num=30),
    }

    # Ensure sweep parameters are not in base_params_phasediagram
    for key in sweep_params_config_phasediagram.keys():
        if key in base_params_phasediagram:
            del base_params_phasediagram[key]

    print("\n--- Running Phase Diagram Scan Simulations ---")
    phasediagram_log_dir = run_parameter_sweep(base_params_phasediagram, sweep_params_config_phasediagram)
    print(f"\nPhase diagram raw logs saved to base directory: '{phasediagram_log_dir}'")
    print(f"You can now run 'python process_logs.py --log_dir {phasediagram_log_dir} --process_type steady_state' to process the logs for phase diagrams.")

    print("\nPhase diagram simulation runs complete.")

