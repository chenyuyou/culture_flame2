# run_main_scan.py

import argparse
import numpy as np
import os
from parameter_sweep import run_parameter_sweep, SIMULATION_PARAMETERS

# ==========================================================================
# MAIN EXECUTION BLOCK FOR RUNNING MAIN SCAN SIMULATIONS
# ==========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLAMEGPU2 Cultural Game Main Scan Simulations.")
    parser.add_argument("--output_dir", type=str, default="results_flamegpu_main_scan_raw",
                        help="Base directory to save main scan simulation logs. Default: results_flamegpu_main_scan_raw")
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
    base_params_main["p_mut_culture"] = 0.005  
    base_params_main["p_mut_strategy"] = 0.001  
    # Update other params if added to parser
    values_part1 = np.arange(1.1, 3.6 + 0.1, 0.1) # 从 1.1 到 3.6，步长 0.1，包含 3.6
# 生成第二个范围的数值
    values_part2 = np.arange(3.7, 8.7 + 0.5, 0.5) # 从 3.7 到 8.7，步长 0.5，包含 8.7

    # Define the parameter sweep configuration for the main scan
    # Remove parameters that will be swept from the base params for clarity
#    values_part1 = np.arange(1.1, 3.7, 0.2) # 从 1.1 到 3.6，步长 0.1，包含 3.6
# 生成第二个范围的数值
#    values_part2 = np.arange(3.7, 8.7 + 0.5, 0.5)
# 合并所有值并去重

    sweep_params_config_main = {
        'L': [20, 30, 40, 50],
        'b': np.unique(np.concatenate((values_part1, values_part2)))
    }

    # Ensure sweep parameters are not in base_params_main if they are swept
    for key in sweep_params_config_main.keys():
        if key in base_params_main:
            del base_params_main[key]

    print("\n--- Running Main Scan Simulations ---")
    main_scan_log_dir = run_parameter_sweep(base_params_main, sweep_params_config_main)
    print(f"\nMain scan raw logs saved to base directory: '{main_scan_log_dir}'")
    print(f"You can now run 'python process_logs.py --log_dir {main_scan_log_dir} --process_type steady_state' to process the steady state data.")
    print(f"You can also run 'python process_logs.py --log_dir {main_scan_log_dir} --process_type time_series' to process the time series data.")

    print("\nMain scan simulation runs complete.")

