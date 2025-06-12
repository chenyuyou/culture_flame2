# run_parameter_sweep.py

import subprocess
import sys
import argparse
import os
import time
from tqdm import tqdm # For progress bar

# Define the path to the simulation script
SIMULATION_SCRIPT = "cultural_game_simulation_snapshot.py"

# ==============================================================================
# SWEEP PARAMETERS - CONFIGURE YOUR SWEEP DEFAULTS HERE
# These will be used if not overridden by command line arguments
# ==============================================================================
DEFAULT_SWEEP_PARAMETERS = {
    "output_directory": "snapshot_output", # Base directory to save the snapshot files
    "snapshot_filename_pattern": "snapshot_step_{}.json", # Filename pattern for snapshots
    "steps": 2000,  # Number of simulation steps to run for each b value
    "snapshot_steps": [], # List of steps at which to save snapshots (empty means only final)
    "snapshot_interval": 0, # Save snapshot every N steps (0 means no interval saving) # <-- ADDED COMMA HERE

    # Add other default parameters that will be passed to the simulation script
    # These should match the arguments accepted by cultural_game_simulation_snapshot.py
    "L": 50,
    "K": 0.01,
    "K_C": 0.1,
    "p_update_C": 0.1,
    "p_mut_culture": 0.005,
    "p_mut_strategy": 0.001,
    "initial_coop_ratio": 0.5,
    "C_dist": "uniform",
    "mu": 0.5,
    "sigma": 0.1,
    # Note: 'b' and 'seed' are handled specially by the sweep logic
}
# ==============================================================================
# END OF DEFAULT SWEEP PARAMETER CONFIGURATION
# ==============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLAMEGPU2 Cultural Game Simulation parameter sweep using subprocesses.")

    # Parameters for the sweep itself
    parser.add_argument("--b_values", type=str, default=None,
                        help="Comma-separated list of b values to run (e.g., '1.5,1.6,1.7'). Overrides --b_range if provided.")
    parser.add_argument("--b_range", type=float, nargs=3, default=None,
                        help="Range of b values to run: start stop step (e.g., '1.5 2.0 0.1'). Overrides --b_values if provided.")
    # Optional fixed seed for the *entire sweep* (passed to each subprocess)
    # If not provided, each subprocess will generate its own time-based seed
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional fixed random seed to pass to each simulation subprocess. If None, each subprocess generates its own unique seed.")


    # Parameters to pass through to the simulation script
    parser.add_argument("--output_dir", type=str, default=DEFAULT_SWEEP_PARAMETERS["output_directory"],
                        help=f"Base directory to save the snapshot files. Default: {DEFAULT_SWEEP_PARAMETERS['output_directory']}")
    parser.add_argument("--snapshot_filename_pattern", type=str, default=DEFAULT_SWEEP_PARAMETERS["snapshot_filename_pattern"],
                        help=f"Filename pattern for the snapshot JSONs (use {{}} for step number). Default: {DEFAULT_SWEEP_PARAMETERS['snapshot_filename_pattern']}")
    parser.add_argument("--steps", type=int, default=DEFAULT_SWEEP_PARAMETERS["steps"],
                        help=f"Number of simulation steps to run. Default: {DEFAULT_SWEEP_PARAMETERS['steps']}")
    parser.add_argument("--snapshot_steps", type=int, nargs='*', default=DEFAULT_SWEEP_PARAMETERS["snapshot_steps"],
                        help="List of specific steps at which to save snapshots (e.g., --snapshot_steps 100 500 1000).")
    parser.add_argument("--snapshot_interval", type=int, default=DEFAULT_SWEEP_PARAMETERS["snapshot_interval"],
                        help=f"Save snapshot every N steps (e.g., --snapshot_interval 100). 0 means no interval saving. Default: {DEFAULT_SWEEP_PARAMETERS['snapshot_interval']}")

    # Add other parameters to pass through
    parser.add_argument("--L", type=int, default=DEFAULT_SWEEP_PARAMETERS["L"], help=f"Grid size L. Default: {DEFAULT_SWEEP_PARAMETERS['L']}")
    parser.add_argument("--K", type=float, default=DEFAULT_SWEEP_PARAMETERS["K"], help=f"Selection strength K. Default: {DEFAULT_SWEEP_PARAMETERS['K']}")
    parser.add_argument("--K_C", type=float, default=DEFAULT_SWEEP_PARAMETERS["K_C"], help=f"Cultural noise K_C. Default: {DEFAULT_SWEEP_PARAMETERS['K_C']}")
    parser.add_argument("--p_update_C", type=float, default=DEFAULT_SWEEP_PARAMETERS["p_update_C"], help=f"Cultural update probability p_update_C. Default: {DEFAULT_SWEEP_PARAMETERS['p_update_C']}")
    parser.add_argument("--p_mut_culture", type=float, default=DEFAULT_SWEEP_PARAMETERS["p_mut_culture"], help=f"Cultural mutation probability p_mut_culture. Default: {DEFAULT_SWEEP_PARAMETERS['p_mut_culture']}")
    parser.add_argument("--p_mut_strategy", type=float, default=DEFAULT_SWEEP_PARAMETERS['p_mut_strategy'], help=f"Strategy mutation probability p_mut_strategy. Default: {DEFAULT_SWEEP_PARAMETERS['p_mut_strategy']}")
    parser.add_argument("--initial_coop_ratio", type=float, default=DEFAULT_SWEEP_PARAMETERS["initial_coop_ratio"], help=f"Initial cooperation ratio. Default: {DEFAULT_SWEEP_PARAMETERS['initial_coop_ratio']}")
    parser.add_argument("--C_dist", type=str, default=DEFAULT_SWEEP_PARAMETERS["C_dist"], help=f"Initial C distribution. Default: {DEFAULT_SWEEP_PARAMETERS['C_dist']}")
    parser.add_argument("--mu", type=float, default=DEFAULT_SWEEP_PARAMETERS["mu"], help=f"Mean/parameter for C distribution. Default: {DEFAULT_SWEEP_PARAMETERS['mu']}")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SWEEP_PARAMETERS["sigma"], help=f"Std dev for normal C distribution. Default: {DEFAULT_SWEEP_PARAMETERS['sigma']}")


    args = parser.parse_args()

    # Determine the list of b values to run
    b_to_run = []
    if args.b_range:
        start, stop, step = args.b_range
        # Use a small tolerance for floating point comparison at the stop value
        num_steps = int(round((stop - start) / step)) + 1
        b_to_run = [start + i * step for i in range(num_steps)]
        # Round the values to avoid floating point issues in directory names
        b_to_run = [round(val, 5) for val in b_to_run]
    elif args.b_values:
        try:
            b_to_run = [float(b_str) for b_str in args.b_values.split(',')]
        except ValueError:
            print("Error: --b_values must be a comma-separated list of numbers.")
            sys.exit(1)
    else:
        print("Error: Either --b_values or --b_range must be provided for the sweep.")
        sys.exit(1)

    print(f"Starting parameter sweep for b values: {b_to_run}")

    # Get the path to the simulation script
    simulation_script_path = os.path.join(os.path.dirname(__file__), SIMULATION_SCRIPT)
    if not os.path.exists(simulation_script_path):
        print(f"Error: Simulation script not found at {simulation_script_path}")
        sys.exit(1)

    # Loop through the b values and launch a subprocess for each
    for b_val in tqdm(b_to_run, desc="Running b values"):
        print(f"\n--- Launching simulation for b = {b_val} ---")

        # Construct the command to run the simulation script
        command = [
            sys.executable, # Use the same python interpreter
            simulation_script_path,
            "--b", str(b_val), # Pass the current b value
            "--output_dir", args.output_dir,
            "--snapshot_filename_pattern", args.snapshot_filename_pattern,
            "--steps", str(args.steps),
            "--snapshot_interval", str(args.snapshot_interval),
            "--L", str(args.L),
            "--K", str(args.K),
            "--K_C", str(args.K_C),
            "--p_update_C", str(args.p_update_C),
            "--p_mut_culture", str(args.p_mut_culture),
            "--p_mut_strategy", str(args.p_mut_strategy),
            "--initial_coop_ratio", str(args.initial_coop_ratio),
            "--C_dist", args.C_dist,
            "--mu", str(args.mu),
            "--sigma", str(args.sigma),
        ]

        # Add snapshot_steps if provided
        if args.snapshot_steps:
             command.append("--snapshot_steps")
             command.extend(map(str, args.snapshot_steps))

        # Add the fixed seed if provided to the sweep script
        if args.seed is not None:
             command.extend(["--seed", str(args.seed)])
             print(f"Passing fixed seed {args.seed} to subprocess.")
        else:
             # If no fixed seed for the sweep, the subprocess will generate its own unique seed
             print("Allowing subprocess to generate a unique seed.")


        print(f"Running command: {' '.join(command)}")

        try:
            # Run the command as a subprocess
            # capture_output=True captures stdout/stderr, text=True decodes as text
            # check=True raises CalledProcessError if the subprocess returns non-zero exit code
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Simulation for b = {b_val} finished successfully.")
            # Optional: Print subprocess output
            # print("--- Subprocess stdout ---")
            # print(result.stdout)
            # print("--- Subprocess stderr ---")
            # print(result.stderr)

        except subprocess.CalledProcessError as e:
            print(f"Error: Simulation for b = {b_val} failed with exit code {e.returncode}")
            print("--- Subprocess stdout ---")
            print(e.stdout)
            print("--- Subprocess stderr ---")
            print(e.stderr)
            # Decide if you want to stop the sweep or continue
            # sys.exit(1) # Exit the sweep script on first error
            print("Continuing with the next parameter...")
        except FileNotFoundError:
             print(f"Error: The script {simulation_script_path} was not found.")
             sys.exit(1)
        except Exception as e:
             print(f"An unexpected error occurred while launching or running the subprocess for b = {b_val}: {e}")
             sys.exit(1)


    print("\nParameter sweep complete.")

