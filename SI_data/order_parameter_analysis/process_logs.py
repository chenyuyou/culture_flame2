# process_logs.py

import os
import json
import sys
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import traceback

def process_flamegpu_logs(output_directory, steady_state_window=500):
    """
    Reads FLAMEGPU JSON logs from a single parameter set directory,
    extracts environment data from 'steps', calculates averages/SEMs
    for steady state, and returns a DataFrame.
    """
    print(f"\n--- Processing FLAMEGPU logs from {output_directory} ---")

    all_run_data = []

    # List log files and sort them numerically by run index (assuming filenames are like 0.json, 1.json, ...)
    log_files = [f for f in os.listdir(output_directory) if f.endswith('.json')]
    try:
        log_files.sort(key=lambda f: int(f.split('.')[0]))
    except ValueError:
        print(f"Warning: Log files in {output_directory} do not appear to be named numerically (e.g., 0.json, 1.json). Sorting alphabetically.")
        log_files.sort()

    if not log_files:
        print(f"Error: No JSON log files found in {output_directory}.")
        return pd.DataFrame() # Return empty DataFrame

    # Attempt to infer parameters from the directory name (assuming sweep structure)
    # This is a heuristic and might need adjustment based on your naming convention
    dir_name = os.path.basename(output_directory)
    inferred_params = {}
    # Split by '_' and then by parameter name prefixes (e.g., 'L', 'b', 'K_C')
    param_parts = dir_name.split('_')
    for part in param_parts:
        if part.startswith('L'):
            try:
                inferred_params['L'] = int(part[1:])
            except ValueError:
                pass
        elif part.startswith('b'):
            try:
                inferred_params['b'] = float(part[1:].replace('p', '.'))
            except ValueError:
                pass
        elif part.startswith('K_C'):
             try:
                 inferred_params['K_C'] = float(part[3:].replace('p', '.'))
             except ValueError:
                 pass
        # Add other parameter prefixes as needed

    # If L is not inferred, try to get it from the first log file (less reliable)
    L = inferred_params.get('L')
    if L is None and log_files:
         try:
             with open(os.path.join(output_directory, log_files[0]), 'r') as f:
                 log_data = json.load(f)
                 if 'environment' in log_data and 'GRID_DIM_L' in log_data['environment']:
                     L = log_data['environment']['GRID_DIM_L']
                     inferred_params['L'] = L
         except Exception as e:
             print(f"Warning: Could not infer L from directory name or first log file: {e}")


    N = L * L if L is not None else np.nan


    # Store steady state averages for susceptibility calculation
    coop_rate_steady_state_means = []
    avg_C_steady_state_means = []
    segregation_index_type_steady_state_means = [] # Overall S

    # **NEW** Lists for group-specific steady state means (4 types)
    coop_rate_type_1_steady_state_means = []
    coop_rate_type_2_steady_state_means = []
    coop_rate_type_3_steady_state_means = []
    coop_rate_type_4_steady_state_means = []
    segregation_index_type_1_steady_state_means = []
    segregation_index_type_2_steady_state_means = []
    segregation_index_type_3_steady_state_means = []
    segregation_index_type_4_steady_state_means = []


    for run_id, log_file in enumerate(tqdm(log_files, desc="Processing logs")):
        log_path = os.path.join(output_directory, log_file)
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)

            # Extract environment data from the 'steps' list
            steps_data = log_data.get('steps', [])
            if not steps_data:
                # print(f"Warning: No 'steps' data found in {log_file}. Skipping.")
                continue

            # Collect environment data for each step
            env_data_list = [step.get('environment', {}) for step in steps_data]

            # Convert list of dictionaries to DataFrame
            if env_data_list:
                # Get all unique keys from all step dictionaries
                all_keys = set().union(*(d.keys() for d in env_data_list))
                # Create a list of dictionaries with consistent keys (fill missing with NaN)
                consistent_env_data = [{key: step.get(key, np.nan) for key in all_keys} for step in env_data_list]
                log_df = pd.DataFrame(consistent_env_data)
            else:
                log_df = pd.DataFrame() # Should be caught by the 'steps_data' check

            if log_df.empty:
                 # print(f"Warning: DataFrame created from environment data is empty for {log_file}. Skipping.")
                 continue

            total_steps_run = len(log_df)
            # Ensure we have enough steps for the steady state window
            if total_steps_run < steady_state_window:
                 # print(f"Warning: Log file {log_file} has only {total_steps_run} steps, steady state window is {steady_state_window}. Using all available steps for steady state.")
                 steady_state_window_actual = total_steps_run
            else:
                 steady_state_window_actual = steady_state_window

            if steady_state_window_actual == 0:
                 # print(f"Warning: Steady state window is 0 for {log_file}. Skipping averaging.")
                 continue

            steady_state_start_idx = max(0, total_steps_run - steady_state_window_actual)
            steady_state_df = log_df.iloc[steady_state_start_idx:]

            if steady_state_df.empty:
                 # print(f"Warning: Steady state window DataFrame is empty for {log_file}. Skipping.")
                 continue

            run_result = inferred_params.copy() # Start with inferred params
            run_result['run_id'] = run_id
            run_result['steps'] = total_steps_run # Add total steps for context
            run_result['steady_state_window'] = steady_state_window_actual # Add actual window size

            # Calculate mean and SEM for relevant metrics
            # Overall Cooperation Rate
            if 'coop_rate' in steady_state_df.columns:
                mean_coop = steady_state_df['coop_rate'].mean()
                sem_coop = steady_state_df['coop_rate'].sem()
                run_result['avg_CooperationRate'] = mean_coop
                run_result['sem_CooperationRate'] = sem_coop
                coop_rate_steady_state_means.append(mean_coop) # Collect for susceptibility
            else:
                 run_result['avg_CooperationRate'] = np.nan
                 run_result['sem_CooperationRate'] = np.nan

            # Average Culture (avg_C)
            if 'avg_C' in steady_state_df.columns:
                mean_avg_C = steady_state_df['avg_C'].mean()
                sem_avg_C = steady_state_df['avg_C'].sem()
                run_result['avg_AverageCulture'] = mean_avg_C
                run_result['sem_AverageCulture'] = sem_avg_C
                avg_C_steady_state_means.append(mean_avg_C) # Collect for susceptibility
            else:
                 run_result['avg_AverageCulture'] = np.nan
                 run_result['sem_AverageCulture'] = np.nan

            # Overall Segregation Index (Average Same Type Neighbor Proportion)
            if 'segregation_index_type' in steady_state_df.columns:
                mean_seg_type = steady_state_df['segregation_index_type'].mean()
                sem_seg_type = steady_state_df['segregation_index_type'].sem()
                run_result['avg_SegregationIndex'] = mean_seg_type # Use this for the overall S
                run_result['sem_SegregationIndex'] = sem_seg_type
                segregation_index_type_steady_state_means.append(mean_seg_type) # Collect for susceptibility
            else:
                 run_result['avg_SegregationIndex'] = np.nan
                 run_result['sem_SegregationIndex'] = np.nan


            # Boundary Fraction
            if 'boundary_fraction' in steady_state_df.columns:
                mean_boundary_frac = steady_state_df['boundary_fraction'].mean()
                sem_boundary_frac = steady_state_df['boundary_fraction'].sem()
                run_result['avg_BoundaryFraction'] = mean_boundary_frac
                run_result['sem_BoundaryFraction'] = sem_boundary_frac
            else:
                 run_result['avg_BoundaryFraction'] = np.nan
                 run_result['sem_BoundaryFraction'] = np.nan

            # Boundary Cooperation Rate
            if 'boundary_coop_rate' in steady_state_df.columns:
                mean_boundary_coop = steady_state_df['boundary_coop_rate'].mean()
                sem_boundary_coop = steady_state_df['boundary_coop_rate'].sem()
                run_result['avg_BoundaryCoopRate'] = mean_boundary_coop
                run_result['sem_BoundaryCoopRate'] = sem_boundary_coop
            else:
                 run_result['avg_BoundaryCoopRate'] = np.nan
                 run_result['sem_BoundaryCoopRate'] = np.nan

            # Bulk Cooperation Rate
            if 'bulk_coop_rate' in steady_state_df.columns:
                mean_bulk_coop = steady_state_df['bulk_coop_rate'].mean()
                sem_bulk_coop = steady_state_df['bulk_coop_rate'].sem()
                run_result['avg_BulkCoopRate'] = mean_bulk_coop
                run_result['sem_BulkCoopRate'] = sem_bulk_coop
            else:
                 run_result['avg_BulkCoopRate'] = np.nan
                 run_result['sem_BulkCoopRate'] = np.nan

            # **NEW** Group-Specific Metrics (4 types)
            # Cooperation Rate Type 1 (0 <= C < 0.25)
            if 'coop_rate_type_1' in steady_state_df.columns:
                mean_coop_type_1 = steady_state_df['coop_rate_type_1'].mean()
                sem_coop_type_1 = steady_state_df['coop_rate_type_1'].sem()
                run_result['avg_CooperationRate_type_1'] = mean_coop_type_1
                run_result['sem_CooperationRate_type_1'] = sem_coop_type_1
                coop_rate_type_1_steady_state_means.append(mean_coop_type_1) # Collect for susceptibility
            else:
                 run_result['avg_CooperationRate_type_1'] = np.nan
                 run_result['sem_CooperationRate_type_1'] = np.nan

            # Cooperation Rate Type 2 (0.25 <= C < 0.5)
            if 'coop_rate_type_2' in steady_state_df.columns:
                mean_coop_type_2 = steady_state_df['coop_rate_type_2'].mean()
                sem_coop_type_2 = steady_state_df['coop_rate_type_2'].sem()
                run_result['avg_CooperationRate_type_2'] = mean_coop_type_2
                run_result['sem_CooperationRate_type_2'] = sem_coop_type_2
                coop_rate_type_2_steady_state_means.append(mean_coop_type_2) # Collect for susceptibility
            else:
                 run_result['avg_CooperationRate_type_2'] = np.nan
                 run_result['sem_CooperationRate_type_2'] = np.nan

            # Cooperation Rate Type 3 (0.5 <= C < 0.75)
            if 'coop_rate_type_3' in steady_state_df.columns:
                mean_coop_type_3 = steady_state_df['coop_rate_type_3'].mean()
                sem_coop_type_3 = steady_state_df['coop_rate_type_3'].sem()
                run_result['avg_CooperationRate_type_3'] = mean_coop_type_3
                run_result['sem_CooperationRate_type_3'] = sem_coop_type_3
                coop_rate_type_3_steady_state_means.append(mean_coop_type_3) # Collect for susceptibility
            else:
                 run_result['avg_CooperationRate_type_3'] = np.nan
                 run_result['sem_CooperationRate_type_3'] = np.nan

            # Cooperation Rate Type 4 (0.75 <= C <= 1.0)
            if 'coop_rate_type_4' in steady_state_df.columns:
                mean_coop_type_4 = steady_state_df['coop_rate_type_4'].mean()
                sem_coop_type_4 = steady_state_df['coop_rate_type_4'].sem()
                run_result['avg_CooperationRate_type_4'] = mean_coop_type_4
                run_result['sem_CooperationRate_type_4'] = sem_coop_type_4
                coop_rate_type_4_steady_state_means.append(mean_coop_type_4) # Collect for susceptibility
            else:
                 run_result['avg_CooperationRate_type_4'] = np.nan
                 run_result['sem_CooperationRate_type_4'] = np.nan

            # Segregation Index Type 1 (0 <= C < 0.25)
            if 'segregation_index_type_1' in steady_state_df.columns:
                mean_seg_type_1 = steady_state_df['segregation_index_type_1'].mean()
                sem_seg_type_1 = steady_state_df['segregation_index_type_1'].sem()
                run_result['avg_SegregationIndex_type_1'] = mean_seg_type_1
                run_result['sem_SegregationIndex_type_1'] = sem_seg_type_1
                segregation_index_type_1_steady_state_means.append(mean_seg_type_1) # Collect for susceptibility
            else:
                 run_result['avg_SegregationIndex_type_1'] = np.nan
                 run_result['sem_SegregationIndex_type_1'] = np.nan

            # Segregation Index Type 2 (0.25 <= C < 0.5)
            if 'segregation_index_type_2' in steady_state_df.columns:
                mean_seg_type_2 = steady_state_df['segregation_index_type_2'].mean()
                sem_seg_type_2 = steady_state_df['segregation_index_type_2'].sem()
                run_result['avg_SegregationIndex_type_2'] = mean_seg_type_2
                run_result['sem_SegregationIndex_type_2'] = sem_seg_type_2
                segregation_index_type_2_steady_state_means.append(mean_seg_type_2) # Collect for susceptibility
            else:
                 run_result['avg_SegregationIndex_type_2'] = np.nan
                 run_result['sem_SegregationIndex_type_2'] = np.nan

            # Segregation Index Type 3 (0.5 <= C < 0.75)
            if 'segregation_index_type_3' in steady_state_df.columns:
                mean_seg_type_3 = steady_state_df['segregation_index_type_3'].mean()
                sem_seg_type_3 = steady_state_df['segregation_index_type_3'].sem()
                run_result['avg_SegregationIndex_type_3'] = mean_seg_type_3
                run_result['sem_SegregationIndex_type_3'] = sem_seg_type_3
                segregation_index_type_3_steady_state_means.append(mean_seg_type_3) # Collect for susceptibility
            else:
                 run_result['avg_SegregationIndex_type_3'] = np.nan
                 run_result['sem_SegregationIndex_type_3'] = np.nan

            # Segregation Index Type 4 (0.75 <= C <= 1.0)
            if 'segregation_index_type_4' in steady_state_df.columns:
                mean_seg_type_4 = steady_state_df['segregation_index_type_4'].mean()
                sem_seg_type_4 = steady_state_df['segregation_index_type_4'].sem()
                run_result['avg_SegregationIndex_type_4'] = mean_seg_type_4
                run_result['sem_SegregationIndex_type_4'] = sem_seg_type_4
                segregation_index_type_4_steady_state_means.append(mean_seg_type_4) # Collect for susceptibility
            else:
                 run_result['avg_SegregationIndex_type_4'] = np.nan
                 run_result['sem_SegregationIndex_type_4'] = np.nan


            # Add placeholders for susceptibility (calculated later) and other data
            run_result['chi_CooperationRate'] = np.nan
            run_result['chi_SegregationIndex'] = np.nan # Overall S
            run_result['chi_AverageCulture'] = np.nan # Avg C
            # Susceptibility for the four types
            run_result['chi_CooperationRate_type_1'] = np.nan
            run_result['chi_CooperationRate_type_2'] = np.nan
            run_result['chi_CooperationRate_type_3'] = np.nan
            run_result['chi_CooperationRate_type_4'] = np.nan
            run_result['chi_SegregationIndex_type_1'] = np.nan
            run_result['chi_SegregationIndex_type_2'] = np.nan
            run_result['chi_SegregationIndex_type_3'] = np.nan
            run_result['chi_SegregationIndex_type_4'] = np.nan
            run_result['ClusterSizeDistribution'] = None # Placeholder for raw data


            all_run_data.append(run_result)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {log_file}: {e}. File might be corrupted or incomplete.")
            continue
        except Exception as e:
            print(f"Error processing log file {log_file}: {e}")
            traceback.print_exc()
            continue

    if not all_run_data:
        print("Error: No valid run data processed from logs.")
        return pd.DataFrame() # Return empty DataFrame

    processed_df = pd.DataFrame(all_run_data)

    # --- Calculate Susceptibility (Chi) = N * Var(<Reporter>) ---
    # This requires data from multiple runs for the SAME parameter set
    if not processed_df.empty and pd.notna(N) and N > 0:
        # Overall Cooperation Rate Susceptibility
        if coop_rate_steady_state_means and len(coop_rate_steady_state_means) > 1:
            variance_coop = np.var(coop_rate_steady_state_means, ddof=1) # Use ddof=1 for sample variance
            processed_df['chi_CooperationRate'] = N * variance_coop

        # Overall Segregation Index Susceptibility
        if segregation_index_type_steady_state_means and len(segregation_index_type_steady_state_means) > 1:
             variance_seg = np.var(segregation_index_type_steady_state_means, ddof=1)
             processed_df['chi_SegregationIndex'] = N * variance_seg

        # Average Culture Susceptibility
        if avg_C_steady_state_means and len(avg_C_steady_state_means) > 1:
             variance_avg_C = np.var(avg_C_steady_state_means, ddof=1)
             processed_df['chi_AverageCulture'] = N * variance_avg_C

        # **NEW** Group-Specific Susceptibility (4 types)
        # Cooperation Rate Type 1 Susceptibility
        if coop_rate_type_1_steady_state_means and len(coop_rate_type_1_steady_state_means) > 1:
            variance_coop_type_1 = np.var(coop_rate_type_1_steady_state_means, ddof=1)
            processed_df['chi_CooperationRate_type_1'] = N * variance_coop_type_1

        # Cooperation Rate Type 2 Susceptibility
        if coop_rate_type_2_steady_state_means and len(coop_rate_type_2_steady_state_means) > 1:
            variance_coop_type_2 = np.var(coop_rate_type_2_steady_state_means, ddof=1)
            processed_df['chi_CooperationRate_type_2'] = N * variance_coop_type_2

        # Cooperation Rate Type 3 Susceptibility
        if coop_rate_type_3_steady_state_means and len(coop_rate_type_3_steady_state_means) > 1:
            variance_coop_type_3 = np.var(coop_rate_type_3_steady_state_means, ddof=1)
            processed_df['chi_CooperationRate_type_3'] = N * variance_coop_type_3

        # Cooperation Rate Type 4 Susceptibility
        if coop_rate_type_4_steady_state_means and len(coop_rate_type_4_steady_state_means) > 1:
            variance_coop_type_4 = np.var(coop_rate_type_4_steady_state_means, ddof=1)
            processed_df['chi_CooperationRate_type_4'] = N * variance_coop_type_4

        # Segregation Index Type 1 Susceptibility
        if segregation_index_type_1_steady_state_means and len(segregation_index_type_1_steady_state_means) > 1:
            variance_seg_type_1 = np.var(segregation_index_type_1_steady_state_means, ddof=1)
            processed_df['chi_SegregationIndex_type_1'] = N * variance_seg_type_1

        # Segregation Index Type 2 Susceptibility
        if segregation_index_type_2_steady_state_means and len(segregation_index_type_2_steady_state_means) > 1:
            variance_seg_type_2 = np.var(segregation_index_type_2_steady_state_means, ddof=1)
            processed_df['chi_SegregationIndex_type_2'] = N * variance_seg_type_2

        # Segregation Index Type 3 Susceptibility
        if segregation_index_type_3_steady_state_means and len(segregation_index_type_3_steady_state_means) > 1:
            variance_seg_type_3 = np.var(segregation_index_type_3_steady_state_means, ddof=1)
            processed_df['chi_SegregationIndex_type_3'] = N * variance_seg_type_3

        # Segregation Index Type 4 Susceptibility
        if segregation_index_type_4_steady_state_means and len(segregation_index_type_4_steady_state_means) > 1:
            variance_seg_type_4 = np.var(segregation_index_type_4_steady_state_means, ddof=1)
            processed_df['chi_SegregationIndex_type_4'] = N * variance_seg_type_4


    # Add a param_set_id column based on inferred parameters for easier grouping later
    id_parts = []
    for param_name, value in inferred_params.items():
        # Use .4g for better float representation in ID
        if isinstance(value, float):
             id_parts.append(f"{param_name}{value:.4g}".replace('.', 'p'))
        elif isinstance(value, int):
             id_parts.append(f"{param_name}{value}")
        else:
             id_parts.append(f"{param_name}{str(value)}")

    param_set_id = "_".join(id_parts) if id_parts else "default_params"
    processed_df['param_set_id'] = param_set_id

    print(f"Finished processing logs for parameter set: {param_set_id}. Processed {len(processed_df)} runs.")

    return processed_df

def main():
    parser = argparse.ArgumentParser(description="Process FLAMEGPU2 Cultural Game logs.")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Base directory containing subdirectories of raw JSON logs from parameter sweeps.")
    parser.add_argument("--output_file", type=str, default="processed_results.pkl",
                        help="Output pickle file to save the processed DataFrame.")
    parser.add_argument("--steady_state_window", type=int, default=500,
                        help="Number of steps at the end of each run to consider for steady state averaging.")
    args = parser.parse_args()

    base_log_dir = args.log_dir
    output_file = args.output_file
    steady_state_window = args.steady_state_window

    if not os.path.isdir(base_log_dir):
        print(f"Error: Log directory '{base_log_dir}' not found.")
        sys.exit(1)

    all_processed_data = []

    # Iterate through all subdirectories in the base log directory
    # Each subdirectory is assumed to contain logs for a single parameter set
    for subdir in os.listdir(base_log_dir):
        subdir_path = os.path.join(base_log_dir, subdir)
        if os.path.isdir(subdir_path):
            processed_df_subdir = process_flamegpu_logs(subdir_path, steady_state_window)
            if not processed_df_subdir.empty:
                all_processed_data.append(processed_df_subdir)

    if not all_processed_data:
        print("No data processed from any subdirectory. Exiting.")
        sys.exit(0)

    # Concatenate DataFrames from all subdirectories
    final_processed_df = pd.concat(all_processed_data, ignore_index=True)

    # Save the final DataFrame to a pickle file
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(final_processed_df, f)
        print(f"\nSuccessfully processed logs and saved results to '{output_file}'")
        print(f"Processed DataFrame shape: {final_processed_df.shape}")
    except Exception as e:
        print(f"Error saving processed data to pickle file: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
