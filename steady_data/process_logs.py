# process_logs.py

import os
import json
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import traceback
import re

# ==============================================================================
# DATA PROCESSING PARAMETERS
# These are parameters needed for processing, not simulation run
# ==============================================================================
PROCESSING_PARAMETERS = {
    "steady_state_window": 100, # Number of steps to average over for steady state
    # Add other processing-specific parameters here if needed
}
# ==============================================================================
# END OF DATA PROCESSING PARAMETERS
# ==============================================================================


def process_flamegpu_logs(log_directory, processing_params, process_type='all', scan_type='main_scan'):
    """
    Reads FLAMEGPU JSON logs from a directory structure where subdirectories
    contain parameter information in their names. Extracts environment data,
    calculates steady state averages/SEMs/Chi, AND calculates average time series
    across runs for each parameter combination, based on process_type and scan_type.
    """
    print(f"\n--- Processing FLAMEGPU logs from base directory: {log_directory} ---")

    all_processed_dataframes = []
    aggregated_time_series = {} # Dictionary to store aggregated time series data per parameter combination

    steady_state_window = processing_params.get("steady_state_window", 500)

    # List all subdirectories within the base log directory
    param_combo_dirs = [d for d in os.listdir(log_directory) if os.path.isdir(os.path.join(log_directory, d))]

    if not param_combo_dirs:
        print(f"Error: No parameter combination subdirectories found in {log_directory}.")
        return pd.DataFrame(), {} # Return empty DataFrame and empty dict for time series

    print(f"Found {len(param_combo_dirs)} parameter combination directories.")

    # Define a mapping of parameter prefixes to their expected types
    param_types = {
        'L': int,
        'b': float,
        'K': float,
        'K_C': float,
        'p_update_C': float,
        'p_mut_culture': float,
        'p_mut_strategy': float,
        'mu': float,
        'sigma': float,
        # Add any other parameters that might be in the directory name here
    }

    # Create a regex pattern to match parameter=value pairs
    param_pattern = re.compile(r'(' + '|'.join(re.escape(key) for key in param_types.keys()) + r')((?:-?\d+)(?:p\d+)?(?:m\d+)?(?:\d*))')

    # Define required parameters based on scan type
    if scan_type == 'phasediagram_scan':
        required_for_plotting = ['b', 'K_C'] # For phase diagrams, both b and K_C are required
    elif scan_type == 'main_scan':
        required_for_plotting = ['b'] # For main scan, only b is required for typical plots
    else:
        required_for_plotting = [] # No required parameters for unknown scan types


    for combo_subdir_name in tqdm(param_combo_dirs, desc="Processing Parameter Combinations"):
        combo_dir_path = os.path.join(log_directory, combo_subdir_name)

        # --- Parameter Parsing using Regex ---
        param_set_id = combo_subdir_name
        fixed_params = {}

        matches = param_pattern.findall(combo_subdir_name)

        for match in matches:
            key = match[0]
            value_str = match[1]
            cleaned_value_str = value_str.replace('p', '.').replace('m', '-')

            expected_type = param_types.get(key)
            if expected_type:
                try:
                    if expected_type == int:
                        fixed_params[key] = int(cleaned_value_str)
                    elif expected_type == float:
                        fixed_params[key] = float(cleaned_value_str)
                except ValueError:
                    print(f"Warning: Could not parse value '{cleaned_value_str}' for parameter '{key}' from directory name '{combo_subdir_name}'. Skipping this parameter.")
            else:
                 print(f"Warning: Found unexpected parameter key '{key}' in directory name '{combo_subdir_name}'. Skipping.")

        # Check if required parameters are present ONLY if processing steady state or all
        if process_type in ['steady_state', 'all']:
            missing_required = [key for key in required_for_plotting if key not in fixed_params]
            if missing_required:
                print(f"Warning: Required parameters {missing_required} not found in directory name '{combo_subdir_name}'. Skipping steady state processing for this combination.")
                process_steady_state_for_combo = False
            else:
                process_steady_state_for_combo = True
        else:
            process_steady_state_for_combo = False


        # --- Process Log Files within the Parameter Combination Directory ---
        all_run_data_for_combo = []
        run_time_series_list = []

        steady_state_means_for_chi = {
            'coop_rate': [],
            'avg_C': [],
            'segregation_index_strategy': [],
            'segregation_index_type': [],
            'boundary_fraction': [],
            'boundary_coop_rate': [],
            'bulk_coop_rate': [],
        }

        log_filenames = [f for f in os.listdir(combo_dir_path) if f.endswith('.json')]
        try:
            log_filenames.sort(key=lambda f: int(f.split('.')[0]))
        except ValueError:
            log_filenames.sort()

        if not log_filenames:
            print(f"Warning: No JSON log files found in {combo_dir_path}. Skipping combination.")
            continue

        # Determine total steps from the first log file (more robust)
        total_steps = 0
        if log_filenames:
             first_log_path_for_steps = os.path.join(combo_dir_path, log_filenames[0])
             try:
                 with open(first_log_path_for_steps, 'r') as f:
                     first_log_data = json.load(f)
                     total_steps = len(first_log_data.get('steps', []))
             except Exception as e:
                 print(f"Warning: Could not determine total steps from {first_log_path_for_steps}: {e}. Steady state window might be inaccurate.")
                 total_steps = 0


        for run_id, log_filename in enumerate(log_filenames):
            log_path = os.path.join(combo_dir_path, log_filename)

            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)

                steps_data = log_data.get('steps', [])
                if not steps_data:
                    continue

                env_data_list = [step.get('environment', {}) for step in steps_data]

                if env_data_list:
                    all_keys = set().union(*(d.keys() for d in env_data_list))
                    consistent_env_data = [{key: step.get(key, np.nan) for key in all_keys} for step in env_data_list]
                    log_df = pd.DataFrame(consistent_env_data)
                    log_df['step'] = log_df.index
                else:
                    log_df = pd.DataFrame()

                if log_df.empty:
                     continue

                # --- Process for Steady State (if requested and required params are present) ---
                if process_type in ['steady_state', 'all'] and process_steady_state_for_combo:
                    current_run_steps = len(log_df)
                    steady_state_window_actual = min(steady_state_window, current_run_steps)

                    if steady_state_window_actual > 0:
                        steady_state_start_idx = max(0, current_run_steps - steady_state_window_actual)
                        steady_state_df = log_df.iloc[steady_state_start_idx:]

                        if not steady_state_df.empty:
                            run_result = fixed_params.copy()
                            run_result['run_id'] = run_id
                            run_result['steps'] = current_run_steps
                            run_result['steady_state_window'] = steady_state_window_actual
                            run_result['param_set_id'] = param_set_id

                            metrics_to_process = {
                                'coop_rate': 'CooperationRate',
                                'avg_C': 'AverageCulture',
                                'segregation_index_strategy': 'SegregationIndexStrategy',
                                'segregation_index_type': 'SegregationIndexType',
                                'boundary_fraction': 'BoundaryFraction',
                                'boundary_coop_rate': 'BoundaryCoopRate',
                                'bulk_coop_rate': 'BulkCoopRate',
                            }

                            for log_key, result_key_base in metrics_to_process.items():
                                if log_key in steady_state_df.columns:
                                    mean_val = steady_state_df[log_key].mean()
                                    run_result[f'avg_{result_key_base}'] = mean_val
                                    if log_key in steady_state_means_for_chi:
                                        steady_state_means_for_chi[log_key].append(mean_val)
                                else:
                                     run_result[f'avg_{result_key_base}'] = np.nan

                            run_result['ClusterSizeDistribution'] = None # Placeholder
                            all_run_data_for_combo.append(run_result)

                # --- Store Time Series for this run (if requested) ---
                if process_type in ['time_series', 'all']:
                    ts_df = log_df.copy()
                    for key, value in fixed_params.items():
                        ts_df[key] = value
                    ts_df['run_id'] = run_id
                    ts_df['param_set_id'] = param_set_id
                    run_time_series_list.append(ts_df)


            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {log_path}: {e}. File might be corrupted or incomplete. Skipping run {run_id}.")
                continue
            except FileNotFoundError:
                 print(f"Error: File not found at {log_path}. Skipping run {run_id}.")
                 continue
            except Exception as e:
                print(f"Error processing log file {log_path}: {e}")
                traceback.print_exc()
                continue

        # --- Aggregate Steady State Results for this combo (if processed) ---
        if all_run_data_for_combo: # Check if any steady state data was collected for this combo
            processed_df_combo = pd.DataFrame(all_run_data_for_combo)

            # Calculate SEM across runs and Chi for this combo
            if not processed_df_combo.empty:
                L = processed_df_combo['L'].iloc[0] if 'L' in processed_df_combo.columns and not processed_df_combo['L'].empty else np.nan
                N = L * L if pd.notna(L) and L > 0 else np.nan

                metrics_for_chi_and_sem = {
                    'coop_rate': 'CooperationRate',
                    'avg_C': 'AverageCulture',
                    'segregation_index_strategy': 'SegregationIndexStrategy',
                    'segregation_index_type': 'SegregationIndexType',
                    'boundary_fraction': 'BoundaryFraction',
                    'boundary_coop_rate': 'BoundaryCoopRate',
                    'bulk_coop_rate': 'BulkCoopRate',
                }

                for log_key, result_key_base in metrics_for_chi_and_sem.items():
                    means_across_runs = steady_state_means_for_chi.get(log_key, [])

                    if means_across_runs and len(means_across_runs) > 1:
                        sem_across_runs = np.std(means_across_runs, ddof=1) / np.sqrt(len(means_across_runs))
                        processed_df_combo[f'sem_{result_key_base}'] = sem_across_runs
                    else:
                         processed_df_combo[f'sem_{result_key_base}'] = np.nan

                    if pd.notna(N) and N > 0 and means_across_runs and len(means_across_runs) > 1:
                        variance_across_runs = np.var(means_across_runs, ddof=1)
                        processed_df_combo[f'chi_{result_key_base}'] = N * variance_across_runs
                    else:
                         processed_df_combo[f'chi_{result_key_base}'] = np.nan

            all_processed_dataframes.append(processed_df_combo)

        # --- Aggregate Time Series Results for this combo (if processed) ---
        if run_time_series_list: # Check if the list is not empty
            combo_time_series_df = pd.concat(run_time_series_list, ignore_index=True)

            metrics_for_time_series = ['coop_rate', 'avg_C', 'segregation_index_strategy', 'segregation_index_type', 'boundary_fraction', 'boundary_coop_rate', 'bulk_coop_rate']
            existing_metrics_for_ts = [metric for metric in metrics_for_time_series if metric in combo_time_series_df.columns]

            if existing_metrics_for_ts:
                aggregated_ts_data_combo = combo_time_series_df.groupby('step').agg({
                    metric: ['mean', 'sem'] for metric in existing_metrics_for_ts
                })

                aggregated_ts_data_combo.columns = ['_'.join(col).strip() for col in aggregated_ts_data_combo.columns.values]

                if not combo_time_series_df.empty:
                     params_to_add = {key: combo_time_series_df.iloc[0][key] for key in fixed_params.keys() if key in combo_time_series_df.columns}
                     params_to_add['param_set_id'] = combo_time_series_df.iloc[0]['param_set_id'] if 'param_set_id' in combo_time_series_df.columns else param_set_id

                     for key, value in params_to_add.items():
                          aggregated_ts_data_combo[key] = value

                aggregated_time_series[param_set_id] = aggregated_ts_data_combo.reset_index()


    # Concatenate steady state results from all parameter combinations
    combined_df = pd.concat(all_processed_dataframes, ignore_index=True) if all_processed_dataframes else pd.DataFrame()

    print(f"\nFinished processing logs from {len(param_combo_dirs)} parameter combinations.")
    print(f"Combined Steady State DataFrame contains {len(combined_df)} rows.")
    print(f"Aggregated Time Series data available for {len(aggregated_time_series)} parameter combinations.")


    return combined_df, aggregated_time_series

# ==========================================================================
# MAIN EXECUTION BLOCK FOR PROCESSING LOGS
# ==========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FLAMEGPU2 Cultural Game Simulation Logs.")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Base directory containing subdirectories of raw JSON logs from run_simulations.py")
    parser.add_argument("--process_type", type=str, choices=['steady_state', 'time_series', 'all'], default='all',
                        help="Type of data to process: 'steady_state', 'time_series', or 'all'. Default: 'all'")
    parser.add_argument("--scan_type", type=str, choices=['main_scan', 'phasediagram_scan'], default='main_scan',
                        help="Type of parameter scan performed: 'main_scan' or 'phasediagram_scan'. Default: 'main_scan'")
    parser.add_argument("--output_file_steady_state", type=str, default="processed_steady_state_results.pkl",
                        help="Output pickle file name for the processed steady state DataFrame. Default: processed_steady_state_results.pkl")
    parser.add_argument("--output_file_timeseries", type=str, default="aggregated_timeseries_data.pkl",
                        help="Output pickle file name for the aggregated time series data. Default: aggregated_timeseries_data.pkl")
    parser.add_argument("--steady_state_window", type=int, default=PROCESSING_PARAMETERS["steady_state_window"],
                        help=f"Number of steps to average over for steady state. Default: {PROCESSING_PARAMETERS['steady_state_window']}")

    args = parser.parse_args()

    # Update processing parameters from command line arguments
    processing_params = PROCESSING_PARAMETERS.copy()
    processing_params["steady_state_window"] = args.steady_state_window

    # Process the logs based on the requested type and scan type
    if args.process_type == 'steady_state':
        combined_processed_data, _ = process_flamegpu_logs(args.log_dir, processing_params, process_type='steady_state', scan_type=args.scan_type)
        aggregated_time_series_data = {} # Empty time series data
    elif args.process_type == 'time_series':
        _, aggregated_time_series_data = process_flamegpu_logs(args.log_dir, processing_params, process_type='time_series', scan_type=args.scan_type)
        combined_processed_data = pd.DataFrame() # Empty steady state data
    elif args.process_type == 'all':
        combined_processed_data, aggregated_time_series_data = process_flamegpu_logs(args.log_dir, processing_params, process_type='all', scan_type=args.scan_type)


    # Save the combined Steady State DataFrame (if processed)
    if not combined_processed_data.empty:
        output_path_steady_state = os.path.join(args.log_dir, args.output_file_steady_state)
        try:
            with open(output_path_steady_state, 'wb') as f:
                pickle.dump(combined_processed_data, f)
            print(f"\nProcessed steady state results saved to {output_path_steady_state}")
        except Exception as e:
            print(f"\nError saving processed steady state results to {output_path_steady_state}: {e}")
    else:
        if args.process_type in ['steady_state', 'all']:
             print("\nNo steady state data to save.")


    # Save the Aggregated Time Series data (if processed)
    if aggregated_time_series_data:
        output_path_ts = os.path.join(args.log_dir, args.output_file_timeseries)
        try:
            with open(output_path_ts, 'wb') as f:
                pickle.dump(aggregated_time_series_data, f)
            print(f"\nAggregated time series data saved to {output_path_ts}")
        except Exception as e:
            print(f"\nError saving aggregated time series data to {output_path_ts}: {e}")
    else:
        if args.process_type in ['time_series', 'all']:
             print("\nNo aggregated time series data to save.")
