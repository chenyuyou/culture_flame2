# process_logs.py

import os
import json
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
# No longer need glob for this approach
# import glob

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


def process_flamegpu_logs(log_directory, processing_params):
    """
    Reads FLAMEGPU JSON logs from a directory, extracts environment data,
    calculates steady state averages/SEMs/Chi, AND calculates average time series
    across runs.
    """
    print(f"\n--- Processing FLAMEGPU logs from base directory: {log_directory} ---")

    all_processed_dataframes = []
    # all_time_series_dataframes = [] # This list is not needed anymore, we aggregate directly

    steady_state_window = processing_params.get("steady_state_window", 500)

    param_combo_dirs = [d for d in os.listdir(log_directory) if os.path.isdir(os.path.join(log_directory, d))]

    if not param_combo_dirs:
        print(f"Error: No parameter combination subdirectories found in {log_directory}.")
        return pd.DataFrame(), {} # Return empty DataFrame and empty dict for time series

    print(f"Found {len(param_combo_dirs)} parameter combination directories.")

    # Dictionary to store aggregated time series data per parameter combination
    aggregated_time_series = {}

    for combo_subdir_name in tqdm(param_combo_dirs, desc="Processing Parameter Combinations"):
        combo_dir_path = os.path.join(log_directory, combo_subdir_name)

        all_run_data_for_combo = []
        run_time_series_list = [] # List to store time series DataFrame for each run in this combo

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
            # print(f"Warning: No JSON log files found in {combo_dir_path}. Skipping combination.")
            continue

        param_set_id = combo_subdir_name
        fixed_params = {}
        parts = param_set_id.split('_')
        for part in parts:
            for key in ['L', 'b', 'K', 'K_C', 'p_update_C', 'p_mut_culture', 'p_mut_strategy', 'mu', 'sigma']:
                if part.startswith(key):
                    try:
                        value_str = part[len(key):].replace('p', '.').replace('m', '-')
                        if '.' in value_str or 'e' in value_str.lower():
                            fixed_params[key] = float(value_str)
                        else:
                            fixed_params[key] = int(value_str)
                        break
                    except ValueError:
                        pass

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
                    # print(f"Warning: No 'steps' data found in {log_path}. Skipping run {run_id}.")
                    continue

                env_data_list = [step.get('environment', {}) for step in steps_data]

                if env_data_list:
                    all_keys = set().union(*(d.keys() for d in env_data_list))
                    consistent_env_data = [{key: step.get(key, np.nan) for key in all_keys} for step in env_data_list]
                    log_df = pd.DataFrame(consistent_env_data)
                    log_df['step'] = log_df.index # Add step number as a column
                else:
                    log_df = pd.DataFrame()

                if log_df.empty:
                     # print(f"Warning: DataFrame created from environment data is empty for {log_path}. Skipping run {run_id}.")
                     continue

                # --- Process for Steady State ---
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

                # --- Store Time Series for this run ---
                # Add parameter columns to the time series DataFrame for this run
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
                import traceback
                traceback.print_exc()
                continue

        # --- Aggregate Steady State Results for this combo ---
        if all_run_data_for_combo:
            processed_df_combo = pd.DataFrame(all_run_data_for_combo)

            # Calculate SEM across runs and Chi for this combo
            if not processed_df_combo.empty:
                L = fixed_params.get('L')
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
                    means_across_runs = steady_state_means_for_chi[log_key]

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

        # --- Aggregate Time Series Results for this combo ---
        if run_time_series_list: # Check if the list is not empty
            # Concatenate time series data from all runs for this combo
            combo_time_series_df = pd.concat(run_time_series_list, ignore_index=True)

            # Group by step and calculate mean and SEM for each metric
            # We only need the metrics we plan to plot over time
            metrics_for_time_series = ['coop_rate', 'avg_C', 'segregation_index_strategy', 'segregation_index_type', 'boundary_fraction', 'boundary_coop_rate', 'bulk_coop_rate'] # Include all logged metrics for time series

            # Filter for metrics that actually exist in the concatenated DataFrame
            existing_metrics_for_ts = [metric for metric in metrics_for_time_series if metric in combo_time_series_df.columns]

            if existing_metrics_for_ts: # Check if there are any metrics to aggregate
                aggregated_ts_data_combo = combo_time_series_df.groupby('step').agg({
                    metric: ['mean', 'sem'] for metric in existing_metrics_for_ts
                })

                # Flatten the multi-level columns
                aggregated_ts_data_combo.columns = ['_'.join(col).strip() for col in aggregated_ts_data_combo.columns.values]

                # Add parameter columns to the aggregated time series DataFrame
                # Take parameters from the first row (they are the same for all rows in this combo)
                if not combo_time_series_df.empty:
                     first_row_params = combo_time_series_df.iloc[0][list(fixed_params.keys()) + ['param_set_id']].to_dict()
                     for key, value in first_row_params.items():
                          aggregated_ts_data_combo[key] = value

                aggregated_time_series[param_set_id] = aggregated_ts_data_combo.reset_index() # Add step back as a column
            # else:
                # print(f"Warning: No relevant metrics found for time series aggregation in {combo_subdir_name}.")


    # Concatenate steady state results from all parameter combinations
    combined_df = pd.concat(all_processed_dataframes, ignore_index=True) if all_processed_dataframes else pd.DataFrame()

    print(f"\nFinished processing logs from {len(param_combo_dirs)} parameter combinations.")
    print(f"Combined Steady State DataFrame contains {len(combined_df)} rows.")
    print(f"Aggregated Time Series data available for {len(aggregated_time_series)} parameter combinations.")


    # Return both the combined steady state DataFrame and the aggregated time series dictionary
    return combined_df, aggregated_time_series

# ==========================================================================
# MAIN EXECUTION BLOCK FOR PROCESSING LOGS
# ==========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FLAMEGPU2 Cultural Game Simulation Logs.")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Base directory containing subdirectories of raw JSON logs from run_simulations.py")
    parser.add_argument("--output_file", type=str, default="processed_sweep_results.pkl",
                        help="Output pickle file name for the processed DataFrame. Default: processed_sweep_results.pkl")
    parser.add_argument("--steady_state_window", type=int, default=PROCESSING_PARAMETERS["steady_state_window"],
                        help=f"Number of steps to average over for steady state. Default: {PROCESSING_PARAMETERS['steady_state_window']}")

    args = parser.parse_args()

    # Update processing parameters from command line arguments
    processing_params = PROCESSING_PARAMETERS.copy()
    processing_params["steady_state_window"] = args.steady_state_window

    # Process the logs - now returns two things
    combined_processed_data, aggregated_time_series_data = process_flamegpu_logs(args.log_dir, processing_params)

    # Save the combined Steady State DataFrame
    if not combined_processed_data.empty:
        output_path_steady_state = os.path.join(args.log_dir, args.output_file)
        try:
            with open(output_path_steady_state, 'wb') as f:
                pickle.dump(combined_processed_data, f)
            print(f"\nProcessed steady state results saved to {output_path_steady_state}")
        except Exception as e:
            print(f"\nError saving processed steady state results to {output_path_steady_state}: {e}")
    else:
        print("\nNo steady state data to save.")

    # Save the Aggregated Time Series data
    if aggregated_time_series_data:
        # Use a different filename for time series data
        output_file_ts = args.output_file.replace(".pkl", "_timeseries.pkl")
        output_path_ts = os.path.join(args.log_dir, output_file_ts)
        try:
            with open(output_path_ts, 'wb') as f:
                pickle.dump(aggregated_time_series_data, f)
            print(f"\nAggregated time series data saved to {output_path_ts}")
        except Exception as e:
            print(f"\nError saving aggregated time series data to {output_path_ts}: {e}")
    else:
        print("\nNo aggregated time series data to save.")

