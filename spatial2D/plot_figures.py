# plot_figures.py

import pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# Set default plotting style
sns.set_theme(style="whitegrid")

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_cooperation_rate_vs_b(df, output_dir="figures"):
    """
    Plots steady-state cooperation rate vs. b for different L values (Fig 1 equivalent).
    Assumes the DataFrame has columns 'L', 'b', 'avg_CooperationRate', 'sem_CooperationRate'.
    """
    print("\n--- Plotting Steady-State Cooperation Rate vs. b ---")
    if df.empty:
        print("No data to plot.")
        return

    # Ensure required columns exist
    required_cols = ['L', 'b', 'avg_CooperationRate', 'sem_CooperationRate']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns for this plot. Need: {required_cols}")
        return

    # Group by L and b, calculate mean across runs for plotting
    # (The input df already has avg/sem per run, but we might want to average across runs for the same L/b if num_runs > 1)
    # However, process_logs.py already gives us avg/sem per run, so we can plot those directly or average them.
    # Let's assume we plot the average across runs for the same L/b combination.
    # The DataFrame from process_logs.py already has one row per run, with avg/sem for that run's steady state.
    # To get the overall average/sem for a parameter set (L, b, etc.), we need to group by the parameters.
    # The 'param_set_id' column is useful here.

    # Group by the parameters that define a unique point in the sweep space
    # Assuming L and b are the sweep parameters for this plot
    plot_data = df.groupby(['L', 'b']).agg(
        mean_coop=('avg_CooperationRate', 'mean'), # Average of avg_CooperationRate across runs
        sem_coop=('avg_CooperationRate', 'sem')    # SEM of avg_CooperationRate across runs
    ).reset_index()

    if plot_data.empty:
        print("No aggregated data to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Plot for each unique L value
    for L_val in sorted(plot_data['L'].unique()):
        subset = plot_data[plot_data['L'] == L_val].sort_values('b')
        plt.errorbar(subset['b'], subset['mean_coop'], yerr=subset['sem_coop'], fmt='-o', capsize=3, label=f'L={L_val}')

    plt.xlabel('Benefit of Cooperation (b)')
    plt.ylabel('Steady-State Cooperation Rate')
    plt.title('Steady-State Cooperation Rate vs. b for different L')
    plt.legend(title='Grid Size (L)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xscale('linear') # Or 'log' if appropriate
    plt.ylim(0, 1.05) # Cooperation rate is between 0 and 1

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "cooperation_rate_vs_b.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    plt.show()


def plot_susceptibility_vs_b(df, output_dir="figures"):
    """
    Plots susceptibility (Chi) of cooperation rate vs. b for different L values (Fig 2 equivalent).
    Assumes the DataFrame has columns 'L', 'b', 'chi_CooperationRate'.
    """
    print("\n--- Plotting Susceptibility of Cooperation Rate vs. b ---")
    if df.empty:
        print("No data to plot.")
        return

    required_cols = ['L', 'b', 'chi_CooperationRate']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns for this plot. Need: {required_cols}")
        return

    # The 'chi_CooperationRate' is calculated per parameter set in process_logs.py
    # So we can directly use the DataFrame grouped by parameters.
    # We need to get a unique row for each parameter set (L, b)
    # Since chi is the same for all runs within a param set, we can just take the first one
    plot_data = df.drop_duplicates(subset=['L', 'b']).sort_values(['L', 'b'])

    if plot_data.empty:
        print("No unique parameter sets found for plotting.")
        return

    plt.figure(figsize=(10, 6))

    # Plot for each unique L value
    for L_val in sorted(plot_data['L'].unique()):
        subset = plot_data[plot_data['L'] == L_val]
        plt.plot(subset['b'], subset['chi_CooperationRate'], '-o', label=f'L={L_val}')

    plt.xlabel('Benefit of Cooperation (b)')
    plt.ylabel('Susceptibility ($\chi$)')
    plt.title('Susceptibility of Cooperation Rate vs. b for different L')
    plt.legend(title='Grid Size (L)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xscale('linear') # Or 'log' if appropriate
    plt.yscale('linear') # Susceptibility can vary widely, log scale might be useful

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "susceptibility_vs_b.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    plt.show()


def plot_phase_diagram(df, param1_name, param2_name, metric_name, output_dir="figures"):
    """
    Plots a phase diagram using contourf for a given metric vs two sweep parameters.
    Assumes the DataFrame has columns corresponding to param1_name, param2_name, and metric_name.
    Assumes the input df contains one row per parameter set (e.g., already averaged across runs).
    """
    print(f"\n--- Plotting Phase Diagram: {metric_name} vs. {param1_name} and {param2_name} ---")
    if df.empty:
        print("No data to plot.")
        return

    required_cols = [param1_name, param2_name, metric_name]
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns for this plot. Need: {required_cols}")
        return

    # Ensure we have one value per parameter combination.
    # If the input df has multiple runs per combo, average them first.
    # Assuming the input df is already averaged per combo (e.g., from process_logs output)
    # If not, uncomment the following grouping:
    # plot_data = df.groupby([param1_name, param2_name])[metric_name].mean().reset_index()
    plot_data = df.drop_duplicates(subset=[param1_name, param2_name]).sort_values([param1_name, param2_name])


    if plot_data.empty:
        print("No unique parameter sets found for plotting.")
        return

    # Prepare data for contour plot
    # Need to reshape the data into a grid (X, Y, Z)
    x = plot_data[param1_name].unique()
    y = plot_data[param2_name].unique()
    x.sort()
    y.sort()

    X, Y = np.meshgrid(x, y)
    Z = np.full(X.shape, np.nan) # Initialize Z grid with NaN

    # Fill the Z grid with the metric values
    for _, row in plot_data.iterrows():
        try:
            # Find the indices in the sorted unique arrays
            xi = np.where(x == row[param1_name])[0][0]
            yi = np.where(y == row[param2_name])[0][0]
            Z[yi, xi] = row[metric_name] # Note: Meshgrid Y corresponds to rows (yi), X to columns (xi)
        except IndexError:
            print(f"Warning: Parameter combination ({row[param1_name]}, {row[param2_name]}) not found in meshgrid axes. Skipping.")
            continue


    plt.figure(figsize=(10, 8))

    # Use contourf for filled contours
    # Adjust levels and colormap as needed
    # levels = np.linspace(Z.min(), Z.max(), 20) # Example: 20 levels
    # Or define specific levels for clarity, e.g., for cooperation rate (0 to 1)
    if metric_name == 'avg_CooperationRate':
         levels = np.linspace(0, 1, 21) # Levels from 0 to 1
         cmap = 'viridis' # Or 'RdYlGn' for green=coop, red=defect
    elif metric_name.startswith('chi_'):
         # Susceptibility can be large, log scale might be better for color mapping
         # Use a colormap that highlights peaks
         cmap = 'hot_r' # Reverse hot colormap
         # levels = np.logspace(np.log10(np.nanmin(Z[Z>0])), np.log10(np.nanmax(Z)), 20) # Log levels for Chi
         levels = 20 # Default levels
    elif metric_name.startswith('avg_SegregationIndex'):
         levels = np.linspace(-1, 1, 21) # Segregation index is between -1 and 1
         cmap = 'coolwarm' # Blue for negative, red for positive
    else:
         levels = 20 # Default levels
         cmap = 'viridis'


    contour = plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
    plt.colorbar(contour, label=metric_name.replace('avg_', '').replace('chi_', '$\chi$ of ')) # Label colorbar

    # Add contour lines for clarity (optional)
    # plt.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.5)

    plt.xlabel(param1_name)
    plt.ylabel(param2_name)
    plt.title(f'Phase Diagram: {metric_name.replace("avg_", "").replace("chi_", "Susceptibility of ")}')

    # Set scales based on parameter names if known (e.g., log scale for K_C)
    if param1_name == 'K_C' or param2_name == 'K_C':
        if param1_name == 'K_C':
            plt.xscale('log')
        if param2_name == 'K_C':
            plt.yscale('log')
    # Add other parameter-specific scales if needed

    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"phase_diagram_{metric_name}_vs_{param1_name}_{param2_name}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    plt.show()


def plot_other_metrics_vs_b(df, output_dir="figures"):
    """
    Plots other steady-state metrics vs. b for a fixed L (Fig 4-8 equivalent).
    Assumes the DataFrame has columns 'L', 'b', and various 'avg_' and 'sem_' columns.
    Requires specifying which L value to plot for.
    """
    print("\n--- Plotting Other Metrics vs. b for a Fixed L ---")
    if df.empty:
        print("No data to plot.")
        return

    # We need to select a specific L value to plot these metrics against b
    # Let's find the unique L values and ask the user or pick one
    unique_Ls = sorted(df['L'].dropna().unique())
    if not unique_Ls:
        print("No valid L values found in data.")
        return

    print(f"Available L values: {unique_Ls}")
    # For simplicity, let's plot for the largest L value found
    L_to_plot = unique_Ls[-1]
    print(f"Plotting for L = {L_to_plot}")

    subset_df = df[df['L'] == L_to_plot]

    if subset_df.empty:
        print(f"No data found for L = {L_to_plot}.")
        return

    # Group by b to get overall average/sem for this L
    plot_data = subset_df.groupby('b').agg(
        mean_avg_C=('avg_AverageCulture', 'mean'),
        sem_avg_C=('avg_AverageCulture', 'sem'),
        mean_seg_strat=('avg_SegregationIndexStrategy', 'mean'),
        sem_seg_strat=('avg_SegregationIndexStrategy', 'sem'),
        mean_seg_type=('avg_SegregationIndexType', 'mean'),
        sem_seg_type=('avg_SegregationIndexType', 'sem'),
        mean_bound_frac=('avg_BoundaryFraction', 'mean'),
        sem_bound_frac=('avg_BoundaryFraction', 'sem'),
        mean_bound_coop=('avg_BoundaryCoopRate', 'mean'),
        sem_bound_coop=('avg_BoundaryCoopRate', 'sem'),
        mean_bulk_coop=('avg_BulkCoopRate', 'mean'),
        sem_bulk_coop=('avg_BulkCoopRate', 'sem'),
    ).reset_index().sort_values('b')

    if plot_data.empty:
        print(f"No aggregated data found for L = {L_to_plot}.")
        return

    # Define which metrics to plot
    metrics_to_plot = {
        'Average Culture (avg_C)': ('mean_avg_C', 'sem_avg_C'),
        'Segregation Index (Strategy)': ('mean_seg_strat', 'sem_seg_strat'),
        'Segregation Index (Type)': ('mean_seg_type', 'sem_seg_type'),
        'Boundary Fraction': ('mean_bound_frac', 'sem_bound_frac'),
        'Boundary Cooperation Rate': ('mean_bound_coop', 'sem_bound_coop'),
        'Bulk Cooperation Rate': ('mean_bulk_coop', 'sem_bulk_coop'),
    }

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(nrows=len(metrics_to_plot), ncols=1, figsize=(10, 4 * len(metrics_to_plot)), sharex=True)
    axes = axes.flatten() # Ensure axes is a 1D array

    for i, (metric_label, (mean_col, sem_col)) in enumerate(metrics_to_plot.items()):
        if mean_col in plot_data.columns and sem_col in plot_data.columns:
            axes[i].errorbar(plot_data['b'], plot_data[mean_col], yerr=plot_data[sem_col], fmt='-o', capsize=3)
            axes[i].set_ylabel(metric_label)
            axes[i].grid(True, linestyle='--', alpha=0.6)
            axes[i].set_title(f'{metric_label} vs. b (L={L_to_plot})', fontsize=10) # Add subplot title

    axes[-1].set_xlabel('Benefit of Cooperation (b)') # Set xlabel only on the last subplot

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"other_metrics_vs_b_L{L_to_plot}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    plt.show()


# Add functions for plotting snapshots or cluster size distributions if you implement their data collection
# def plot_snapshot(snapshot_data, output_dir="figures"):
#     pass # Implementation depends on snapshot data format

# def plot_cluster_size_distribution(csd_data, output_dir="figures"):
#     pass # Implementation depends on csd data format


# ==============================================================================
# MAIN EXECUTION BLOCK FOR PLOTTING
# ==============================================================================
# Inside plot_figures.py

# ... (existing imports and functions) ...

def plot_segregation_index_vs_b(df, output_dir="figures", index_type="Type"):
    """
    Plots steady-state segregation index vs. b for different L values (Fig 1b equivalent).
    Assumes the DataFrame has columns 'L', 'b', 'avg_SegregationIndex[Type/Strategy]', 'sem_SegregationIndex[Type/Strategy]'.
    """
    print(f"\n--- Plotting Steady-State Segregation Index ({index_type}) vs. b ---")
    if df.empty:
        print("No data to plot.")
        return

    avg_col = f'avg_SegregationIndex{index_type}'
    sem_col = f'sem_SegregationIndex{index_type}'

    required_cols = ['L', 'b', avg_col, sem_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns for this plot. Need: {required_cols}")
        return

    plot_data = df.groupby(['L', 'b']).agg(
        mean_seg=(avg_col, 'mean'),
        sem_seg=(avg_col, 'sem')
    ).reset_index()

    if plot_data.empty:
        print("No aggregated data to plot.")
        return

    plt.figure(figsize=(10, 6))

    for L_val in sorted(plot_data['L'].unique()):
        subset = plot_data[plot_data['L'] == L_val].sort_values('b')
        plt.errorbar(subset['b'], subset['mean_seg'], yerr=subset['sem_seg'], fmt='-o', capsize=3, label=f'L={L_val}')

    plt.xlabel('Benefit of Cooperation (b)')
    plt.ylabel(f'Steady-State Segregation Index ({index_type})')
    plt.title(f'Steady-State Segregation Index ({index_type}) vs. b for different L')
    plt.legend(title='Grid Size (L)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xscale('linear')
    plt.ylim(-1.05, 1.05) # Segregation index is typically between -1 and 1

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"segregation_index_{index_type.lower()}_vs_b.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    plt.show()


def plot_susceptibility_segregation_vs_b(df, output_dir="figures", index_type="Type"):
    """
    Plots susceptibility (Chi) of segregation index vs. b for different L values (Fig 2b equivalent).
    Assumes the DataFrame has columns 'L', 'b', 'chi_SegregationIndex[Type/Strategy]'.
    """
    print(f"\n--- Plotting Susceptibility of Segregation Index ({index_type}) vs. b ---")
    if df.empty:
        print("No data to plot.")
        return

    chi_col = f'chi_SegregationIndex{index_type}'

    required_cols = ['L', 'b', chi_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns for this plot. Need: {required_cols}")
        return

    plot_data = df.drop_duplicates(subset=['L', 'b']).sort_values(['L', 'b'])

    if plot_data.empty:
        print("No unique parameter sets found for plotting.")
        return

    plt.figure(figsize=(10, 6))

    for L_val in sorted(plot_data['L'].unique()):
        subset = plot_data[plot_data['L'] == L_val]
        plt.plot(subset['b'], subset[chi_col], '-o', label=f'L={L_val}')

    plt.xlabel('Benefit of Cooperation (b)')
    plt.ylabel(f'Susceptibility ($\chi_{{S_{index_type}}}$)') # LaTeX label
    plt.title(f'Susceptibility of Segregation Index ({index_type}) vs. b for different L')
    plt.legend(title='Grid Size (L)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xscale('linear')
    plt.yscale('linear') # Or 'log' if peaks vary widely

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"susceptibility_segregation_{index_type.lower()}_vs_b.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    plt.show()


def plot_time_series(aggregated_ts_data, output_dir="figures"):
    """
    Plots average time series for cooperation rate and average culture.
    Assumes aggregated_ts_data is a dictionary where keys are param_set_id
    and values are DataFrames with 'step', 'coop_rate_mean', 'coop_rate_sem',
    'avg_C_mean', 'avg_C_sem' columns, plus parameter columns.
    """
    print("\n--- Plotting Time Series ---")
    if not aggregated_ts_data:
        print("No time series data to plot.")
        return
    os.makedirs(output_dir, exist_ok=True)
    # Iterate through each parameter set in the aggregated data
    for param_set_id, ts_df in aggregated_ts_data.items():
        if ts_df.empty:
            print(f"Time series data is empty for {param_set_id}. Skipping.")
            continue
        # Extract parameters for title/filename
        # Assuming parameters are stored as columns in the ts_df
        params_str = ", ".join([f"{col}={ts_df[col].iloc[0]:.4g}" if isinstance(ts_df[col].iloc[0], float) else f"{col}={ts_df[col].iloc[0]}" for col in ts_df.columns if col not in ['step', 'coop_rate_mean', 'coop_rate_sem', 'avg_C_mean', 'avg_C_sem', 'param_set_id']])
        title_suffix = f" ({params_str})" if params_str else ""
        plt.figure(figsize=(12, 8))
        # Plot Cooperation Rate Time Series
        if 'coop_rate_mean' in ts_df.columns:
            plt.plot(ts_df['step'], ts_df['coop_rate_mean'], label='Cooperation Rate (Mean)', color='blue')
            if 'coop_rate_sem' in ts_df.columns:
                # Plot shaded error band (mean +/- SEM)
                plt.fill_between(ts_df['step'],
                                 ts_df['coop_rate_mean'] - ts_df['coop_rate_sem'],
                                 ts_df['coop_rate_mean'] + ts_df['coop_rate_sem'],
                                 color='blue', alpha=0.2, label='Cooperation Rate (SEM)')
        # Plot Average Culture Time Series
        if 'avg_C_mean' in ts_df.columns:
            plt.plot(ts_df['step'], ts_df['avg_C_mean'], label='Average Culture (Mean)', color='red')
            if 'avg_C_sem' in ts_df.columns:
                 # Plot shaded error band (mean +/- SEM)
                 plt.fill_between(ts_df['step'],
                                  ts_df['avg_C_mean'] - ts_df['avg_C_sem'],
                                  ts_df['avg_C_mean'] + ts_df['avg_C_sem'],
                                  color='red', alpha=0.2, label='Average Culture (SEM)')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'Time Series of Cooperation Rate and Average Culture{title_suffix}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(-0.05, 1.05) # Cooperation rate and C are between 0 and 1
        # Save figure with a filename based on param_set_id
        fig_filename = f"timeseries_{param_set_id}.png"
        fig_path = os.path.join(output_dir, fig_filename)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved time series figure to {fig_path}")
        plt.close() # Close the figure to free memory
# Modify the main execution block in plot_figures.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results from processed FLAMEGPU2 Cultural Game Simulation Logs.")
    parser.add_argument("--data_file_steady_state", type=str, required=True,
                        help="Path to the processed steady state data pickle file generated by process_logs.py")
    parser.add_argument("--data_file_timeseries", type=str,
                        help="Path to the aggregated time series data pickle file generated by process_logs.py (Optional)")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory to save the generated figures. Default: figures")
    parser.add_argument("--plot", nargs='+', default=['all'],
                        choices=['all', 'coop_rate_vs_b', 'segregation_index_type_vs_b', 'segregation_index_strategy_vs_b', 'susceptibility_coop_vs_b', 'susceptibility_seg_type_vs_b', 'susceptibility_seg_strat_vs_b', 'phase_diagram_coop', 'phase_diagram_chi_coop', 'phase_diagram_seg_type', 'phase_diagram_seg_strat', 'other_metrics_vs_b', 'time_series'],
                        help="Specify which plots to generate. 'all' plots all available figures. Choose from: coop_rate_vs_b, susceptibility_vs_b, phase_diagram_coop, phase_diagram_chi_coop, phase_diagram_seg_strat, phase_diagram_seg_type, other_metrics_vs_b, time_series")
    args = parser.parse_args()
    # Load the processed steady state data
    try:
        with open(args.data_file_steady_state, 'rb') as f:
            processed_df = pickle.load(f)
        print(f"Successfully loaded steady state data from {args.data_file_steady_state}")
        print(f"DataFrame shape: {processed_df.shape}")
    except FileNotFoundError:
        print(f"Error: Steady state data file not found at {args.data_file_steady_state}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading steady state data from {args.data_file_steady_state}: {e}")
        sys.exit(1)
    # Load the aggregated time series data (Optional)
    aggregated_ts_data = {}
    if args.data_file_timeseries:
        try:
            with open(args.data_file_timeseries, 'rb') as f:
                aggregated_ts_data = pickle.load(f)
            print(f"Successfully loaded aggregated time series data from {args.data_file_timeseries}")
            print(f"Number of parameter sets with time series data: {len(aggregated_ts_data)}")
        except FileNotFoundError:
            print(f"Warning: Time series data file not found at {args.data_file_timeseries}. Skipping time series plots.")
            aggregated_ts_data = {} # Ensure it's an empty dict
        except Exception as e:
            print(f"Error loading time series data from {args.data_file_timeseries}: {e}. Skipping time series plots.")
            aggregated_ts_data = {} # Ensure it's an empty dict
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Determine which plots to generate
    plots_to_generate = args.plot
    if 'all' in plots_to_generate:
        plots_to_generate = [
            'coop_rate_vs_b',
            'segregation_index_type_vs_b',
            'segregation_index_strategy_vs_b',
            'susceptibility_coop_vs_b',
            'susceptibility_seg_type_vs_b',
            'susceptibility_seg_strat_vs_b',
            'phase_diagram_coop',
            'phase_diagram_chi_coop',
            'phase_diagram_seg_type',
            'phase_diagram_seg_strat',
            'other_metrics_vs_b',
            'time_series' # Add time series to 'all'
        ]
    # Generate plots based on selection
    if 'time_series' in plots_to_generate:
        plot_time_series(aggregated_ts_data, args.output_dir)
    # ... (existing calls to other plotting functions) ...
    if 'coop_rate_vs_b' in plots_to_generate:
        plot_cooperation_rate_vs_b(processed_df, args.output_dir)
    if 'segregation_index_type_vs_b' in plots_to_generate:
        plot_segregation_index_vs_b(processed_df, args.output_dir, index_type="Type")
    if 'segregation_index_strategy_vs_b' in plots_to_generate:
        plot_segregation_index_vs_b(processed_df, args.output_dir, index_type="Strategy")
    if 'susceptibility_coop_vs_b' in plots_to_generate:
        plot_susceptibility_vs_b(processed_df, args.output_dir)
    if 'susceptibility_seg_type_vs_b' in plots_to_generate:
        plot_susceptibility_segregation_vs_b(processed_df, args.output_dir, index_type="Type")
    if 'susceptibility_seg_strat_vs_b' in plots_to_generate:
        plot_susceptibility_segregation_vs_b(processed_df, args.output_dir, index_type="Strategy")
    if 'phase_diagram_coop' in plots_to_generate:
         pd_df = processed_df # Filter if needed
         plot_phase_diagram(pd_df, 'b', 'K_C', 'avg_CooperationRate', args.output_dir)
    if 'phase_diagram_chi_coop' in plots_to_generate:
         pd_df = processed_df # Filter if needed
         plot_phase_diagram(pd_df, 'b', 'K_C', 'chi_CooperationRate', args.output_dir)
    if 'phase_diagram_seg_type' in plots_to_generate:
         pd_df = processed_df # Filter if needed
         plot_phase_diagram(pd_df, 'b', 'K_C', 'avg_SegregationIndexType', args.output_dir)
    if 'phase_diagram_seg_strat' in plots_to_generate:
         pd_df = processed_df # Filter if needed
         plot_phase_diagram(pd_df, 'b', 'K_C', 'avg_SegregationIndexStrategy', args.output_dir)
    if 'other_metrics_vs_b' in plots_to_generate:
        plot_other_metrics_vs_b(processed_df, args.output_dir)
    print("\nPlotting complete.")

