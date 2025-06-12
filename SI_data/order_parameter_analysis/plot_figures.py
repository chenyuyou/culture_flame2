# plot_figures.py

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn
import pickle
import argparse
import os

# --- Plotting Style Settings ---
# Use a clean seaborn style (optional, can be used with or without manual rcParams)
# sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.family": "sans-serif", # Use a common sans-serif font
    "font.size": 12, # Adjust font size as needed
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "lines.linewidth": 1.5, # Default line width
    "lines.markersize": 6,  # Default marker size
    "lines.markeredgewidth": 0.5, # Marker edge width
    "axes.linewidth": 1.0, # Axis line width
    "xtick.major.width": 1.0, # Tick width
    "ytick.major.width": 1.0,
    "xtick.direction": "in", # Ticks point inwards
    "ytick.direction": "in",
    "xtick.major.pad": 5, # Padding between tick and label
    "ytick.major.pad": 5,
    "figure.dpi": 300 # High resolution for saving
})

# Define a list of markers to cycle through for different L values
MARKERS = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*'] # Circle, Square, Triangle up, Diamond, etc.
# Define color palettes for the four cultural types
Color4_Types = ["#003f5c", "#7a5195", "#ef5675", "#ffa600"] # Colors for Type 1, 2, 3, 4

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_processed_data(filepath):
    """Loads the processed data from a pickle file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded data from '{filepath}'")
        print(f"Data shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"Error loading data from '{filepath}': {e}")
        return None

def save_figure(fig, filename, output_dir="figures"):
    """Saves the given figure to the specified directory in PNG and PDF formats."""
    os.makedirs(output_dir, exist_ok=True)
    filepath_png = os.path.join(output_dir, f"{filename}.png")
    filepath_pdf = os.path.join(output_dir, f"{filename}.pdf")
    try:
        fig.savefig(filepath_png, dpi=300, bbox_inches='tight')
        fig.savefig(filepath_pdf, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {filepath_png} and {filepath_pdf}")
    except Exception as e:
        print(f"Error saving figure {filename}: {e}")
    plt.close(fig) # Close the figure after saving

# ==============================================================================
# PLOTTING FUNCTIONS FOR SPECIFIC FIGURES (Modified for 4 types)
# ==============================================================================

# (Keep your existing plot_figure1, plot_figure2, plot_figure4 functions here if you need them,
# but they might still use the old A/B classification unless modified)
# For this response, I will only include the modified plot_supplementary_group_metrics
# and the main function that calls it.

def plot_supplementary_group_metrics(df, output_dir="figures", target_L=50):
    """
    Plots Supplementary Figure: Group-Specific Average Cooperation Rate and Segregation Index vs b for a specific L.
    Splits the two subplots into separate figures and saves them.
    Uses the new 4-type classification.
    """
    print(f"\n--- Plotting Supplementary Group Metrics for L={target_L} (4 Types) ---")
    df_L = df[df['L'] == target_L].sort_values('b')

    if df_L.empty:
        print(f"Warning: No data found for L={target_L}. Skipping Supplementary Group Metrics plot.")
        return

    required_cols = ['b',
                       'avg_CooperationRate_type_1', 'sem_CooperationRate_type_1',
                       'avg_CooperationRate_type_2', 'sem_CooperationRate_type_2',
                       'avg_CooperationRate_type_3', 'sem_CooperationRate_type_3',
                       'avg_CooperationRate_type_4', 'sem_CooperationRate_type_4',
                       'avg_SegregationIndex_type_1', 'sem_SegregationIndex_type_1',
                       'avg_SegregationIndex_type_2', 'sem_SegregationIndex_type_2',
                       'avg_SegregationIndex_type_3', 'sem_SegregationIndex_type_3',
                       'avg_SegregationIndex_type_4', 'sem_SegregationIndex_type_4']

    if not all(col in df_L.columns for col in required_cols):
         missing = [col for col in required_cols if col not in df_L.columns]
         print(f"Missing required columns for this plot. Need: {missing}")
         return

    # Aggregate data by 'b' for the specific L
    plot_data = df_L.groupby('b').agg(
        mean_coop_type_1=('avg_CooperationRate_type_1', 'mean'),
        sem_coop_type_1=('avg_CooperationRate_type_1', 'sem'),
        mean_coop_type_2=('avg_CooperationRate_type_2', 'mean'),
        sem_coop_type_2=('avg_CooperationRate_type_2', 'sem'),
        mean_coop_type_3=('avg_CooperationRate_type_3', 'mean'),
        sem_coop_type_3=('avg_CooperationRate_type_3', 'sem'),
        mean_coop_type_4=('avg_CooperationRate_type_4', 'mean'),
        sem_coop_type_4=('avg_CooperationRate_type_4', 'sem'),
        mean_seg_type_1=('avg_SegregationIndex_type_1', 'mean'),
        sem_seg_type_1=('avg_SegregationIndex_type_1', 'sem'),
        mean_seg_type_2=('avg_SegregationIndex_type_2', 'mean'),
        sem_seg_type_2=('avg_SegregationIndex_type_2', 'sem'),
        mean_seg_type_3=('avg_SegregationIndex_type_3', 'mean'),
        sem_seg_type_3=('avg_SegregationIndex_type_3', 'sem'),
        mean_seg_type_4=('avg_SegregationIndex_type_4', 'mean'),
        sem_seg_type_4=('avg_SegregationIndex_type_4', 'sem')
    ).reset_index().sort_values('b')

    if plot_data.empty:
        print(f"No aggregated data found for L = {target_L}.")
        return

    # --- Plot Group-Specific Average Cooperation Rate (Left Subplot) ---
    fig_coop, ax_coop = plt.subplots(figsize=(6, 5)) # Adjust figure size for a standalone plot

    # Plot Cooperation Rate for the four types
    yerr_type_1_coop = plot_data['sem_coop_type_1'].replace([np.inf, -np.inf], np.nan).fillna(0)
    ax_coop.errorbar(plot_data['b'], plot_data['mean_coop_type_1'], yerr=yerr_type_1_coop if np.any(yerr_type_1_coop) else None,
                 fmt='o-', capsize=3, color=Color4_Types[0], label='0 ≤ C < 0.25')

    yerr_type_2_coop = plot_data['sem_coop_type_2'].replace([np.inf, -np.inf], np.nan).fillna(0)
    ax_coop.errorbar(plot_data['b'], plot_data['mean_coop_type_2'], yerr=yerr_type_2_coop if np.any(yerr_type_2_coop) else None,
                 fmt='s-', capsize=3, color=Color4_Types[1], label='0.25 ≤ C < 0.5')

    yerr_type_3_coop = plot_data['sem_coop_type_3'].replace([np.inf, -np.inf], np.nan).fillna(0)
    ax_coop.errorbar(plot_data['b'], plot_data['mean_coop_type_3'], yerr=yerr_type_3_coop if np.any(yerr_type_3_coop) else None,
                 fmt='^-', capsize=3, color=Color4_Types[2], label='0.5 ≤ C < 0.75')

    yerr_type_4_coop = plot_data['sem_coop_type_4'].replace([np.inf, -np.inf], np.nan).fillna(0)
    ax_coop.errorbar(plot_data['b'], plot_data['mean_coop_type_4'], yerr=yerr_type_4_coop if np.any(yerr_type_4_coop) else None,
                 fmt='D-', capsize=3, color=Color4_Types[3], label='0.75 ≤ C ≤ 1')


    ax_coop.set_xlabel('Temptation (b)')
    ax_coop.set_ylabel('Average Cooperation Rate $\\langle f_C \\rangle$')
#    ax_coop.set_title(f'Group-Specific Average Cooperation Rate (L={target_L})') # Optional title
    ax_coop.legend(loc='best')
    # # # # ax_coop.grid(True, linestyle='--', alpha=0.6) # Optional grid
    ax_coop.set_ylim(0, 1.05)
    ax_coop.set_xlim(plot_data['b'].min(), plot_data['b'].max())

    fig_coop.tight_layout()
    save_figure(fig_coop, f"group_coop_rate_vs_b_L{target_L}_4types", output_dir)


    # --- Plot Group-Specific Average Segregation Index (Right Subplot) ---
    fig_seg, ax_seg = plt.subplots(figsize=(6, 5)) # Adjust figure size for a standalone plot

    # Plot Segregation Index for the four types
    yerr_type_1_seg = plot_data['sem_seg_type_1'].replace([np.inf, -np.inf], np.nan).fillna(0)
    ax_seg.errorbar(plot_data['b'], plot_data['mean_seg_type_1'], yerr=yerr_type_1_seg if np.any(yerr_type_1_seg) else None,
                 fmt='o-', capsize=3, color=Color4_Types[0], label='0 ≤ C < 0.25')

    yerr_type_2_seg = plot_data['sem_seg_type_2'].replace([np.inf, -np.inf], np.nan).fillna(0)
    ax_seg.errorbar(plot_data['b'], plot_data['mean_seg_type_2'], yerr=yerr_type_2_seg if np.any(yerr_type_2_seg) else None,
                 fmt='s-', capsize=3, color=Color4_Types[1], label='0.25 ≤ C < 0.5')

    yerr_type_3_seg = plot_data['sem_seg_type_3'].replace([np.inf, -np.inf], np.nan).fillna(0)
    ax_seg.errorbar(plot_data['b'], plot_data['mean_seg_type_3'], yerr=yerr_type_3_seg if np.any(yerr_type_3_seg) else None,
                 fmt='^-', capsize=3, color=Color4_Types[2], label='0.5 ≤ C < 0.75')

    yerr_type_4_seg = plot_data['sem_seg_type_4'].replace([np.inf, -np.inf], np.nan).fillna(0)
    ax_seg.errorbar(plot_data['b'], plot_data['mean_seg_type_4'], yerr=yerr_type_4_seg if np.any(yerr_type_4_seg) else None,
                 fmt='D-', capsize=3, color=Color4_Types[3], label='0.75 ≤ C ≤ 1')


    ax_seg.set_xlabel('Temptation (b)')
    ax_seg.set_ylabel('Average Segregation Index $\\langle S \\rangle$')
#    ax_seg.set_title(f'Group-Specific Average Segregation Index (L={target_L})') # Optional title
    ax_seg.legend(loc='best')
    # # # # ax_seg.grid(True, linestyle='--', alpha=0.6) # Optional grid
    ax_seg.set_ylim(0, 1.05) # Segregation Index (proportion) is between 0 and 1
    ax_seg.set_xlim(plot_data['b'].min(), plot_data['b'].max())

    fig_seg.tight_layout()
    save_figure(fig_seg, f"group_seg_index_vs_b_L{target_L}_4types", output_dir)


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot figures from processed FLAMEGPU2 Cultural Game data.")
    parser.add_argument("--data_file", type=str, default="processed_results.pkl",
                        help="Path to the processed pickle data file.")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory to save the generated figures.")
    parser.add_argument("--target_L_figure4", type=int, default=50,
                        help="Grid size L to use for Figure 4 (Boundary vs Bulk Coop Rate).")
    parser.add_argument("--target_L_group_metrics", type=int, default=50,
                        help="Grid size L to use for Supplementary Group Metrics plots.")
    args = parser.parse_args()

    data_file = args.data_file
    output_dir = args.output_dir
    target_L_figure4 = args.target_L_figure4
    target_L_group_metrics = args.target_L_group_metrics

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the processed data
    df = load_processed_data(data_file)

    if df is None or df.empty:
        print("No data available for plotting. Exiting.")
        return

    # Ensure necessary columns exist
    required_cols = [
        'L', 'b', 'avg_CooperationRate', 'sem_CooperationRate',
        'avg_SegregationIndex', 'sem_SegregationIndex',
        'chi_CooperationRate', 'chi_SegregationIndex',
        'avg_BoundaryCoopRate', 'sem_BoundaryCoopRate',
        'avg_BulkCoopRate', 'sem_BulkCoopRate',
        # Add new columns for the four types
        'avg_CooperationRate_type_1', 'sem_CooperationRate_type_1',
        'avg_CooperationRate_type_2', 'sem_CooperationRate_type_2',
        'avg_CooperationRate_type_3', 'sem_CooperationRate_type_3',
        'avg_CooperationRate_type_4', 'sem_CooperationRate_type_4',
        'avg_SegregationIndex_type_1', 'sem_SegregationIndex_type_1',
        'avg_SegregationIndex_type_2', 'sem_SegregationIndex_type_2',
        'avg_SegregationIndex_type_3', 'sem_SegregationIndex_type_3',
        'avg_SegregationIndex_type_4', 'sem_SegregationIndex_type_4',
        'chi_CooperationRate_type_1',
        'chi_CooperationRate_type_2',
        'chi_CooperationRate_type_3',
        'chi_CooperationRate_type_4',
        'chi_SegregationIndex_type_1',
        'chi_SegregationIndex_type_2',
        'chi_SegregationIndex_type_3',
        'chi_SegregationIndex_type_4',
    ]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing required columns in data file: {missing}")
        print("Please ensure process_logs.py generated the necessary columns.")
        return

    # Plot the supplementary group metrics figures as separate files using the 4-type classification
    plot_supplementary_group_metrics(df, output_dir, target_L=target_L_group_metrics)

    # You can uncomment the calls to other plotting functions if you need those figures as well
    # plot_figure1(df, output_dir) # Note: Figure 1 and 2 might still use old A/B classification
    # plot_figure2(df, output_dir)
    # plot_figure4(df, output_dir, target_L=target_L_figure4)


    print("\nPlotting complete.")

if __name__ == "__main__":
    main()
