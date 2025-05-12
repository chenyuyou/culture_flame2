# plot_specific_figures_standalone.py

import pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# --- Plotting Style Settings (Non-LaTeX) ---
# Use a clean seaborn style
#sns.set_theme(style="whitegrid")

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
Color4 = ["#003f5c", "#7a5195", "#ef5675", "#ffa600"]
Color2 = ["#488f31", "#de425b"]
# ==============================================================================
# PLOTTING FUNCTIONS FOR SPECIFIC FIGURES (DRAWING INDIVIDUAL SUBPLOTS)
# ==============================================================================

def plot_avg_cooperation_rate_vs_b(df, output_dir="figures"):
    """
    Plots steady-state average cooperation rate vs. b for different L.
    (Similar to Figure 1a in the provided text)
    """
    print("\n--- Plotting Average Cooperation Rate vs. b ---")
    if df.empty:
        print("No data to plot.")
        return

    required_cols = ['L', 'b', 'avg_CooperationRate', 'sem_CooperationRate']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns for this plot. Need: {required_cols}")
        return

    plot_data = df.groupby(['L', 'b']).agg(
        mean_coop=('avg_CooperationRate', 'mean'),
        sem_coop=('avg_CooperationRate', 'sem')
    ).reset_index()

    if plot_data.empty:
        print("No aggregated data to plot.")
        return

    plt.figure(figsize=(6, 5)) # Adjust figure size for a standalone plot

    L_values = sorted(plot_data['L'].unique())
    num_L = len(L_values)
    # Define marker sizes: larger for smaller L
    marker_sizes = np.linspace(plt.rcParams['lines.markersize'] * 1.5, plt.rcParams['lines.markersize'] * 0.8, num_L)
    # Reverse the order for plotting: plot smaller L first with larger markers
    L_values_plot_order = sorted(L_values, reverse=True)

    for i, L_val in enumerate(L_values_plot_order):
        subset = plot_data[plot_data['L'] == L_val].sort_values('b')
        yerr_data = subset['sem_coop'].replace([np.inf, -np.inf], np.nan).fillna(0)
        # Use different marker and size for each L
        plt.errorbar(subset['b'], subset['mean_coop'], yerr=yerr_data if np.any(yerr_data) else None,
                     fmt='-', # Use line
                     marker=MARKERS[i % len(MARKERS)], # Use marker
                     markersize=marker_sizes[i], # Use calculated marker size
                     capsize=3,
                     color = Color4[i],
                     label=f'L={L_val}') # Use plain text label

    plt.xlabel('Temptation (b)') # Use plain text label
    plt.ylabel('Average Cooperation Rate $\\langle f_C \\rangle$') # Use plain text label
#    plt.title('Average Cooperation Rate vs. b') # Title for this specific plot
    plt.legend(title='System Size (L)', loc='best') # Use plain text title, adjust location
    # # # # plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    plt.xlim(plot_data['b'].min(), plot_data['b'].max()) # Set x-limits based on data range

    # Adjust layout
    plt.tight_layout()

    # Save the figure in both PNG and PDF formats
    os.makedirs(output_dir, exist_ok=True)
    fig_path_png = os.path.join(output_dir, "fig1a_avg_cooperation_rate_vs_b.png")
    fig_path_pdf = os.path.join(output_dir, "fig1a_avg_cooperation_rate_vs_b.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path_png} and {fig_path_pdf}")
    plt.close() # Close the figure


def plot_avg_segregation_index_vs_b(df, output_dir="figures", index_type="Type"):
    """
    Plots steady-state average segregation index vs. b for different L.
    (Similar to Figure 1b in the provided text)
    """
    print(f"\n--- Plotting Average Segregation Index ({index_type}) vs. b ---")
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

    plt.figure(figsize=(6, 5)) # Adjust figure size

    L_values = sorted(plot_data['L'].unique())
    num_L = len(L_values)
    marker_sizes = np.linspace(plt.rcParams['lines.markersize'] * 1.5, plt.rcParams['lines.markersize'] * 0.8, num_L)
    L_values_plot_order = sorted(L_values, reverse=True)

    for i, L_val in enumerate(L_values_plot_order):
        subset = plot_data[plot_data['L'] == L_val].sort_values('b')
        yerr_data = subset['sem_seg'].replace([np.inf, -np.inf], np.nan).fillna(0)
        plt.errorbar(subset['b'], subset['mean_seg'], yerr=yerr_data if np.any(yerr_data) else None,
                     fmt='-', marker=MARKERS[i % len(MARKERS)], markersize=marker_sizes[i], capsize=3, color = Color4[i],
                     label=f'L={L_val}')

    plt.xlabel('Temptation (b)')
    plt.ylabel('Average Segregation Index $\\langle S \\rangle$')
#    plt.title(f'Average Cultural Segregation Index ({index_type}) vs. b')
    plt.legend(title='System Size (L)', loc='best')
    # # # # plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.xlim(plot_data['b'].min(), plot_data['b'].max())

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path_png = os.path.join(output_dir, f"fig1b_avg_segregation_index_{index_type.lower()}_vs_b.png")
    fig_path_pdf = os.path.join(output_dir, f"fig1b_avg_segregation_index_{index_type.lower()}_vs_b.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path_png} and {fig_path_pdf}")
    plt.close()


def plot_susceptibility_cooperation_vs_b(df, output_dir="figures"):
    """
    Plots susceptibility of cooperation rate vs. b for different L.
    (Similar to Figure 2a in the provided text)
    """
    print("\n--- Plotting Susceptibility of Cooperation Rate vs. b ---")
    if df.empty:
        print("No data to plot.")
        return

    required_cols = ['L', 'b', 'chi_CooperationRate']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns for this plot. Need: {required_cols}")
        return

    plot_data = df.drop_duplicates(subset=['L', 'b']).sort_values(['L', 'b'])

    if plot_data.empty:
        print("No unique parameter sets found for plotting.")
        return

    plt.figure(figsize=(6, 5)) # Adjust figure size

    L_values = sorted(plot_data['L'].unique())
    num_L = len(L_values)
    marker_sizes = np.linspace(plt.rcParams['lines.markersize'] * 1.5, plt.rcParams['lines.markersize'] * 0.8, num_L)
    L_values_plot_order = sorted(L_values, reverse=True)

    for i, L_val in enumerate(L_values_plot_order):
        subset = plot_data[plot_data['L'] == L_val]
        plt.plot(subset['b'], subset['chi_CooperationRate'], '-',
                 marker=MARKERS[i % len(MARKERS)], markersize=marker_sizes[i], color = Color4[i],
                 label=f'L={L_val}')

    plt.xlabel('Temptation (b)')
    plt.ylabel('Susceptibility $\\chi_{f_C}$')
#    plt.title('Cooperation Rate Susceptibility vs. b')
    plt.legend(title='System Size (L)', loc='best')
    # # # # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.yscale('linear') # Can change to 'log' if needed
    plt.xlim(plot_data['b'].min(), plot_data['b'].max())

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path_png = os.path.join(output_dir, "fig2a_susceptibility_cooperation_vs_b.png")
    fig_path_pdf = os.path.join(output_dir, "fig2a_susceptibility_cooperation_vs_b.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path_png} and {fig_path_pdf}")
    plt.close()


def plot_susceptibility_segregation_vs_b(df, output_dir="figures", index_type="Type"):
    """
    Plots susceptibility of segregation index vs. b for different L.
    (Similar to Figure 2b in the provided text)
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

    plt.figure(figsize=(6, 5)) # Adjust figure size

    L_values = sorted(plot_data['L'].unique())
    num_L = len(L_values)
    marker_sizes = np.linspace(plt.rcParams['lines.markersize'] * 1.5, plt.rcParams['lines.markersize'] * 0.8, num_L)
    L_values_plot_order = sorted(L_values, reverse=True)

    for i, L_val in enumerate(L_values_plot_order):
        subset = plot_data[plot_data['L'] == L_val]
        plt.plot(subset['b'], subset[chi_col], '-',
                 marker=MARKERS[i % len(MARKERS)], markersize=marker_sizes[i], color = Color4[i],
                 label=f'L={L_val}')

    plt.xlabel('Temptation (b)')
#    plt.ylabel('Susceptibility $\chi_S$')
#    plt.ylabel('Susceptibility $\chi_S$')
#    plt.title(f'Segregation Index Susceptibility ({index_type}) vs. b')
    plt.legend(title='System Size (L)', loc='best')
    # # # # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.yscale('linear') # Can change to 'log' if needed
    plt.xlim(plot_data['b'].min(), plot_data['b'].max())


    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path_png = os.path.join(output_dir, f"fig2b_susceptibility_segregation_{index_type.lower()}_vs_b.png")
    fig_path_pdf = os.path.join(output_dir, f"fig2b_susceptibility_segregation_{index_type.lower()}_vs_b.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path_png} and {fig_path_pdf}")
    plt.close()


def plot_boundary_fraction_vs_b(df, output_dir="figures"):
    """
    Plots average boundary fraction vs. b for different L.
    (Similar to Figure 7a in the provided text)
    """
    print("\n--- Plotting Average Boundary Fraction vs. b ---")
    if df.empty:
        print("No data to plot.")
        return

    required_cols = ['L', 'b', 'avg_BoundaryFraction', 'sem_BoundaryFraction']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns for this plot. Need: {required_cols}")
        return

    plot_data = df.groupby(['L', 'b']).agg(
        mean_bound_frac=('avg_BoundaryFraction', 'mean'),
        sem_bound_frac=('avg_BoundaryFraction', 'sem')
    ).reset_index()

    if plot_data.empty:
        print("No aggregated data to plot.")
        return

    plt.figure(figsize=(6, 5)) # Adjust figure size

    L_values = sorted(plot_data['L'].unique())
    num_L = len(L_values)
    marker_sizes = np.linspace(plt.rcParams['lines.markersize'] * 1.5, plt.rcParams['lines.markersize'] * 0.8, num_L)
    L_values_plot_order = sorted(L_values, reverse=True)

    for i, L_val in enumerate(L_values_plot_order):
        subset = plot_data[plot_data['L'] == L_val].sort_values('b')
        yerr_data = subset['sem_bound_frac'].replace([np.inf, -np.inf], np.nan).fillna(0)
        plt.errorbar(subset['b'], subset['mean_bound_frac'], yerr=yerr_data if np.any(yerr_data) else None,
                     fmt='-', marker=MARKERS[i % len(MARKERS)], markersize=marker_sizes[i], capsize=3, color = Color4[i],
                     label=f'L={L_val}')

    plt.xlabel('Temptation (b)')
    plt.ylabel('Boundary Fraction $\\langle f_{bound} \\rangle$')
#    plt.title('Average Boundary Fraction vs. b')
    plt.legend(title='System Size (L)', loc='best')
    # # # # plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    plt.xlim(plot_data['b'].min(), plot_data['b'].max())

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path_png = os.path.join(output_dir, "fig7a_avg_boundary_fraction_vs_b.png")
    fig_path_pdf = os.path.join(output_dir, "fig7a_avg_boundary_fraction_vs_b.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path_png} and {fig_path_pdf}")
    plt.close()


def plot_boundary_bulk_cooperation_rate_vs_b(df, output_dir="figures", fixed_L=50):
    """
    Plots boundary vs. bulk cooperation rates vs. b for a fixed L.
    (Similar to Figure 7b in the provided text)
    """
    print(f"\n--- Plotting Boundary vs. Bulk Cooperation Rate vs. b (L={fixed_L}) ---")
    if df.empty:
        print("No data to plot.")
        return

    required_cols = ['L', 'b', 'avg_BoundaryCoopRate', 'sem_BoundaryCoopRate',
                       'avg_BulkCoopRate', 'sem_BulkCoopRate']

    if not all(col in df.columns for col in required_cols):
         print(f"Missing required columns for this plot. Need: {required_cols}")
         return

    # Filter for the specific L
    plot_data = df[df['L'] == fixed_L].groupby('b').agg(
        mean_bound_coop=('avg_BoundaryCoopRate', 'mean'),
        sem_bound_coop=('avg_BoundaryCoopRate', 'sem'),
        mean_bulk_coop=('avg_BulkCoopRate', 'mean'),
        sem_bulk_coop=('sem_BulkCoopRate', 'sem') # Corrected column name
    ).reset_index().sort_values('b')


    if plot_data.empty:
        print(f"No aggregated data found for L = {fixed_L}.")
        return

    plt.figure(figsize=(6, 5)) # Adjust figure size

    # Boundary Cooperation Rate (Red Circles)
    yerr_bound = plot_data['sem_bound_coop'].replace([np.inf, -np.inf], np.nan).fillna(0)
    plt.errorbar(plot_data['b'], plot_data['mean_bound_coop'], yerr=yerr_bound if np.any(yerr_bound) else None,
                 fmt='o-', capsize=3, color='#488f31', label='Boundary') # Use line and marker

    # Bulk Cooperation Rate (Blue Squares)
    yerr_bulk = plot_data['sem_bulk_coop'].replace([np.inf, -np.inf], np.nan).fillna(0)
    plt.errorbar(plot_data['b'], plot_data['mean_bulk_coop'], yerr=yerr_bulk if np.any(yerr_bulk) else None,
                 fmt='s-', capsize=3, color='#de425b', label='Bulk') # Use line and marker

    plt.xlabel('Temptation (b)')
    plt.ylabel('Average Cooperation Rate $\\langle f_C \\rangle$')
#    plt.title(f'Average Cooperation Rate (L={fixed_L})')
    plt.legend(loc='best')
    # # # # plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    plt.xlim(plot_data['b'].min(), plot_data['b'].max())


    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path_png = os.path.join(output_dir, f"fig7b_boundary_bulk_coop_rate_vs_b_L{fixed_L}.png")
    fig_path_pdf = os.path.join(output_dir, f"fig7b_boundary_bulk_coop_rate_vs_b_L{fixed_L}.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path_png} and {fig_path_pdf}")
    plt.close()


# Add these functions to your plot_specific_figures_standalone.py file

def plot_phase_diagram1(df, param1_name, param2_name, metric_name, metric_label, title, output_dir="figures"):
    """
    Plots a phase diagram (heatmap/contour) of a metric on a 2D parameter plane.
    """
    print(f"\n--- Plotting Phase Diagram: {metric_name} on {param1_name} vs {param2_name} ---")
    if df.empty:
        print("No data to plot.")
        return

    avg_metric_col = f'avg_{metric_name}'
    required_cols = [param1_name, param2_name, avg_metric_col]

    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns for this plot. Need: {required_cols}")
        return

    # Ensure only one data point per parameter combination for the heatmap
    # If you have multiple runs per combo, the process_logs.py should have averaged them.
    # We drop duplicates based on the parameter names.
    plot_data = df.drop_duplicates(subset=[param1_name, param2_name]).sort_values([param1_name, param2_name])

    if plot_data.empty:
        print("No unique parameter sets found for plotting phase diagram.")
        return

    # Pivot the data to create a grid for the heatmap
    # Use the average metric value for the heatmap color
    heatmap_data = plot_data.pivot(index=param2_name, columns=param1_name, values=avg_metric_col)

    if heatmap_data.empty:
        print("Could not pivot data for heatmap.")
        return

    plt.figure(figsize=(8, 6)) # Adjust figure size for a phase diagram

    # Use seaborn's heatmap for a clear visualization
    sns.heatmap(heatmap_data, cmap="viridis", annot=False, cbar_kws={'label': metric_label}) # 'viridis' is a good default colormap

    plt.xlabel(param1_name)
    plt.ylabel("$"+param2_name+"$")
    plt.title(title)

    plt.tight_layout()

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path_png = os.path.join(output_dir, f"phase_diagram_{metric_name.lower()}_vs_{param1_name}_{param2_name}.png")
    fig_path_pdf = os.path.join(output_dir, f"phase_diagram_{metric_name.lower()}_vs_{param1_name}_{param2_name}.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved phase diagram to {fig_path_png} and {fig_path_pdf}")
    plt.close()


def plot_phase_diagram(df, param1_name, param2_name, metric_name, metric_label, title, output_dir="figures"):
    """
    Plots a phase diagram (heatmap/contour) of a metric on a 2D parameter plane.
    """
    print(f"\n--- Plotting Phase Diagram: {metric_name} on {param1_name} vs {param2_name} ---")
    if df.empty:
        print("No data to plot.")
        return

    avg_metric_col = f'avg_{metric_name}'
    required_cols = [param1_name, param2_name, avg_metric_col]

    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns for this plot. Need: {required_cols}")
        return

    # Ensure only one data point per parameter combination for the plot
    plot_data = df.drop_duplicates(subset=[param1_name, param2_name]).sort_values([param1_name, param2_name])

    if plot_data.empty:
        print("No unique parameter sets found for plotting phase diagram.")
        return

    # Pivot the data to create a grid for the contour plot
    # Use the average metric value for the contour levels
    pivot_data = plot_data.pivot(index=param2_name, columns=param1_name, values=avg_metric_col)

    if pivot_data.empty:
        print("Could not pivot data for contour plot.")
        return

    # Get the unique values for the parameters to define the grid
    param1_values = pivot_data.columns.values
    param2_values = pivot_data.index.values
    metric_values = pivot_data.values

    # Create a meshgrid for the contour plot
    X, Y = np.meshgrid(param1_values, param2_values)

    plt.figure(figsize=(8, 6)) # Adjust figure size for a phase diagram

    # --- Use contourf for filled contour plot ---
    # You might want to specify levels for the contour lines
    # Example: levels = np.linspace(metric_values.min(), metric_values.max(), 20) # 20 levels
    # Or specify specific levels: levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    # If levels is not specified, matplotlib will choose automatically.
    contourf_plot = plt.contourf(X, Y, metric_values, cmap="viridis") # 'viridis' is a good default colormap

    # Optionally, add contour lines on top of the filled contours for better readability
    # plt.contour(X, Y, metric_values, colors='black', linewidths=0.5)

    # Add a color bar to show the mapping of colors to metric values
    cbar = plt.colorbar(contourf_plot)
    cbar.set_label(metric_label)

    plt.xlabel(param1_name)
    plt.ylabel("$"+param2_name+"$")
    plt.title(title)

    plt.tight_layout()

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    # Change filename to reflect contour plot
    fig_path_png = os.path.join(output_dir, f"phase_diagram_contour_{metric_name.lower()}_vs_{param1_name}_{param2_name}.png")
    fig_path_pdf = os.path.join(output_dir, f"phase_diagram_contour_{metric_name.lower()}_vs_{param1_name}_{param2_name}.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved phase diagram to {fig_path_png} and {fig_path_pdf}")
    plt.close()



def plot_phase_diagram_segregation_index_type(df, param1_name, param2_name, output_dir="figures"):
    """
    Plots the phase diagram of average cultural segregation index.
    """
    plot_phase_diagram(df, param1_name, param2_name,
                       metric_name="SegregationIndexType",
                       metric_label="Average Cultural Segregation Index $\\langle S \\rangle$",
                       title=f"Average Cultural Segregation Index on ${param1_name}$ vs ${param2_name}$",
                       output_dir=output_dir)


def plot_phase_diagram_boundary_fraction(df, param1_name, param2_name, output_dir="figures"):
    """
    Plots the phase diagram of average boundary fraction.
    """
    plot_phase_diagram(df, param1_name, param2_name,
                       metric_name="BoundaryFraction",
                       metric_label="Average Boundary Fraction $\\langle f_{bound} \\rangle$",
                       title=f"Average Boundary Fraction on ${param1_name}$ vs ${param2_name}$",
                       output_dir=output_dir)



# ==============================================================================
# MAIN EXECUTION BLOCK FOR PLOTTING
# ==============================================================================

# Modify the __main__ block in your plot_specific_figures_standalone.py file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot specific figures (Standalone Style) from processed FLAMEGPU2 Cultural Game Simulation Logs.")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the processed data pickle file generated by process_logs.py")
    parser.add_argument("--output_dir", type=str, default="figures_standalone",
                        help="Directory to save the generated figures. Default: figures_standalone")
    parser.add_argument("--plot", nargs='+', default=['all'],
                        choices=['all', 'fig1a', 'fig1b_type', 'fig1b_strategy', 'fig2a', 'fig2b_type', 'fig2b_strategy', 'fig7a', 'fig7b', 'phasediagram_seg_type', 'phasediagram_boundary_frac'],
                        help="Specify which plots to generate. 'all' plots all available figures. Choose from: fig1a, fig1b_type, fig2a, fig2b_type, fig2b_strategy, fig7a, fig7b, phasediagram_seg_type, phasediagram_boundary_frac")
    parser.add_argument("--L_boundary_bulk", type=int, default=50,
                        help="System size L to use for plotting boundary vs bulk cooperation rate (Figure 7b). Default: 50")
    # Add arguments for phase diagram parameters
    parser.add_argument("--pd_param1", type=str, default="b",
                        help="Name of the first parameter for phase diagrams (x-axis). Default: b")
    parser.add_argument("--pd_param2", type=str, default="K_C",
                        help="Name of the second parameter for phase diagrams (y-axis). Default: K_C")


    args = parser.parse_args()

    # Load the processed data
    try:
        with open(args.data_file, 'rb') as f:
            # Assuming the pickle file contains only the combined_df for now
            # If it contains both df and time series, you'll need to unpack
            processed_data = pickle.load(f)
            if isinstance(processed_data, tuple):
                 processed_df = processed_data[0] # Assuming the first element is the DataFrame
                 # aggregated_time_series_data = processed_data[1] # If you need time series later
            else:
                 processed_df = processed_data

        print(f"Successfully loaded data from {args.data_file}")
        print(f"DataFrame shape: {processed_df.shape}")
        # Print available columns to help user
        print(f"Available columns for plotting: {processed_df.columns.tolist()}")

    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data from {args.data_file}: {e}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which plots to generate
    plots_to_generate = args.plot
    if 'all' in plots_to_generate:
        # Include phase diagrams in 'all' if desired, or keep them separate
        plots_to_generate = ['fig1a', 'fig1b_type', 'fig2a', 'fig2b_type', 'fig7a', 'fig7b', 'phasediagram_seg_type', 'phasediagram_boundary_frac']


    # Generate plots based on selection
    if 'fig1a' in plots_to_generate:
        plot_avg_cooperation_rate_vs_b(processed_df, args.output_dir)

    if 'fig1b_type' in plots_to_generate:
        plot_avg_segregation_index_vs_b(processed_df, args.output_dir, index_type="Type")

    if 'fig1b_strategy' in plots_to_generate:
        plot_avg_segregation_index_vs_b(processed_df, args.output_dir, index_type="Strategy")

    if 'fig2a' in plots_to_generate:
        plot_susceptibility_cooperation_vs_b(processed_df, args.output_dir)

    if 'fig2b_type' in plots_to_generate:
        plot_susceptibility_segregation_vs_b(processed_df, args.output_dir, index_type="Type")

    if 'fig2b_strategy' in plots_to_generate:
        plot_susceptibility_segregation_vs_b(processed_df, args.output_dir, index_type="Strategy")

    if 'fig7a' in plots_to_generate:
        plot_boundary_fraction_vs_b(processed_df, args.output_dir)

    if 'fig7b' in plots_to_generate:
        plot_boundary_bulk_cooperation_rate_vs_b(processed_df, args.output_dir, fixed_L=args.L_boundary_bulk)

    # --- Add calls for Phase Diagrams ---
    if 'phasediagram_seg_type' in plots_to_generate:
        plot_phase_diagram_segregation_index_type(processed_df, args.pd_param1, args.pd_param2, args.output_dir)

    if 'phasediagram_boundary_frac' in plots_to_generate:
        plot_phase_diagram_boundary_fraction(processed_df, args.pd_param1, args.pd_param2, args.output_dir)


    print("\nPlotting complete.")

