# plot_spatial_snapshots.py
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import argparse
import seaborn as sns

# --- Plotting Style Settings (From your provided file) ---
# Use a clean seaborn style (optional, but good for consistency)
sns.set_theme(style="whitegrid") # 可以选择 'whitegrid', 'darkgrid', 'white', 'dark'

plt.rcParams.update({
    "font.family": "sans-serif", # Use a common sans-serif font
    "font.size": 12, # Adjust font size as needed
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16, # For individual plot titles, but we'll use fig.suptitle for main title
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

# --- Custom Colors for the Spatial Snapshots ---
# These colors are specifically for the 8 categories in your spatial snapshot.
# Ensure these match the mapping logic in CulturalGameModel.get_grid_state_for_snapshot()
# The order here corresponds to state values 0 through 7.
colors_by_state = [
    '#a50026',  # 0: Strongly Self-Interested Defector (Dark Red)
    '#e0f3f8',  # 1: Strongly Self-Interested Cooperator (Light Blue)
    '#f46d43',  # 2: Weakly Self-Interested Defector (Orange Red)
    '#abd9e9',  # 3: Weakly Self-Interested Cooperator (Sky Blue)
    '#fdae61',  # 4: Weakly Other-Regarding Defector (Orange)
    '#74add1',  # 5: Weakly Other-Regarding Cooperator (Steel Blue)
    '#fee090',  # 6: Strongly Other-Regarding Defector (Gold)
    '#4575b4'   # 7: Strongly Other-Regarding Cooperator (Midnight Blue)
]
cmap = ListedColormap(colors_by_state)

# 原始标签列表，按照状态值 0-7 的顺序
original_labels_by_state = [
    "Strongly Self-Interested Defector",    # State 0
    "Strongly Self-Interested Cooperator",  # State 1
    "Weakly Self-Interested Defector",      # State 2
    "Weakly Self-Interested Cooperator",    # State 3
    "Weakly Other-Regarding Defector",      # State 4
    "Weakly Other-Regarding Cooperator",    # State 5
    "Strongly Other-Regarding Defector",    # State 6
    "Strongly Other-Regarding Cooperator"   # State 7
]

# 您希望的图例显示顺序对应的状态值
desired_legend_state_order = [
    0, # Strongly Self-Interested Defector
    2, # Weakly Self-Interested Defector
    4, # Weakly Other-Regarding Defector
    6, # Strongly Other-Regarding Defector
    1, # Strongly Self-Interested Cooperator
    3, # Weakly Self-Interested Cooperator
    5, # Weakly Other-Regarding Cooperator
    7  # Strongly Other-Regarding Cooperator
]

# 创建图例的 patches，按照 desired_legend_state_order 的顺序
patches = []
for state_value in desired_legend_state_order:
    # 获取对应状态值的颜色和标签
    color = colors_by_state[state_value]
    label = original_labels_by_state[state_value]
    # 创建 patch 并添加到列表中
    patches.append(mpatches.Patch(color=color, label=label))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot spatial snapshots from simulation results.")
    parser.add_argument("--L", type=int, default=50,
                        help="Grid size (L x L) used in the simulation.")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Total number of simulation steps when snapshot was taken.")
    parser.add_argument("--K", type=float, default=0.1,
                        help="Selection strength for strategy update used in the simulation.")
    parser.add_argument("--K_C", type=float, default=0.1,
                        help="Selection strength for cultural update used in the simulation.")
    parser.add_argument("--b_values", type=float, nargs='+', required=True,
                        help="List of b values for which to plot snapshots.")

    args = parser.parse_args()

    # Calculate figure width based on number of subplots and desired aspect ratio
    # Each subplot is roughly square. Add extra space for the legend.
    # A good starting point for subplot width is 5-6 inches.
    subplot_width = 5.5
    legend_width_ratio = 0.3 # Estimate legend takes about 30% of a subplot width
    fig_width = len(args.b_values) * subplot_width + subplot_width * legend_width_ratio
    fig_height = subplot_width # Assuming square subplots

    fig, axes = plt.subplots(1, len(args.b_values), figsize=(fig_width, fig_height))

    # Adjust suptitle position. y=1.02 is often a good starting point.
    fig.suptitle("Spatial Snapshots", fontsize=22, x=0.4,y=0.82) # Increased font size for main title

    # Ensure axes is always an array, even for a single subplot
    if len(args.b_values) == 1:
        axes = [axes]

    for i, b_val in enumerate(args.b_values):
        # Reconstruct the directory path where data was saved
        output_subdir_name = f"L{args.L}_b{b_val:.2f}_K{args.K}_KC{args.K_C}_steps{args.num_steps}"
        spatial_snapshot_dir = os.path.join("simulation_results", output_subdir_name, "spatial_snapshots_data")
        filename = os.path.join(spatial_snapshot_dir, f"snapshot_b_{b_val:.2f}_step_{args.num_steps}.npy")

        ax = axes[i] # Get the current subplot axis

        if os.path.exists(filename):
            grid_data = np.load(filename)

            ax.imshow(grid_data, cmap=cmap, origin='lower',
                      extent=[-0.5, grid_data.shape[1] - 0.5, -0.5, grid_data.shape[0] - 0.5])

            ax.set_title(f"b = {b_val:.1f}", fontsize=18) # Subplot title font size
            ax.set_xticks([]) # Remove x-axis ticks
            ax.set_yticks([]) # Remove y-axis ticks
            ax.set_aspect('equal', adjustable='box') # Ensure cells are square

            # Add grid lines
            # These lines are drawn at the edges of the cells, so from -0.5 to L-0.5
            ax.set_xticks(np.arange(-.5, grid_data.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-.5, grid_data.shape[0], 1), minor=True)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
            ax.tick_params(which='minor', size=0) # Hide minor tick marks themselves

        else:
            print(f"Error: Data file not found for b = {b_val}: {filename}")
            ax.set_title(f"b = {b_val:.2f} (Data Missing)", fontsize=18, color='red')
            ax.text(0.5, 0.5, "Data Not Found", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=16)


    # Add the legend to the right of the last subplot
    # bbox_to_anchor=(1.05, 0.5) places the legend's center-left at 1.05 (relative to axes width) and 0.5 (relative to axes height)
    # loc='center left' aligns the legend's center-left point with bbox_to_anchor
    axes[-1].legend(handles=patches, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=12, frameon=False)


    # Adjust layout to prevent overlap and make space for the legend
    # rect=[left, bottom, right, top] in figure coordinates (0 to 1)
    # We need to shrink the plotting area from the right to make space for the legend.
    # The 'right' value should be less than 1.0, depending on the legend's width.
    # A value like 0.85 or 0.8 might work, experiment with it.
    plt.tight_layout(rect=[0, 0, 0.80, 0.95]) # Adjusted rect for title and legend

    # Save the figure
    plt.savefig("SI_combined_snapshot_subprocess.png", dpi=300, bbox_inches='tight')
    plt.savefig("SI_combined_snapshot_subprocess.pdf", dpi=300, bbox_inches='tight')
    plt.show()
