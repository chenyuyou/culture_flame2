import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob
import argparse

def get_grid_states(json_file_path):
    """
    Reads a single FLAMEGPU snapshot JSON and returns the grid state array.

    Args:
        json_file_path (str): Path to the snapshot JSON file.

    Returns:
        tuple: (grid_states, L, b_value, step) or (None, None, None, None) if error occurs.
    """
    try:
        with open(json_file_path, 'r') as f:
            snapshot_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Snapshot file not found at {json_file_path}")
        return None, None, None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return None, None, None, None

    L = snapshot_data.get("environment", {}).get("GRID_DIM_L")
    if L is None:
        print("Error: Could not find GRID_DIM_L in the environment data.")
        return None, None, None, None

    agents_data = snapshot_data.get("agents", {}).get("CulturalAgent", {}).get("default")
    if not agents_data:
        print("Error: Could not find agent data in the snapshot.")
        return None, None, None, None

    grid_states = np.full((L, L), -1, dtype=int)

    for agent in agents_data:
        ix = agent.get("ix")
        iy = agent.get("iy")
        strategy = agent.get("strategy")
        C = agent.get("C")

        if ix is not None and iy is not None and strategy is not None and C is not None:
            if C < 0.5: # Self-Interested
                if strategy == 0: # Defector
                    grid_states[iy][ix] = 0 # Self-Interested Defector
                else: # Cooperator
                    grid_states[iy][ix] = 1 # Self-Interested Cooperator
            else: # Other-Regarding
                if strategy == 0: # Defector
                    grid_states[iy][ix] = 2 # Other-Regarding Defector
                else: # Cooperator
                    grid_states[iy][ix] = 3 # Other-Regarding Cooperator

    b_value = snapshot_data.get("environment", {}).get("b")
    step = snapshot_data.get("step")

    return grid_states, L, b_value, step

def plot_combined_snapshots(b_values, step, root_directory="snapshot_output", output_filename="combined_snapshot.pdf"):
    """
    Plots snapshots for specified b values at a given step in a single figure with subplots.

    Args:
        b_values (list): A list of 3 b values to plot.
        step (int): The simulation step to plot.
        root_directory (str): The base directory containing simulation run subdirectories.
        output_filename (str): The filename to save the combined plot.
    """
    if len(b_values) != 3:
        print("Error: Exactly 3 b values must be provided for combined plotting.")
        return

    # Adjust figure size for better readability and space for legend
    fig, axes = plt.subplots(1, 3, figsize=(14, 4)) # Increased figure width and height
    axes = axes.flatten()

    # Define agent states and corresponding colors (consistent across subplots)
    state_colors = {
        0: '#de425b',  # 深红   自利背叛
        3: '#488f31',  # 深绿   利他合作
        2: '#f3babc',  # 浅红   利他背叛
        1: '#bad0af'   # 浅绿   自利合作
    }
    cmap = mcolors.ListedColormap([state_colors[i] for i in sorted(state_colors.keys())])
    bounds = sorted(state_colors.keys()) + [max(state_colors.keys()) + 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Self-Interested Defector', markersize=10, markerfacecolor=state_colors[0]),
        plt.Line2D([0], [0], marker='s', color='w', label='Other-Regarding Defector', markersize=10, markerfacecolor=state_colors[2]),
        plt.Line2D([0], [0], marker='s', color='w', label='Self-Interested Cooperator', markersize=10, markerfacecolor=state_colors[1]),
        plt.Line2D([0], [0], marker='s', color='w', label='Other-Regarding Cooperator', markersize=10, markerfacecolor=state_colors[3])
    ]

    for i, b_val in enumerate(b_values):
        snapshot_subdir = os.path.join(root_directory, f"b_{b_val}")
        snapshot_file = os.path.join(snapshot_subdir, f"snapshot_step_{step}.json")

        grid_states, L, snapshot_b_value, snapshot_step = get_grid_states(snapshot_file)

        ax = axes[i] # Get the current subplot axis

        if grid_states is not None:
            im = ax.imshow(grid_states, cmap=cmap, norm=norm, origin='lower')

            # Optional: Add grid lines
            ax.set_xticks(np.arange(-.5, L, 1), minor=True)
            ax.set_yticks(np.arange(-.5, L, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)

            # Remove axis ticks and labels for a cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Set title for each subplot with increased font size
            ax.set_title(f"b = {b_val:.2f}", fontsize=18) # Increased title font size

        else:
            # If snapshot not found or error, display a message in the subplot
            ax.text(0.5, 0.5, f"Snapshot for b={b_val:.2f}\nStep {step} not found",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='red', fontsize=16) # Increased error message font size
            ax.set_title(f"b = {b_val:.2f} (Data Missing)", fontsize=18) # Increased title font size


    # Add a single legend to the right of the last subplot
    # Adjust bbox_to_anchor to move the legend further to the right
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=12) # Increased legend font size and adjusted position

    # Adjust layout to prevent overlap and make space for the legend
    # Increase the right margin to accommodate the legend
    plt.tight_layout(rect=[0, 0, 0.8, 0.95]) # Adjusted rect to leave more space on the right

    # Add a main title for the entire figure with increased font size
    fig.suptitle(f"Spatial Snapshots", fontsize=22, y=0.99) # Increased main title font size and adjusted position
#    fig.suptitle(f"Spatial Snapshots at Step {step}", fontsize=22, y=0.99) # Increased main title font size and adjusted position



    # Save the combined plot
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_filename, dpi=300)
    print(f"Combined snapshot plot saved to {output_filename}")

    plt.close(fig)


# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot combined snapshots for specified b values at a given step.")
    parser.add_argument("--b_values", type=float, nargs=3, required=True,
                        help="Three b values to plot (e.g., 1.5 1.6 1.7)")
    parser.add_argument("--step", type=int, required=True,
                        help="The simulation step to plot.")
    parser.add_argument("--root_dir", type=str, default="snapshot_output",
                        help="The base directory containing simulation run subdirectories. Default: snapshot_output")
    parser.add_argument("--output_file", type=str, default="combined_snapshot.pdf",
                        help="The filename to save the combined plot. Default: combined_snapshot.png")

    args = parser.parse_args()

    plot_combined_snapshots(args.b_values, args.step, args.root_dir, args.output_file)
