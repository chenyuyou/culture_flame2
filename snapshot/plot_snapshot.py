import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob # Import the glob module for pattern matching

def plot_single_snapshot(json_file_path, output_filename):
    """
    Reads a single FLAMEGPU snapshot JSON, plots the spatial distribution of agents
    based on culture type and strategy, and saves the plot.

    Args:
        json_file_path (str): Path to the snapshot JSON file.
        output_filename (str): Filename to save the output plot.
    """
    try:
        with open(json_file_path, 'r') as f:
            snapshot_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Snapshot file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return

    # Extract grid dimension L
    L = snapshot_data.get("environment", {}).get("GRID_DIM_L")
    if L is None:
        print("Error: Could not find GRID_DIM_L in the environment data.")
        return

    # Extract agent data
    agents_data = snapshot_data.get("agents", {}).get("CulturalAgent", {}).get("default")
    if not agents_data:
        print("Error: Could not find agent data in the snapshot.")
        return

    # Define agent states and corresponding colors
    # State mapping:
    # 0: Type A Defector (C < 0.5, strategy=0) - Dark Blue
    # 1: Type A Cooperator (C < 0.5, strategy=1) - Light Blue
    # 2: Type B Defector (C >= 0.5, strategy=0) - Dark Red
    # 3: Type B Cooperator (C >= 0.5, strategy=1) - Light Red
    state_colors = {
        0: '#de425b',  # 深红   自利背叛
        3: '#488f31',  # 深绿   利他合作
        2: '#f3babc',  # 浅红   利他背叛
        1: '#bad0af'   # 浅绿   自利合作
    }
    # Create a colormap from the defined colors
    cmap = mcolors.ListedColormap([state_colors[i] for i in sorted(state_colors.keys())])
    bounds = sorted(state_colors.keys()) + [max(state_colors.keys()) + 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create a grid to store agent states
    grid_states = np.full((L, L), -1, dtype=int) # Use -1 for empty cells if any

    # Populate the grid with agent states
    for agent in agents_data:
        ix = agent.get("ix")
        iy = agent.get("iy")
        strategy = agent.get("strategy")
        C = agent.get("C")

        if ix is not None and iy is not None and strategy is not None and C is not None:
            if C < 0.5: # Type A
                if strategy == 0: # Defector        自利背叛
                    grid_states[iy][ix] = 0
                else: # Cooperator          自利合作
                    grid_states[iy][ix] = 1
            else: # Type B
                if strategy == 0: # Defector   利他背叛
                    grid_states[iy][ix] = 2
                else: # Cooperator      利他合作
                    grid_states[iy][ix] = 3

    # Plot the grid
    fig, ax = plt.subplots(figsize=(6, 6)) # Adjust figure size as needed
    # Use imshow to display the grid
    im = ax.imshow(grid_states, cmap=cmap, norm=norm, origin='lower') # origin='lower' matches ix, iy as (0,0) at bottom-left

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

    # Create a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Self-Interested Defector', markersize=10, markerfacecolor=state_colors[0]),
        plt.Line2D([0], [0], marker='s', color='w', label='Other-Regarding Defector', markersize=10, markerfacecolor=state_colors[2]),
        plt.Line2D([0], [0], marker='s', color='w', label='Self-Interested Cooperator', markersize=10, markerfacecolor=state_colors[1]),
        plt.Line2D([0], [0], marker='s', color='w', label='Other-Regarding Cooperator', markersize=10, markerfacecolor=state_colors[3])
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    # Add a title (optional, can include parameters from environment)
    b_value = snapshot_data.get("environment", {}).get("b")
    step = snapshot_data.get("step") # Assuming step is available in the snapshot data
    if b_value is not None and step is not None:
         plt.title(f"Spatial Snapshot (L={L}, b={b_value:.2f}, Step={step})")
    elif b_value is not None:
         plt.title(f"Spatial Snapshot (L={L}, b={b_value:.2f})")
    elif step is not None:
         plt.title(f"Spatial Snapshot (L={L}, Step={step})")
    else:
         plt.title(f"Spatial Snapshot (L={L})")


    plt.tight_layout() # Adjust layout to prevent legend overlap

    # Save the plot
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_filename, dpi=300) # Save with high resolution
    print(f"Snapshot plot saved to {output_filename}")

    plt.close(fig) # Close the figure to free up memory

def batch_plot_snapshots_single_folder(root_directory="snapshot_output", output_directory="batch_plots_single"):
    """
    Traverses subdirectories in the root_directory, finds snapshot JSON files,
    and generates plots for each, saving all plots to a single output directory.

    Args:
        root_directory (str): The base directory containing simulation run subdirectories.
        output_directory (str): The single directory to save all generated plots.
    """
    print(f"Starting batch plotting from directory: {root_directory}")
    print(f"Saving all plots to single directory: {output_directory}")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Use os.walk to traverse directories recursively
    for subdir, dirs, files in os.walk(root_directory):
        # Look for JSON files in the current subdirectory
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(subdir, file)

                # Attempt to extract information from the file path or filename
                # This assumes a directory structure like root_directory/b_X.X/snapshot_step_Y.json
                try:
                    # Extract b value from the subdirectory name
                    b_value_str = os.path.basename(subdir).replace("b_", "")
                    b_value = float(b_value_str)
                except (ValueError, AttributeError):
                    b_value = "unknown_b" # Handle cases where b value cannot be extracted

                # Extract step number from the filename
                step_str = file.replace("snapshot_step_", "").replace(".json", "")
                try:
                    step = int(step_str)
                except ValueError:
                    step = "unknown_step" # Handle cases where step cannot be extracted

                # Create a meaningful output filename in the single output directory
                # Example: batch_plots_single/b_1.6_step_100.png
                output_filename_png = os.path.join(output_directory, f"b_{b_value}_step_{step}.png")
                output_filename_pdf = os.path.join(output_directory, f"b_{b_value}_step_{step}.pdf")

                print(f"Processing snapshot: {json_file_path}")
                plot_single_snapshot(json_file_path, output_filename_png)
#                plot_single_snapshot(json_file_path, output_filename_pdf)

    print("Batch plotting complete.")


# --- Example Usage ---
if __name__ == "__main__":
    # Define the root directory where your simulation outputs are saved
    simulation_output_root = "snapshot_output"

    # Define the single directory where you want to save all generated plots
    plot_output_single_folder = "batch_plots_single"

    batch_plot_snapshots_single_folder(simulation_output_root, plot_output_single_folder)
