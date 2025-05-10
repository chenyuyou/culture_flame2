import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def plot_snapshot(json_file_path, output_filename="snapshot.png"):
    """
    Reads FLAMEGPU snapshot JSON, plots the spatial distribution of agents
    based on culture type and strategy, and saves the plot.

    Args:
        json_file_path (str): Path to the end_snapshot.json file.
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
        0: '#de425b',  # Dark Blue
        1: '#488f31',  # Light Blue
        2: '#f3babc',  # Dark Red
        3: '#bad0af'   # Light Red
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
                if strategy == 0: # Defector
                    grid_states[iy][ix] = 0
                else: # Cooperator
                    grid_states[iy][ix] = 1
            else: # Type B
                if strategy == 0: # Defector
                    grid_states[iy][ix] = 2
                else: # Cooperator
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
        plt.Line2D([0], [0], marker='s', color='w', label='Type A Defector', markersize=10, markerfacecolor=state_colors[0]),
        plt.Line2D([0], [0], marker='s', color='w', label='Type A Cooperator', markersize=10, markerfacecolor=state_colors[1]),
        plt.Line2D([0], [0], marker='s', color='w', label='Type B Defector', markersize=10, markerfacecolor=state_colors[2]),
        plt.Line2D([0], [0], marker='s', color='w', label='Type B Cooperator', markersize=10, markerfacecolor=state_colors[3])
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    # Add a title (optional, can include parameters from environment)
    b_value = snapshot_data.get("environment", {}).get("b")
    if b_value is not None:
         plt.title(f"Spatial Snapshot (L={L}, b={b_value:.2f})")
    else:
         plt.title(f"Spatial Snapshot (L={L})")


    plt.tight_layout() # Adjust layout to prevent legend overlap

    # Save the plot
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_filename, dpi=300) # Save with high resolution
    print(f"Snapshot plot saved to {output_filename}")

    # plt.show() # Uncomment to display the plot immediately

# --- Example Usage ---
if __name__ == "__main__":
    # Assuming your end_snapshot.json is in the current directory or a subdirectory
    # Replace with the actual path to your JSON file
    snapshot_file = "snapshot_output/end_snapshot.json" # Example path

    # Define the output filename for the plot
    # You might want to make this dynamic based on parameters if you plot multiple snapshots
    output_plot_file = "Figure/snapshot_plot.png" # Example output path

    plot_snapshot(snapshot_file, output_plot_file)
