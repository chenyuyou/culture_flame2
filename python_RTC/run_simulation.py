# run_simulation.py (修正版)

import pyflamegpu
import sys
import os
import random # Use standard random ONLY for initial population generation
# import math # No longer needed here if agent logic is in C++
from typing import List # Optional type hinting

# --- Import model definition ---
# This now imports the define_model function which sets up everything including RTC functions
try:
    # model_definition now defines the complete model structure
    # MAX_NEIGHBORS might still be needed here for population initialization if needed
    from model_definition import define_model, MAX_NEIGHBORS
    # We no longer import agent functions from agent_functions.py
except ImportError as e:
    print(f"Error importing from model_definition: {e}")
    print("Ensure model_definition.py and cuda_kernels.py are accessible.")
    sys.exit(1)

# --- Simulation Parameters ---
GRID_WIDTH = 50.0       # Use float for coordinates
GRID_HEIGHT = 50.0      # Use float for coordinates
N_AGENTS = GRID_WIDTH * GRID_HEIGHT # Example: Fill the grid initially
# Ensure N_AGENTS is an integer if used for loops range
N_AGENTS = int(N_AGENTS)

SIMULATION_STEPS = 100
LOGGING_FREQUENCY = 10 # How often to log data (optional)
RANDOM_SEED = 666       # Seed for reproducibility (for FLAME GPU's RNG)

# --- Environment Parameters ---
# These will override the defaults set in model_definition.py
# Payoff Matrix (T > R > P > S) - Prisoner's Dilemma example
PAYOFF_T = 1.9  # Temptation
PAYOFF_R = 1.0  # Reward (Matches default)
PAYOFF_P = 0.0  # Punishment (Matches default)
PAYOFF_S = -0.1 # Sucker

# Noise and Update Parameters
PARAM_K = 0.1           # Noise in strategy update (Fermi K)
PARAM_K_C = 0.05        # Noise in culture update (Fermi K_C)
PARAM_P_UPDATE_C = 0.1  # Probability of attempting cultural update per step
MUTATION_RATE_STRATEGY = 0.01 # Mutation rate for strategy
MUTATION_RATE_C = 0.01        # Mutation rate for C value

# -----------------------------
# Main Simulation Setup
# -----------------------------
def run():
    print("--- Starting FLAME GPU Cultural Evolution Simulation (with C++ RTC Kernels) ---")

    # --- 1. Define Model ---
    print("Defining model structure (including agent RTC functions)...")
    # define_model now returns the complete model description with RTC functions attached
    model = define_model()

    # --- 2. Create CUDASimulation ---
    print("Creating CUDASimulation object...")
    # Use pyflamegpu.LOG_INFO or pyflamegpu.LOG_DEBUG for more verbose output if needed
    simulation = pyflamegpu.CUDASimulation(model, verbosity=pyflamegpu.LOG_WARNING)

    # --- 3. Configure Simulation Settings ---
    print("Configuring simulation settings (seed, steps)...")
    sim_config = simulation.SimulationConfig()
    sim_config.random_seed = RANDOM_SEED
    sim_config.steps = SIMULATION_STEPS
    # Set input/output files if needed
    # sim_config.input_file = "init.xml"
    # sim_config.output_file = "end.xml"

    # --- 4. Set Environment Properties ---
    # Override defaults defined in model_definition.py if necessary
    print("Setting environment properties...")
    env = simulation.environment # Get the environment property interface
    # Payoffs
    env.setPropertyFloat("T_PAYOFF", PAYOFF_T)
    env.setPropertyFloat("R_PAYOFF", PAYOFF_R)
    env.setPropertyFloat("P_PAYOFF", PAYOFF_P)
    env.setPropertyFloat("S_PAYOFF", PAYOFF_S)
    # Noise
    env.setPropertyFloat("K", PARAM_K)
    env.setPropertyFloat("K_C", PARAM_K_C)
    # Probabilities
    env.setPropertyFloat("p_update_C", PARAM_P_UPDATE_C)
    env.setPropertyFloat("mutation_rate_strategy", MUTATION_RATE_STRATEGY)
    env.setPropertyFloat("mutation_rate_C", MUTATION_RATE_C)
    # Grid dimensions (if needed by environment or visualization)
    # env.setPropertyFloat("grid_width", GRID_WIDTH)
    # env.setPropertyFloat("grid_height", GRID_HEIGHT)

    # --- 5. Generate Initial Agent Population ---
    print(f"Generating initial population of {N_AGENTS} agents...")
    # Create an AgentVector based on the "CulturalAgent" definition in the model
    # The agent name must match the one defined in model_definition.py
    agent_vector = pyflamegpu.AgentVector(model.Agent("CulturalAgent"), N_AGENTS)

    for i in range(N_AGENTS):
        agent = agent_vector[i]

        # Assign random initial position within the grid
        # Using agent variables "x", "y" defined in model_definition.py
        agent.setVariableFloat("x", random.uniform(0.0, GRID_WIDTH))
        agent.setVariableFloat("y", random.uniform(0.0, GRID_HEIGHT))

        # Assign random initial strategy (0 or 1)
        initial_strategy = random.choice([0, 1])
        agent.setVariableInt32("strategy", initial_strategy)
        # Initialize next state variables to current state initially
        agent.setVariableInt32("next_strategy", initial_strategy)

        # Assign random initial C value (0.0 to 1.0)
        initial_C = random.random() # Standard python random for init
        agent.setVariableFloat("C", initial_C)
        # Initialize next state
        agent.setVariableFloat("next_C", initial_C)

        # Initialize other variables to sensible defaults
        agent.setVariableFloat("utility", 0.0)
        agent.setVariableUInt32("neighbor_count", 0)

        # Initialize neighbor data arrays (optional, default should be 0/null)
        # C++ kernels should handle reading potentially uninitialized values if neighbor_count=0
        # for j in range(MAX_NEIGHBORS):
        #    agent.setVariableInt32Array("neighbor_strategies", j, 0)
        #    agent.setVariableFloatArray("neighbor_Cs", j, 0.0)
        #    agent.setVariableFloatArray("neighbor_utilities", j, 0.0)

    # Set the generated population in the simulation
    simulation.setPopulationData(agent_vector)
    print("Agent population initialized.")

    # --- 6. Define Simulation Layers (Execution Order) ---
    # Layers define the sequence of agent function execution within a step.
    # We now refer to agent functions by the *string names* given to newRTCFunction.
    print("Defining simulation layers using RTC function names...")

    # Layer 1: Output current state (strategy, C) via StateMessage
    layer1 = simulation.newLayer("Output State")
    # Add agent function by AgentName and FunctionName (as defined in model_definition)
    layer1.addAgentFunction("CulturalAgent", "output_state")
    # This layer outputs messages, declare it (optional but good practice)
    # layer1.addMessageOutput("StateMessage") # Implicit if function has output

    # Layer 2: Calculate utility based on received StateMessages
    layer2 = simulation.newLayer("Calculate Utility")
    # This layer reads messages, declare it (optional but good practice)
    # layer2.addMessageInput("StateMessage") # Implicit if function has input
    layer2.addAgentFunction("CulturalAgent", "calculate_utility")

    # Layer 3: Output calculated utility via UtilityMessage
    layer3 = simulation.newLayer("Output Utility")
    layer3.addAgentFunction("CulturalAgent", "output_utility")
    # layer3.addMessageOutput("UtilityMessage")

    # Layer 4: Process neighbor utilities read from UtilityMessage
    layer4 = simulation.newLayer("Process Neighbor Utility")
    # layer4.addMessageInput("UtilityMessage")
    layer4.addAgentFunction("CulturalAgent", "process_neighbor_utility")

    # Layer 5: Decide next strategy based on stored neighbor utilities
    layer5 = simulation.newLayer("Decide Strategy Update")
    layer5.addAgentFunction("CulturalAgent", "decide_strategy_update")

    # Layer 6: Decide next C value based on stored neighbor utilities
    layer6 = simulation.newLayer("Decide Culture Update")
    layer6.addAgentFunction("CulturalAgent", "decide_culture_update")

    # Layer 7: Apply random mutations to next_strategy and next_C
    layer7 = simulation.newLayer("Mutate")
    layer7.addAgentFunction("CulturalAgent", "mutate")

    # Layer 8: Advance state (update strategy and C from next_*)
    layer8 = simulation.newLayer("Advance State")
    layer8.addAgentFunction("CulturalAgent", "advance")

    print("Simulation layers defined.")

    # --- 7. Configure Simulation Run (Logging/Visualization - Optional) ---
    # Add logging, visualization, exit conditions etc. here if needed
    # Example basic logging:
    # step_log = simulation.getStepLog() # Requires FLAMEGPU_ENABLE_LOGGING=1 ? Check docs.
    # step_log.setFrequency(LOGGING_FREQUENCY)
    # step_log.logEnvironment("K", "K_C", "p_update_C") # Log specific env vars
    # step_log.logAgent("CulturalAgent", "strategy", "C", "utility") # Log agent vars

    # Visualization setup (similar to model1.py example)
    if pyflamegpu.VISUALISATION:
        print("Setting up visualization...")
        visualisation = simulation.getVisualisation()
        # Configure camera, background, etc.
        vis_env_width = GRID_WIDTH # Use parameters for camera setup
        vis_env_height = GRID_HEIGHT
        visualisation.setInitialCameraLocation(vis_env_width / 2.0, vis_env_height / 2.0, max(vis_env_width, vis_env_height) * 1.2) # Zoom out based on size
        visualisation.setInitialCameraTarget(vis_env_width / 2.0, vis_env_height / 2.0, 0.0)
        visualisation.setCameraSpeed(0.01 * max(vis_env_width, vis_env_height)) # Adjust speed based on size
        visualisation.setClearColor(0.2, 0.2, 0.2) # Dark background

        # Configure agent visualization
        agt_vis = visualisation.addAgent("CulturalAgent")
        agt_vis.setX("x") # Use agent variables for position
        agt_vis.setY("y")
        # Use a simple model like a sphere or cube
        agt_vis.setModel(pyflamegpu.SPHERE)
        agt_vis.setModelScale(0.5) # Adjust size as needed

        # Color agents based on strategy (e.g., Cooperate=Blue, Defect=Red)
        # strategy_colors = pyflamegpu.uDiscreteColor("strategy")
        # strategy_colors[0] = pyflamegpu.RED   # Defect (strategy=0)
        # strategy_colors[1] = pyflamegpu.BLUE  # Cooperate (strategy=1)
        # agt_vis.setColor(strategy_colors)

        # Or color based on C value (e.g., gradient Black->White)
        c_color = pyflamegpu.uContinuousColor("C", pyflamegpu.BLACK, pyflamegpu.WHITE)
        agt_vis.setColor(c_color)


        # Activate visualization window
        visualisation.activate()
        visualisation.setBeginPaused(True) # Start paused

    # --- 8. Run Simulation ---
    print(f"Starting simulation for {SIMULATION_STEPS} steps...")
    simulation.simulate()
    print("--- Simulation Finished ---")

    # --- 9. Post-processing / Data Extraction (Optional) ---
    print("Extracting final agent states...")
    # Get final population data back into an AgentVector
    # Create a new vector or reuse the initial one if structure is identical
    final_population = pyflamegpu.AgentVector(model.Agent("CulturalAgent"))
    simulation.getPopulationData(final_population)

    # Example: Calculate final average C and cooperation rate
    if len(final_population) > 0:
        final_Cs = [agent.getVariableFloat("C") for agent in final_population]
        # Strategy is Int32, but calculation works the same
        final_strategies = [agent.getVariableInt32("strategy") for agent in final_population]

        avg_C = sum(final_Cs) / len(final_Cs)
        # Assumes strategy 1 = Cooperate, 0 = Defect
        cooperation_rate = sum(final_strategies) / len(final_strategies)

        print(f"Final Agent Count: {len(final_population)}")
        print(f"Final Average C value: {avg_C:.4f}")
        print(f"Final Cooperation Rate (Strategy=1): {cooperation_rate:.4f}")
    else:
        print("No agents remaining in the final state.")

    # Access logs if logging was enabled and configured properly
    # Cleanup (especially important if using certain features like visualization?)
    # pyflamegpu.cleanup() # Usually called automatically on exit? Check docs.

# --- Entry Point ---
if __name__ == "__main__":
    # Basic command line argument parsing (Example)
    # You might want to use argparse for more robust handling
    args = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            args[key.upper()] = value # Store keys in upper case for consistency

    # Override parameters if provided via command line
    N_AGENTS = int(args.get("N_AGENTS", N_AGENTS))
    SIMULATION_STEPS = int(args.get("STEPS", SIMULATION_STEPS)) # Common alternative name
    PARAM_K = float(args.get("K", PARAM_K))
    PARAM_K_C = float(args.get("K_C", PARAM_K_C))
    PAYOFF_T = float(args.get("T", PAYOFF_T)) # Allow overriding T
    # Add other parameters as needed...

    print("--- Effective Parameters ---")
    print(f"  N_AGENTS: {N_AGENTS}")
    print(f"  SIMULATION_STEPS: {SIMULATION_STEPS}")
    print(f"  PAYOFF_T: {PAYOFF_T}")
    print(f"  PARAM_K: {PARAM_K}")
    print(f"  PARAM_K_C: {PARAM_K_C}")
    print(f"  PARAM_P_UPDATE_C: {PARAM_P_UPDATE_C}")
    print(f"  MUTATION_RATE_STRATEGY: {MUTATION_RATE_STRATEGY}")
    print(f"  MUTATION_RATE_C: {MUTATION_RATE_C}")
    print("--------------------------")

    # Run the simulation setup and execution function
    run()
