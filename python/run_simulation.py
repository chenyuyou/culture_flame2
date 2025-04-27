import pyflamegpu
import sys
import os
import random # Use standard random for initial population generation ONLY
import math
from typing import List # Optional type hinting

# --- Import model definition and agent functions ---
# Ensure model_definition.py and agent_functions.py are in the same directory
# or accessible via PYTHONPATH
try:
    from model_definition import define_model, GRID_WIDTH, GRID_HEIGHT, INTERACTION_RADIUS, MAX_NEIGHBORS
    from agent_functions import (
        output_state,
        process_neighbors,
        calculate_utility,
        output_utility,
        process_neighbor_utility,
        decide_strategy_update,
        decide_culture_update,
        mutate,
        advance
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure model_definition.py and agent_functions.py are in the current directory or Python path.")
    sys.exit(1)

# --- Simulation Parameters ---
N_AGENTS = 1000         # Number of agents
SIMULATION_STEPS = 100  # Number of simulation steps
LOGGING_FREQUENCY = 10 # How often to log data (optional)
RANDOM_SEED = 666       # Seed for reproducibility (for FLAME GPU's RNG)

# --- Environment Parameters ---
# Payoff Matrix (T > R > P > S) - Prisoner's Dilemma example
# T=Temptation, R=Reward, P=Punishment, S=Sucker
PAYOFF_T = 1.9
PAYOFF_R = 1.0
PAYOFF_P = 0.0
PAYOFF_S = -0.1

# Noise and Update Parameters
PARAM_K = 0.1           # Noise in strategy update (Fermi K)
PARAM_K_C = 0.05          # Noise in culture update (Fermi K_C)
PARAM_P_UPDATE_C = 0.1  # Probability of attempting cultural update per step
MUTATION_RATE_STRATEGY = 0.01 # Mutation rate for strategy
MUTATION_RATE_C = 0.01        # Mutation rate for C value

# -----------------------------
# Main Simulation Setup
# -----------------------------
def run():
    print("--- Starting FLAME GPU Cultural Evolution Simulation ---")

    # --- 1. Define Model ---
    print("Defining model...")
    model = define_model()

    # --- 2. Create CUDASimulation ---
    # Use pyflamegpu.CUDAEnsembleController for parameter sweeps later if needed
    print("Creating CUDASimulation object...")
    # Enable verbose mode for debugging if needed: pyflamegpu.LOG_TRACE, pyflamegpu.LOG_INFO, pyflamegpu.LOG_WARNING etc.
    simulation = pyflamegpu.CUDASimulation(model, verbosity=pyflamegpu.LOG_INFO)
    # Set simulation random seed
    simulation.SimulationConfig().random_seed = RANDOM_SEED
    # Set simulation steps
    simulation.SimulationConfig().steps = SIMULATION_STEPS

    # --- 3. Set Environment Properties ---
    print("Setting environment properties...")
    env = simulation.environment # Get the environment property interface
    env.setPropertyFloat("T", PAYOFF_T)
    env.setPropertyFloat("R", PAYOFF_R)
    env.setPropertyFloat("P", PAYOFF_P)
    env.setPropertyFloat("S", PAYOFF_S)
    env.setPropertyFloat("K", PARAM_K)
    env.setPropertyFloat("K_C", PARAM_K_C)
    env.setPropertyFloat("p_update_C", PARAM_P_UPDATE_C)
    env.setPropertyFloat("mutation_rate_strategy", MUTATION_RATE_STRATEGY)
    env.setPropertyFloat("mutation_rate_C", MUTATION_RATE_C)
    # Ensure GRID_WIDTH, GRID_HEIGHT, INTERACTION_RADIUS are also available if needed by agent functions
    # If they are constant, they can be accessed directly via import
    # If they need to be dynamic environment properties:
    # env.setPropertyFloat("GRID_WIDTH", GRID_WIDTH)
    # env.setPropertyFloat("GRID_HEIGHT", GRID_HEIGHT)
    # env.setPropertyFloat("INTERACTION_RADIUS", INTERACTION_RADIUS)


    # --- 4. Generate Initial Agent Population ---
    print(f"Generating initial population of {N_AGENTS} agents...")
    # Create an AgentVector based on the "CulturalAgent" definition in the model
    population = pyflamegpu.AgentVector(model.Agent("CulturalAgent"), N_AGENTS)

    for i in range(N_AGENTS):
        agent = population[i]

        # Assign random initial position (ensure within bounds if using fixed space)
        # FLAME GPU spatial models usually handle wrapping or boundaries
        agent.setVariableFloat("x", random.uniform(0, GRID_WIDTH))
        agent.setVariableFloat("y", random.uniform(0, GRID_HEIGHT))

        # Assign random initial strategy (0 or 1)
        initial_strategy = random.choice([0, 1])
        agent.setVariableInt32("strategy", initial_strategy)
        agent.setVariableInt32("next_strategy", initial_strategy) # Initialize next state

        # Assign random initial C value (0.0 to 1.0)
        initial_C = random.random() # Standard python random for init
        agent.setVariableFloat("C", initial_C)
        agent.setVariableFloat("next_C", initial_C) # Initialize next state

        # Initialize other variables
        agent.setVariableFloat("utility", 0.0)
        agent.setVariableUInt32("neighbor_count", 0)

        # Initialize neighbor data arrays (optional, default should be 0 or equivalent)
        # For clarity, you might initialize them if needed, e.g., to -1 or NaN
        # for j in range(MAX_NEIGHBORS):
        #    agent.setVariableInt32Array("neighbor_strategies", j, -1) # Example init value
        #    agent.setVariableFloatArray("neighbor_Cs", j, -1.0)       # Example init value
        #    agent.setVariableFloatArray("neighbor_utilities", j, -1.0) # Example init value

    # Set the generated population in the simulation
    simulation.setPopulationData(population)
    print("Agent population initialized.")

    # --- 5. Define Simulation Layers ---
    print("Defining simulation layers...")
    # Layer 1: Output state for neighbor discovery
    layer1 = simulation.newLayer("Output State")
    layer1.addAgentFunction(output_state)
    layer1.addMessageOutput("StateMessage")

    # Layer 2: Process neighbors based on state messages
    layer2 = simulation.newLayer("Process Neighbors")
    layer2.addAgentFunction(process_neighbors)
    layer2.addMessageInput("StateMessage")

    # Layer 3: Calculate utility based on neighbors found
    layer3 = simulation.newLayer("Calculate Utility")
    layer3.addAgentFunction(calculate_utility)

    # Layer 4: Output utility for imitation
    layer4 = simulation.newLayer("Output Utility")
    layer4.addAgentFunction(output_utility)
    layer4.addMessageOutput("UtilityMessage")

    # Layer 5: Process neighbor utilities
    layer5 = simulation.newLayer("Process Neighbor Utility")
    layer5.addAgentFunction(process_neighbor_utility)
    layer5.addMessageInput("UtilityMessage")

    # Layer 6: Decide strategy update (Fermi rule)
    layer6 = simulation.newLayer("Decide Strategy Update")
    layer6.addAgentFunction(decide_strategy_update)

    # Layer 7: Decide culture update (Fermi rule)
    layer7 = simulation.newLayer("Decide Culture Update")
    layer7.addAgentFunction(decide_culture_update)

    # Layer 8: Apply mutations
    layer8 = simulation.newLayer("Mutate")
    layer8.addAgentFunction(mutate)

    # Layer 9: Advance state (update strategy and C)
    layer9 = simulation.newLayer("Advance State")
    layer9.addAgentFunction(advance)

    print("Simulation layers defined.")

    # --- 6. Configure Simulation Run (Logging/Visualization - Optional) ---
    # Example: Add logging for agent variables every N steps
    # logger = simulation.addLog() # Basic logger
    # logger.logEnvironment("K", "K_C") # Log environment properties
    # agent_log = logger.logAgents("CulturalAgent") # Log agent states
    # agent_log.logVariable("strategy")
    # agent_log.logVariable("C")
    # agent_log.setFrequency(LOGGING_FREQUENCY)

    # Example: Add exit condition based on average C value (advanced)
    # class AvgCCheck(pyflamegpu.HostConditionCallback):
    #     def __init__(self, threshold):
    #         super().__init__()
    #         self.threshold = threshold
    #     def run(self, FLAMEGPU):
    #         agents = FLAMEGPU.getAllAgentData("CulturalAgent")
    #         if not agents: return False # No agents, don't exit
    #         avg_c = sum(a.getVariableFloat("C") for a in agents) / len(agents)
    #         print(f"Step {FLAMEGPU.getStepCounter()}: Avg C = {avg_c:.4f}")
    #         return avg_c > self.threshold # Exit if avg C exceeds threshold
    # simulation.addExitConditionCallback(AvgCCheck(threshold=0.9))


    # --- 7. Run Simulation ---
    print(f"Starting simulation for {SIMULATION_STEPS} steps...")
    simulation.simulate()
    print("--- Simulation Finished ---")

    # --- 8. Post-processing / Data Extraction (Optional) ---
    print("Extracting final agent states...")
    # Get final population data
    final_population = pyflamegpu.AgentVector(model.Agent("CulturalAgent"))
    simulation.getPopulationData(final_population)

    # Example: Calculate final average C and cooperation rate
    if len(final_population) > 0:
        final_Cs = [agent.getVariableFloat("C") for agent in final_population]
        final_strategies = [agent.getVariableInt32("strategy") for agent in final_population] # Assume 1=Cooperate, 0=Defect

        avg_C = sum(final_Cs) / len(final_Cs)
        cooperation_rate = sum(final_strategies) / len(final_strategies) # Fraction of cooperators

        print(f"Final Average C value: {avg_C:.4f}")
        print(f"Final Cooperation Rate (Strategy=1): {cooperation_rate:.4f}")
    else:
        print("No agents remaining in the final state.")

    # You can also access log files if logging was configured

# --- Entry Point ---
if __name__ == "__main__":
    # Check for command line arguments, e.g., to override parameters
    # Example: python run_simulation.py N_AGENTS=2000 STEPS=50
    # (This part is basic, needs proper parsing if used extensively)
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            try:
                # Attempt to set global variables based on argument names
                if key == "N_AGENTS": N_AGENTS = int(value)
                elif key == "SIMULATION_STEPS": SIMULATION_STEPS = int(value)
                elif key == "K": PARAM_K = float(value)
                elif key == "K_C": PARAM_K_C = float(value)
                # Add other parameters as needed
                else: print(f"Warning: Unknown parameter '{key}'")
            except ValueError:
                print(f"Warning: Invalid value '{value}' for parameter '{key}'")

    # Run the simulation setup and execution function
    run()
