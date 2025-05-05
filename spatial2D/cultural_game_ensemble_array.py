# cultural_game_ensemble_array_staged.py
import pyflamegpu
import sys, random, math, time
import numpy as np
# Import from the NEW CUDA file with STAGED functions
from cultural_game_cuda_array import (
    output_state_func,
    calculate_utility_func,
    decide_updates_func,
    mutate_func,
    advance_func
)

# Default Simulation Parameters (same as before)
DEFAULT_PARAMS = {
    "L": 50,
    "initial_coop_ratio": 0.5,
    "b": 1.5,
    "K": 0.1,
    "C_dist": "uniform", "mu": 0.5, "sigma": 0.1,
    "K_C": 0.1, "p_update_C": 0.1,
    "p_mut_culture": 0.01, "p_mut_strategy": 0.001,
    "steps": 500, "seed": 1234,
}

def create_model(model_name="CulturalGameArrayStaged"):
    """Creates the FLAMEGPU ModelDescription."""
    model = pyflamegpu.ModelDescription(model_name)
    return model

def define_environment(model, params):
    """Defines environment properties (same as before)."""
    env = model.Environment()
    env.newPropertyFloat("b", params["b"])
    env.newPropertyFloat("K", params["K"])
    env.newPropertyFloat("K_C", params["K_C"])
    env.newPropertyFloat("p_update_C", params["p_update_C"])
    env.newPropertyFloat("p_mut_culture", params["p_mut_culture"])
    env.newPropertyFloat("p_mut_strategy", params["p_mut_strategy"])
    env.newMacroPropertyFloat("payoff", 2, 2)
    env.newPropertyUInt("GRID_DIM_L", params["L"])
    # Reporting properties (same)
    env.newPropertyUInt("agent_count", params["L"] * params["L"])
    env.newPropertyUInt("cooperators", 0)
    env.newPropertyUInt("defectors", 0)
    env.newPropertyFloat("avg_C", 0.0)
    env.newPropertyFloat("std_C", 0.0)


def define_messages(model, params):
    """Defines messages using MessageArray2D."""
    L = params["L"]

    # Message for sharing state (strategy, C) from t-1
    msg_state = model.newMessageArray2D("agent_state_message")
    msg_state.newVariableID("id")
    msg_state.newVariableUInt("strategy")
    msg_state.newVariableFloat("C")
    msg_state.setDimensions(L, L)
#    msg_state.setBoundaryWrap(True) # Explicitly enable wrapping

    # NEW Message for sharing calculated utility and state needed for adoption
    msg_utility = model.newMessageArray2D("utility_message")
    msg_utility.newVariableID("id")
    msg_utility.newVariableFloat("utility")     # Calculated utility @ t
    msg_utility.newVariableUInt("strategy")     # Strategy @ t-1 (for adoption)
    msg_utility.newVariableFloat("C")           # C @ t-1 (for adoption)
    msg_utility.setDimensions(L, L)
#    msg_utility.setBoundaryWrap(True) # Explicitly enable wrapping


def define_agents(model):
    """Defines the 'CulturalAgent' agent type."""
    agent = model.newAgent("CulturalAgent")

    # Agent variables
    agent.newVariableInt("ix")             # Grid index X
    agent.newVariableInt("iy")             # Grid index Y
    agent.newVariableUInt("strategy")      # Current strategy
    agent.newVariableFloat("C")            # Current C
    agent.newVariableUInt("next_strategy") # Buffer for next strategy
    agent.newVariableFloat("next_C")       # Buffer for next C
    # NEW variable to store calculated utility
    agent.newVariableFloat("current_utility") # Utility calculated in step t

    # Link agent functions to the C++ code strings
    fn = agent.newRTCFunction("output_state_func", output_state_func)
    fn.setMessageOutput("agent_state_message") # Outputs state

    fn = agent.newRTCFunction("calculate_utility_func", calculate_utility_func)
    fn.setMessageInput("agent_state_message")  # Reads state @ t-1
    fn.setMessageOutput("utility_message")     # Outputs utility @ t and state

    fn = agent.newRTCFunction("decide_updates_func", decide_updates_func)
    fn.setMessageInput("utility_message")      # Reads utility @ t and state

    fn = agent.newRTCFunction("mutate_func", mutate_func)
    # No message input/output

    fn = agent.newRTCFunction("advance_func", advance_func)
    # No message input/output


def define_execution_order(model):
    """Defines the layers and execution order for STAGED logic."""
    # Layer 1: Output state from t-1
    layer1 = model.newLayer("Output State")
    layer1.addAgentFunction("CulturalAgent", "output_state_func")

    # Layer 2: Calculate utility based on t-1 state, store it, and output for neighbors
    layer2 = model.newLayer("Calculate Utility")
    layer2.addAgentFunction("CulturalAgent", "calculate_utility_func")

    # Layer 3: Decide next strategy/C based on utilities calculated in Layer 2
    layer3 = model.newLayer("Decide Updates")
    layer3.addAgentFunction("CulturalAgent", "decide_updates_func")

    # Layer 4: Apply mutation to the decided next state
    layer4 = model.newLayer("Mutate")
    layer4.addAgentFunction("CulturalAgent", "mutate_func")

    # Layer 5: Advance state for the next step
    layer5 = model.newLayer("Advance State")
    layer5.addAgentFunction("CulturalAgent", "advance_func")


# --- Host Functions (InitFunction, StepFunction) ---

class InitFunction(pyflamegpu.HostFunction):
    """Host function run once at the beginning."""
    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self, FLAMEGPU):
        L = self.params["L"]
        agent_count = L * L
        b = self.params["b"]
        initial_coop_ratio = self.params["initial_coop_ratio"]
        C_dist = self.params["C_dist"]
        mu = self.params["mu"]
        sigma = self.params["sigma"]

        # Set environment properties
        FLAMEGPU.environment.setPropertyUInt("agent_count", agent_count)
        FLAMEGPU.environment.setPropertyUInt("GRID_DIM_L", L)
        payoff = FLAMEGPU.environment.getMacroPropertyFloat("payoff")
        # Payoff: [My Strategy][Neighbor Strategy] -> My Payoff
        # Assuming payoff matrix in environment gives MY payoff only.
        # Need to check if agent code assumes (my_payoff, neighbor_payoff) tuple.
        # Let's assume payoff gives MY payoff. Agent code calculates neighbor payoff implicitly.
        # P(C,C)=1, P(C,D)=0, P(D,C)=b, P(D,D)=0
        # Row Player (Me), Column Player (Neighbor)
        payoff[1][1] = 1.0 # Me C, Neighbor C -> My Payoff = 1
        payoff[1][0] = 0.0 # Me C, Neighbor D -> My Payoff = 0
        payoff[0][1] = b   # Me D, Neighbor C -> My Payoff = b
        payoff[0][0] = 0.0 # Me D, Neighbor D -> My Payoff = 0
        # Agent code calculate_utility_func looks correct based on this payoff interpretation.

        print(f"Initializing {agent_count} agents on a {L}x{L} wrapped grid (Staged Array)...")
        # ... (rest of print statement)

#        agent_pop_ref = FLAMEGPU.model.Agent("CulturalAgent")
#        population = pyflamegpu.AgentVector(agent_pop_ref, agent_count)
        agents = FLAMEGPU.agent("CulturalAgent")
        for i in range(agent_count):
            agent = agents.newAgent()
            ix = i % L
            iy = i // L
            agent.setVariableInt("ix", ix)
            agent.setVariableInt("iy", iy)

            strategy = 1 if random.random() < initial_coop_ratio else 0
            agent.setVariableUInt("strategy", strategy)
            agent.setVariableUInt("next_strategy", strategy) # Initialize buffer

            # Initial Culture (C) - same logic as before
            C_value = 0.0
            if C_dist == "uniform": C_value = random.uniform(0, 1)
            elif C_dist == "bimodal": C_value = 0.0 if random.random() < (1.0 - mu) else 1.0
            elif C_dist == "normal": C_value = np.clip(random.normalvariate(mu, sigma), 0, 1)
            elif C_dist == "fixed": C_value = mu
            else: C_value = random.uniform(0, 1) # Default to uniform if unknown
            agent.setVariableFloat("C", C_value)
            agent.setVariableFloat("next_C", C_value) # Initialize buffer

            # Initialize the new utility variable
            agent.setVariableFloat("current_utility", 0.0)

#        FLAMEGPU.setPopulationData(population)


# StepFunction remains the same - calculates statistics based on final 'strategy' and 'C'
class StepFunction(pyflamegpu.HostFunction):
     """Host function run after each GPU step."""
     def run(self, FLAMEGPU):
        agent_count_gpu = FLAMEGPU.agent("CulturalAgent").count()
        FLAMEGPU.environment.setPropertyUInt("agent_count", agent_count_gpu)
        if agent_count_gpu > 0:
             # Use reduction to count cooperators efficiently
             coop_count = FLAMEGPU.agent("CulturalAgent").sumUInt("strategy") # Summing 1s gives count
             # Or using the count function:
             # coop_count = FLAMEGPU.agent("CulturalAgent").countUInt("strategy", 1)

             FLAMEGPU.environment.setPropertyUInt("cooperators", coop_count)
             FLAMEGPU.environment.setPropertyUInt("defectors", agent_count_gpu - coop_count)

             # Use reduction for mean and calculate std dev host side for simplicity
             # Note: Calculating std dev efficiently on GPU requires more complex reductions
#             avg_C = FLAMEGPU.agent("CulturalAgent").meanFloat("C")
#             FLAMEGPU.environment.setPropertyFloat("avg_C", avg_C)

             # Get all C values for standard deviation (less efficient but simple)
             try:
                 c_values = FLAMEGPU.agent("CulturalAgent").getVariableFloat("C")
                 std_C = np.std(c_values) if c_values else 0.0
                 FLAMEGPU.environment.setPropertyFloat("std_C", std_C)
             except Exception as e: # Handle potential errors if no agents exist
                 # print(f"Warning: Could not get C values for std dev: {e}")
                 FLAMEGPU.environment.setPropertyFloat("std_C", 0.0)

        else: # No agents left
             FLAMEGPU.environment.setPropertyUInt("cooperators", 0)
             FLAMEGPU.environment.setPropertyUInt("defectors", 0)
    ##         FLAMEGPU.environment.setPropertyFloat("avg_C", 0.0)
             FLAMEGPU.environment.setPropertyFloat("std_C", 0.0)


# --- Logging, Output, Runs Configuration ---

def define_logs(model):
    """Defines logging (same as before)."""
    log_cfg = pyflamegpu.StepLoggingConfig(model)
    log_cfg.setFrequency(1)
    # Log environment properties updated in StepFunction
    log_cfg.logEnvironment("cooperators")
    log_cfg.logEnvironment("defectors")
    log_cfg.logEnvironment("avg_C")
    log_cfg.logEnvironment("std_C")
    # Optionally log agent data too (can be large)
    # log_cfg.logAgentVariable("CulturalAgent", "current_utility", reduction=pyflamegpu.REDUCTION.MEAN)
    return log_cfg

def define_output(ensemble):
    """Configures simulation output."""
    # Change directory name to reflect Staged Array version
    ensemble.Config().out_directory = "results_flamegpu_array_staged"
    ensemble.Config().out_format = "json"
    ensemble.Config().timing = True
    ensemble.Config().truncate_log_files = True
    # Normal error level is fine
    ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Fast
    ensemble.Config().devices = pyflamegpu.IntSet([0]) # Use GPU 0

def define_run_plans(model, params=DEFAULT_PARAMS, num_runs=1):
    """Creates RunPlanVector for simulations (same as before)."""
    run_plans = pyflamegpu.RunPlanVector(model, num_runs)
    run_plans.setSteps(params["steps"])
    # Use simulation seed for agent initialization randomness within InitFunction
    # Use step seed for randomness within agent functions (like Fermi choice)
    run_plans.setRandomSimulationSeed(params["seed"], num_runs)

    # Set environment properties for the runs
    run_plans.setPropertyFloat("b", params["b"])
    run_plans.setPropertyFloat("K", params["K"])
    run_plans.setPropertyFloat("K_C", params["K_C"])
    run_plans.setPropertyFloat("p_update_C", params["p_update_C"])
    run_plans.setPropertyFloat("p_mut_culture", params["p_mut_culture"])
    run_plans.setPropertyFloat("p_mut_strategy", params["p_mut_strategy"])
    run_plans.setPropertyUInt("GRID_DIM_L", params["L"])

    return run_plans

# --- Main Simulation Execution ---
def run_simulation(params=DEFAULT_PARAMS, num_runs=1):
    """Sets up and runs the STAGED FLAMEGPU simulation."""
    print("--- Setting up STAGED Array-Based Cultural Game ---")
    model = create_model()
    define_environment(model, params)
    define_messages(model, params) # Defines both state and utility messages
    define_agents(model)           # Defines agent with utility variable and staged functions
    define_execution_order(model)  # Defines the 5 layers

    # Instantiate and add Host Functions
    init_func_instance = InitFunction(params)
    step_func_instance = StepFunction()
    model.addInitFunction(init_func_instance)
    model.addStepFunction(step_func_instance) # Runs after all GPU layers

    run_plans = define_run_plans(model, params, num_runs)
    log_config = define_logs(model)

    print("Initializing CUDA Ensemble...")
    # Use pyflamegpu.CUDAEnsemble for running on GPU
    ensemble_sim = pyflamegpu.CUDAEnsemble(model)
    define_output(ensemble_sim)
    ensemble_sim.setStepLog(log_config)

    print(f"--- Starting FLAMEGPU Simulation ({num_runs} runs, {params['steps']} steps) ---")
    start_time = time.time()
    ensemble_sim.simulate(run_plans)
    end_time = time.time()
    print(f"--- Simulation Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    output_dir = "results_flamegpu_array_staged" # Define here for final message
    print("Running single STAGED array-based simulation with default parameters...")
    run_simulation(params=DEFAULT_PARAMS, num_runs=1)

    # Example: Run with different parameters
    print("\nRunning STAGED array-based simulation with L=30, K=0.05...")
    mod_params = DEFAULT_PARAMS.copy()
    mod_params["L"] = 30
    mod_params["K"] = 0.05
    mod_params["seed"] = 5678 # Use a different seed
    # Ensure other params are set if needed
    # mod_params["steps"] = 100 # Shorter run for testing
    run_simulation(params=mod_params, num_runs=1)

    print(f"\nSimulation complete. Check the '{output_dir}' directory.")

