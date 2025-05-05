# cultural_game_ensemble.py
import pyflamegpu
import sys, random, math, time
import numpy as np
from cultural_game_cuda import * # Import C++/CUDA code strings

# Default Simulation Parameters (can be overridden by RunPlan)
DEFAULT_PARAMS = {
    "L": 50,
    "initial_coop_ratio": 0.5,
    "b": 1.5,
    "K": 0.1,
    "C_dist": "uniform", # <<< This will be handled by InitFunction
    "mu": 0.5,           # <<< This will be handled by InitFunction
    "sigma": 0.1,        # <<< This will be handled by InitFunction
    "K_C": 0.1,
    "p_update_C": 0.1,
    "p_mut_culture": 0.01,
    "p_mut_strategy": 0.001,
    "steps": 500,
    "seed": 1234
}


def create_model(model_name="CulturalGame"):
    """Creates the FLAMEGPU ModelDescription."""
    model = pyflamegpu.ModelDescription(model_name)
    return model

def define_environment(model, params):
    """Defines environment properties based on input parameters."""
    env = model.Environment()
    # Numerical simulation parameters needed by GPU/logging
    env.newPropertyFloat("b", params["b"])
    env.newPropertyFloat("K", params["K"])
    env.newPropertyFloat("K_C", params["K_C"])
    env.newPropertyFloat("p_update_C", params["p_update_C"])
    env.newPropertyFloat("p_mut_culture", params["p_mut_culture"])
    env.newPropertyFloat("p_mut_strategy", params["p_mut_strategy"])
    # Payoff matrix
    env.newMacroPropertyFloat("payoff", 2, 2)
    # Properties calculated/updated by host functions
    # L is needed to determine agent_count initially
    env.newPropertyUInt("agent_count", params["L"] * params["L"])
    env.newPropertyUInt("cooperators", 0)
    env.newPropertyUInt("defectors", 0)
    env.newPropertyFloat("avg_C", 0.0)
    env.newPropertyFloat("std_C", 0.0)
    # Add more environment properties for other reporters if needed

def define_messages(model):
    """Defines messages for agent communication."""
    # Message for sharing state needed for interaction
    message = model.newMessageBruteForce("agent_state_message")
    message.newVariableID("id") # Agent ID is type ID_T, typically int/uint
    message.newVariableUInt("strategy")
    message.newVariableFloat("C")
    # No need to pass utility in message for this adapted logic

def define_agents(model):
    """Defines the 'CulturalAgent' agent type."""
    agent = model.newAgent("CulturalAgent")

    # Agent variables matching Mesa agent attributes
    agent.newVariableUInt("strategy")      # 0 (Defect) or 1 (Cooperate)
    agent.newVariableFloat("C")            # Cultural value [0, 1]
    agent.newVariableUInt("next_strategy") # Buffer for strategy update
    agent.newVariableFloat("next_C")       # Buffer for C update

    # Add a variable to pass agent count info to agent functions if needed
    # This is a slight hack, updated each step by stepfn


    # Agent functions (linked to C++/CUDA code)
    fn = agent.newRTCFunction("output_state", output_state)
    fn.setMessageOutput("agent_state_message")

    fn = agent.newRTCFunction("calculate_utility_and_update", calculate_utility_and_update)
    fn.setMessageInput("agent_state_message")

    fn = agent.newRTCFunction("mutate", mutate)
    # No message input/output for mutate

    fn = agent.newRTCFunction("advance", advance)
    # No message input/output for advance

def define_execution_order(model):
    """Defines the layers and execution order of agent functions."""
    # Layer 1: Agents output their current state for others to read
    layer1 = model.newLayer()
    layer1.addAgentFunction("CulturalAgent", "output_state")

    # Layer 2: Agents read messages, interact (randomly), decide next state
    layer2 = model.newLayer()
    layer2.addAgentFunction("CulturalAgent", "calculate_utility_and_update")

    # Layer 3: Apply mutations to the decided next state
    layer3 = model.newLayer()
    layer3.addAgentFunction("CulturalAgent", "mutate")

    # Layer 4: Advance state (update strategy/C from next_strategy/next_C)
    layer4 = model.newLayer()
    layer4.addAgentFunction("CulturalAgent", "advance")

    # Add Host Functions
#    model.addInitFunction(InitFunction())
#    model.addStepFunction(StepFunction())
    # Add ExitFunction if needed for final calculations/logging

# --- Host Functions (Run on CPU) ---

class InitFunction(pyflamegpu.HostFunction):
    """Host function run once at the beginning of the simulation."""
    # Accept params dictionary during initialization
    def __init__(self, params):
        super().__init__() # Important: Call parent constructor
        self.params = params
    def run(self, FLAMEGPU):
        # Access parameters directly from the stored dictionary
        b = self.params["b"] # Still useful to get 'b' for payoff matrix
        initial_coop_ratio = self.params["initial_coop_ratio"]
        C_dist = self.params["C_dist"]
        mu = self.params["mu"]
        sigma = self.params["sigma"]
        L = self.params["L"] # Get L from params
        agent_count = L * L  # Calculate agent_count here
        # Update the environment's agent_count property if it wasn't set correctly before
        # (It should be set in define_environment, but this ensures consistency)
        FLAMEGPU.environment.setPropertyUInt("agent_count", agent_count)
        # Initialize Payoff Matrix in environment
        payoff = FLAMEGPU.environment.getMacroPropertyFloat("payoff")
        payoff[0][0] = 0.0  # D vs D
        payoff[0][1] = b    # D vs C
        payoff[1][0] = 0.0  # C vs D
        payoff[1][1] = 1.0  # C vs C
        print(f"Initializing {agent_count} agents (L={L})...")
        print(f"  C_dist='{C_dist}', mu={mu}, sigma={sigma}, initial_coop_ratio={initial_coop_ratio}")
        # Create agents
        agent_pop = FLAMEGPU.agent("CulturalAgent") # Get agent population reference
        for i in range(agent_count):
            # Use agent_pop.newAgent() for potentially better performance
            agent = agent_pop.newAgent()
            # Initial Strategy
            strategy = 1 if random.random() < initial_coop_ratio else 0
            agent.setVariableUInt("strategy", strategy)
            agent.setVariableUInt("next_strategy", strategy)
            # Initial Culture (C) based on distribution - using self.params directly
            C_value = 0.0
            if C_dist == "uniform":
                C_value = random.uniform(0, 1)
            elif C_dist == "bimodal":
                 C_value = 0.0 if random.random() < (1.0 - mu) else 1.0
            elif C_dist == "normal":
                 # Use numpy's random generator for potentially better control if needed
                 # Or stick to Python's random if sufficient
                 C_value = np.clip(random.normalvariate(mu, sigma), 0, 1)
            elif C_dist == "fixed":
                 C_value = mu
            else:
                 print(f"Warning: Unknown C_dist '{C_dist}'. Defaulting to uniform.")
                 C_value = random.uniform(0, 1)
            agent.setVariableFloat("C", C_value)
            agent.setVariableFloat("next_C", C_value)


class StepFunction(pyflamegpu.HostFunction):
     """Host function run after each GPU step."""
     # No __init__ needed unless it also needs params
     def run(self, FLAMEGPU):
        current_agent_count = FLAMEGPU.agent("CulturalAgent").count()
        # Update the ENVIRONMENT property with the current count
        FLAMEGPU.environment.setPropertyUInt("agent_count", current_agent_count) # <<< CORRECT WAY

        if current_agent_count > 0:
             coop_count = FLAMEGPU.agent("CulturalAgent").countUInt("strategy", 1)
             defect_count = current_agent_count - coop_count
             FLAMEGPU.environment.setPropertyUInt("cooperators", coop_count)
             FLAMEGPU.environment.setPropertyUInt("defectors", defect_count)
             # Calculate C stats (handle potential errors/lack of direct methods)
             try:
                 c_values = FLAMEGPU.agent("CulturalAgent").getVariableFloat("C")
                 if c_values:
                     avg_C = np.mean(c_values)
                     std_C = np.std(c_values)
                 else:
                    avg_C = 0.0
                    std_C = 0.0
                 FLAMEGPU.environment.setPropertyFloat("avg_C", avg_C)
                 FLAMEGPU.environment.setPropertyFloat("std_C", std_C)
             except Exception as e:
                 # print(f"Warning: Could not calculate C stats: {e}")
                 FLAMEGPU.environment.setPropertyFloat("avg_C", 0.0)
                 FLAMEGPU.environment.setPropertyFloat("std_C", 0.0)
        else:
             # Set defaults if no agents
             FLAMEGPU.environment.setPropertyUInt("cooperators", 0)
             FLAMEGPU.environment.setPropertyUInt("defectors", 0)
             FLAMEGPU.environment.setPropertyFloat("avg_C", 0.0)
             FLAMEGPU.environment.setPropertyFloat("std_C", 0.0)

# --- Logging, Output, Runs Configuration ---

def define_logs(model):
    """Defines what data to log at each step."""
    log_cfg = pyflamegpu.StepLoggingConfig(model)
    log_cfg.setFrequency(1) # Log every step

    # Log environment properties calculated in StepFunction
    log_cfg.logEnvironment("cooperators")
    log_cfg.logEnvironment("defectors")
    log_cfg.logEnvironment("avg_C")
    log_cfg.logEnvironment("std_C")
    # Add other environment properties if needed

    # Agent logging can be very verbose, usually avoided unless debugging
    # log_cfg.logAgent("CulturalAgent")

    return log_cfg

def define_output(ensemble):
    """Configures simulation output settings."""
    ensemble.Config().out_directory = "results_flamegpu"
    ensemble.Config().out_format = "json" # Or "csv"
    ensemble.Config().timing = True
    ensemble.Config().truncate_log_files = True
    # Use Fast error level for performance, Normal or Verbose for debugging
    ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Fast
    # Specify GPU device(s)
    ensemble.Config().devices = pyflamegpu.IntSet([0]) # Use first GPU

def define_run_plans(model, params=DEFAULT_PARAMS, num_runs=1):
    """Creates RunPlanVector for simulations."""
    run_plans = pyflamegpu.RunPlanVector(model, num_runs)
    run_plans.setSteps(params["steps"])
    run_plans.setRandomSimulationSeed(params["seed"], 1)
    # Set only the environment properties needed by GPU agents or logging
    run_plans.setPropertyFloat("b", params["b"])
    run_plans.setPropertyFloat("K", params["K"])
    run_plans.setPropertyFloat("K_C", params["K_C"])
    run_plans.setPropertyFloat("p_update_C", params["p_update_C"])
    run_plans.setPropertyFloat("p_mut_culture", params["p_mut_culture"])
    run_plans.setPropertyFloat("p_mut_strategy", params["p_mut_strategy"])
    # REMOVED: setProperty for C_dist, mu, sigma, initial_coop_ratio
    # --- Parameter Sweeping ---
    # Sweeping now only works directly on the numeric env properties.
    # To sweep C_dist, mu, etc., you would need to create multiple
    # RunPlanVectors or modify the 'params' dict for each run manually
    # before creating its RunPlan.
    # Example: Sweep 'b'
    # if num_runs > 1:
    #     run_plans.setPropertyLerpRangeFloat("b", 1.0, 2.0)
    return run_plans




# --- Main Simulation Execution ---
def run_simulation(params=DEFAULT_PARAMS, num_runs=1):
    """Sets up and runs the FLAMEGPU simulation."""
    print("1. Creating Model Description...")
    model = create_model()
    print("2. Defining Environment...")
    # Pass params for initial numeric env setup
    define_environment(model, params)
    print("3. Defining Messages...")
    define_messages(model)
    print("4. Defining Agents...")
    define_agents(model)
    print("5. Defining Execution Order (Layers)...")
    # Define layers only, host functions added separately
    define_execution_order(model)
    print("6. Defining Host Functions...")
    # Instantiate InitFunction with params and add it
    init_func_instance = InitFunction(params)
    model.addInitFunction(init_func_instance)
    # Instantiate and add StepFunction
    step_func_instance = StepFunction()
    model.addStepFunction(step_func_instance)
    # Add ExitFunction if needed: model.addExitFunction(...)
    print("7. Defining Run Plans...")
    run_plans = define_run_plans(model, params, num_runs)
    print("8. Defining Logging...")
    log_config = define_logs(model)
    print("9. Initializing CUDA Ensemble...")
    ensemble_sim = pyflamegpu.CUDAEnsemble(model)
    print("10. Configuring Output...")
    define_output(ensemble_sim)
    print("11. Setting Logging Configuration...")
    ensemble_sim.setStepLog(log_config)
    print(f"--- Starting FLAMEGPU Simulation ({num_runs} runs, {params['steps']} steps) ---")
    start_time = time.time()
    ensemble_sim.simulate(run_plans)
    end_time = time.time()
    print(f"--- Simulation Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Cleanup (optional, allows re-running in same script)
    # pyflamegpu.cleanup()

    # You can access logs programmatically after the run if needed
    # logs = ensemble_sim.getStepLog()


if __name__ == "__main__":
    print("Running single simulation with default parameters...")
    run_simulation(params=DEFAULT_PARAMS, num_runs=1)
    # Example of manually changing a parameter handled by InitFunction for a second run
    print("\nRunning simulation with different C_dist (fixed)...")
    fixed_params = DEFAULT_PARAMS.copy()
    fixed_params["C_dist"] = "fixed"
    fixed_params["mu"] = 0.2 # Set the fixed value using mu
    fixed_params["seed"] = 5678 # Change seed for comparison
    run_simulation(params=fixed_params, num_runs=1)

    # --- Example Parameter Sweep ---
    # print("\nRunning parameter sweep example...")
    # sweep_params = DEFAULT_PARAMS.copy()
    # # Define number of runs for the sweep
    # sweep_runs = 5
    # # Create a new model and run plans specifically for the sweep
    # sweep_model = create_model("CulturalGameSweep")
    # define_environment(sweep_model, sweep_params)
    # define_messages(sweep_model)
    # define_agents(sweep_model)
    # define_execution_order(sweep_model)
    # # Create run plans for the sweep, varying parameter 'b'
    # sweep_run_plans = define_run_plans(sweep_model, sweep_params, sweep_runs)
    # sweep_run_plans.setPropertyLerpRangeFloat("b", 1.0, 2.0) # Sweep b from 1.0 to 2.0
    #
    # sweep_log_config = define_logs(sweep_model)
    # sweep_ensemble = pyflamegpu.CUDAEnsemble(sweep_model)
    # define_output(sweep_ensemble)
    # sweep_ensemble.setStepLog(sweep_log_config)
    #
    # print(f"--- Starting FLAMEGPU Sweep (Parameter 'b', {sweep_runs} runs) ---")
    # start_time = time.time()
    # sweep_ensemble.simulate(sweep_run_plans)
    # end_time = time.time()
    # print(f"--- Sweep Finished ---")
    # print(f"Total execution time: {end_time - start_time:.2f} seconds")

    print("\nSimulation complete. Check the 'results_flamegpu' directory.")
