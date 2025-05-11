# host_functions.py

import pyflamegpu
import random
import numpy as np
import time

class InitFunction(pyflamegpu.HostFunction):
    """Host function run once at the beginning."""

    def __init__(self, params):
        super().__init__()
        # Store the parameters passed during instantiation
        self.params = params

    def run(self, FLAMEGPU):
        # Access parameters directly from the stored params dictionary
        L = self.params["L"]
        agent_count = L * L
        b = self.params["b"] # This is also in environment, but can be accessed here too
        initial_coop_ratio = self.params["initial_coop_ratio"]
        C_dist = self.params["C_dist"]
        mu = self.params["mu"]
        sigma = self.params["sigma"]

        # Set environment properties that are needed by Agent Functions or StepFunction
        FLAMEGPU.environment.setPropertyUInt("agent_count", agent_count)
        FLAMEGPU.environment.setPropertyUInt("GRID_DIM_L", L)
        payoff = FLAMEGPU.environment.getMacroPropertyFloat("payoff")
        # Payoff: [My Strategy][Neighbor Strategy] -> My Payoff
        # P(C,C)=1, P(C,D)=0, P(D,C)=b, P(D,D)=0
        # Row Player (Me), Column Player (Neighbor)
        payoff[1][1] = 1.0  # Me C, Neighbor C -> My Payoff = 1
        payoff[1][0] = 0.0  # Me C, Neighbor D -> My Payoff = 0
        payoff[0][1] = b    # Me D, Neighbor C -> My Payoff = b
        payoff[0][0] = 0.0  # Me D, Neighbor D -> My Payoff = 0

        agents = FLAMEGPU.agent("CulturalAgent")
        # Use the simulation seed for initial agent state randomness
        # Get the simulation seed from the environment (set by RunPlanVector)
        # This requires "random_seed" to be defined as an environment property
        sim_seed = FLAMEGPU.environment.getPropertyUInt("random_seed")
        rng = random.Random(sim_seed)

        for i in range(agent_count):
            agent = agents.newAgent()
            ix = i % L
            iy = i // L
            agent.setVariableInt("ix", ix)
            agent.setVariableInt("iy", iy)

            # Initial strategy (random based on rate)
            strategy = 1 if rng.random() < initial_coop_ratio else 0
            agent.setVariableUInt("strategy", strategy)
            agent.setVariableUInt("next_strategy", strategy)  # Initialize buffer

            # Initial Culture (C) - same logic as before
            C_value = 0.0
            if C_dist == "uniform":
                C_value = rng.uniform(0, 1)
            elif C_dist == "bimodal":
                C_value = 0.0 if rng.random() < (1.0 - mu) else 1.0
            elif C_dist == "normal":
                # Use numpy random for normal distribution, seeded from Python random
                np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
                C_value = np.clip(np_rng.normal(mu, sigma), 0, 1)
            elif C_dist == "fixed":
                C_value = mu
            else:
                C_value = rng.uniform(0, 1)  # Default to uniform if unknown
            agent.setVariableFloat("C", C_value)
            agent.setVariableFloat("next_C", C_value)  # Initialize buffer

            # Initialize the new utility variable
            agent.setVariableFloat("current_utility", 0.0)


class StepFunction(pyflamegpu.HostFunction):
    """Host function run after each GPU step."""

    def run(self, FLAMEGPU):
        agent_count_gpu = FLAMEGPU.agent("CulturalAgent").count()

        avg_coop_rate, std_coop_rate = FLAMEGPU.agent("CulturalAgent").meanStandardDeviationUInt("strategy")
        FLAMEGPU.environment.setPropertyFloat("coop_rate", avg_coop_rate)
        FLAMEGPU.environment.setPropertyFloat("std_coop_rate", std_coop_rate)
        FLAMEGPU.environment.setPropertyFloat("defection_rate", 1.0 - avg_coop_rate)
        FLAMEGPU.environment.setPropertyFloat("std_defection_rate", std_coop_rate)

        avg_C, std_C = FLAMEGPU.agent("CulturalAgent").meanStandardDeviationFloat("C")
        FLAMEGPU.environment.setPropertyFloat("avg_C", avg_C)
        FLAMEGPU.environment.setPropertyFloat("std_C", std_C)

        total_same_strategy_neighbors = FLAMEGPU.agent("CulturalAgent").sumUInt("same_strategy_neighbors")
        total_neighbors_sum = FLAMEGPU.agent("CulturalAgent").sumUInt("total_neighbors_count")

        if total_neighbors_sum > 0:
            seg_index_strategy = (2.0 * total_same_strategy_neighbors - total_neighbors_sum) / total_neighbors_sum
            FLAMEGPU.environment.setPropertyFloat("segregation_index_strategy", seg_index_strategy)
            total_same_type_neighbors = FLAMEGPU.agent("CulturalAgent").sumUInt("same_type_neighbors")
            seg_index_type = float(total_same_type_neighbors) / total_neighbors_sum
            FLAMEGPU.environment.setPropertyFloat("segregation_index_type", seg_index_type)
        else:
            FLAMEGPU.environment.setPropertyFloat("segregation_index_strategy", 0.0)
            FLAMEGPU.environment.setPropertyFloat("segregation_index_type", 0.0)

        total_boundary_agents = FLAMEGPU.agent("CulturalAgent").sumUInt("is_boundary_agent")
        FLAMEGPU.environment.setPropertyUInt("total_boundary_agents", total_boundary_agents)
        boundary_frac = float(total_boundary_agents) / agent_count_gpu if agent_count_gpu > 0 else 0.0
        FLAMEGPU.environment.setPropertyFloat("boundary_fraction", boundary_frac)

        total_boundary_cooperators = FLAMEGPU.agent("CulturalAgent").sumUInt("is_boundary_cooperator")
        total_bulk_cooperators = FLAMEGPU.agent("CulturalAgent").sumUInt("is_bulk_cooperator")
        total_bulk_agents = agent_count_gpu - total_boundary_agents
        FLAMEGPU.environment.setPropertyUInt("total_boundary_cooperators", total_boundary_cooperators)
        FLAMEGPU.environment.setPropertyUInt("total_bulk_cooperators", total_bulk_cooperators)
        FLAMEGPU.environment.setPropertyUInt("total_bulk_agents", total_bulk_agents)

        if total_boundary_agents > 0:
            boundary_coop_rate = float(total_boundary_cooperators) / total_boundary_agents
            FLAMEGPU.environment.setPropertyFloat("boundary_coop_rate", boundary_coop_rate)
        else:
            FLAMEGPU.environment.setPropertyFloat("boundary_coop_rate", 0.0)

        if total_bulk_agents > 0:
            bulk_coop_rate = float(total_bulk_cooperators) / total_bulk_agents
            FLAMEGPU.environment.setPropertyFloat("bulk_coop_rate", bulk_coop_rate)
        else:
            FLAMEGPU.environment.setPropertyFloat("bulk_coop_rate", 0.0)

