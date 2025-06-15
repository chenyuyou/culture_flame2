# cultural_game_mesa.py
import mesa
import numpy as np
import random
import math
import os
import json

class CulturalAgent(mesa.Agent):
    """
    A cultural agent with a strategy (Cooperate/Defect) and a cultural value (C).
    """
    def __init__(self, unique_id, model, pos, initial_strategy, initial_C):
        super().__init__(unique_id, model)
#        self.pos = pos  # (x, y) tuple
        self.strategy = initial_strategy  # 0 for Defect, 1 for Cooperate
        self.C = initial_C  # Cultural value, float between 0 and 1
        self.next_strategy = initial_strategy
        self.next_C = initial_C
        self.current_utility = 0.0

        # For local metrics (similar to FLAMEGPU2's calculate_local_metrics_func)
        self.same_strategy_neighbors = 0
        self.same_type_neighbors = 0
        self.total_neighbors_count = 0
        self.is_boundary_agent = 0
        self.is_boundary_cooperator = 0
        self.is_bulk_cooperator = 0
        self.agent_type = 0 # 1 for A (C < 0.5), 2 for B (C >= 0.5)

    def calculate_utility(self):
        """
        Calculates the agent's utility based on interactions with neighbors.
        Uses the payoff matrix and cultural value C.
        """
        total_utility = 0.0
        payoff = self.model.payoff_matrix
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        for neighbor in neighbors:
            # Payoff I get from interacting with neighbor
            my_payoff = payoff[self.strategy][neighbor.strategy]
            # Payoff neighbor gets from interacting with me
            neighbor_payoff = payoff[neighbor.strategy][self.strategy]

            # Accumulate weighted utility based on my C value
            total_utility += (1.0 - self.C) * my_payoff + self.C * neighbor_payoff
        
        self.current_utility = total_utility

    def decide_updates(self):
        """
        Decides the next strategy and cultural value based on Fermi rule
        and comparison with a random neighbor.
        """
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        
        if not neighbors:
            self.next_strategy = self.strategy
            self.next_C = self.C
            return
        # Select one random neighbor
        selected_neighbor = self.random.choice(neighbors)
        # Strategy Update (Fermi Rule)
        delta_utility = selected_neighbor.current_utility - self.current_utility
        
        prob_adopt_strategy = 0.0
        if self.model.K > 1e-9: # Avoid division by zero or very small numbers
            # Limit the argument to prevent overflow/underflow in exp()
            # Common range is [-70, 70] or similar, depending on float precision
            # Using numpy's clip for convenience, or math.fmax/fmin
            argument = -delta_utility / self.model.K
            argument = np.clip(argument, -700.0, 700.0) # Using a slightly larger range for robustness, though 70 is usually enough for float32
                                                        # Python's float (double precision) can handle larger, but 700 is safe.
            prob_adopt_strategy = 1.0 / (1.0 + math.exp(argument))
        else: # K is very small, deterministic update
            prob_adopt_strategy = 1.0 if delta_utility > 0 else 0.0
        if self.random.random() < prob_adopt_strategy:
            self.next_strategy = selected_neighbor.strategy
        else:
            self.next_strategy = self.strategy
        # Cultural Update (Fermi Rule, conditional)
        if self.random.random() < self.model.p_update_C:
            prob_adopt_culture = 0.0
            if self.model.K_C > 1e-9: # Avoid division by zero or very small numbers
                # Limit the argument to prevent overflow/underflow in exp()
                argument_C = -delta_utility / self.model.K_C
                argument_C = np.clip(argument_C, -700.0, 700.0) # Same range as above
                prob_adopt_culture = 1.0 / (1.0 + math.exp(argument_C))
            else: # K_C is very small, deterministic update
                prob_adopt_culture = 1.0 if delta_utility > 0 else 0.0
            if self.random.random() < prob_adopt_culture:
                self.next_C = selected_neighbor.C
                self.next_C = np.clip(self.next_C, 0.0, 1.0) # Ensure C stays within [0, 1]
            else:
                self.next_C = self.C
        else:
            self.next_C = self.C # No cultural update attempt

    def mutate(self):
        """
        Applies random mutations to strategy and cultural value.
        """
        if self.random.random() < self.model.p_mut_strategy:
            self.next_strategy = 1 - self.next_strategy # Flip strategy
        
        if self.random.random() < self.model.p_mut_culture:
            self.next_C = self.random.random() # Random new C value
            self.next_C = np.clip(self.next_C, 0.0, 1.0)

    def advance(self):
        """
        Updates the agent's state to the 'next' state.
        """
        self.strategy = self.next_strategy
        self.C = self.next_C

    def calculate_local_metrics(self):
        """
        Calculates local metrics like same-strategy/type neighbors and boundary status.
        This is for reporting, not directly for agent behavior updates.
        """
        self.same_strategy_neighbors = 0
        self.same_type_neighbors = 0
        self.total_neighbors_count = 0
        self.is_boundary_agent = 0
        self.is_boundary_cooperator = 0
        self.is_bulk_cooperator = 0

        C_threshold = 0.5 # For A/B typing
        self.agent_type = 1 if self.C < C_threshold else 2 # 1 for A, 2 for B

        has_different_neighbor = False
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        
        for neighbor in neighbors:
            self.total_neighbors_count += 1
            if neighbor.strategy == self.strategy:
                self.same_strategy_neighbors += 1
            
            neighbor_type = 1 if neighbor.C < C_threshold else 2
            if neighbor_type == self.agent_type:
                self.same_type_neighbors += 1
            else:
                has_different_neighbor = True
        
        self.is_boundary_agent = 1 if has_different_neighbor else 0

        if self.strategy == 1: # If agent is a cooperator
            if self.is_boundary_agent:
                self.is_boundary_cooperator = 1
            else:
                self.is_bulk_cooperator = 1

    def step(self):
        """
        Defines the sequence of actions for the agent in one simulation step.
        """
        # Step 1: Calculate utility based on current state (t-1)
        self.calculate_utility()
        # Step 2: Decide next strategy and C based on utility comparison
        self.decide_updates()
        # Step 3: Apply mutations
        self.mutate()
        # Step 4: Calculate local metrics (for reporting, uses current state)
        self.calculate_local_metrics()
        # Note: The actual state update (advance) happens in the model's step
        # after all agents have decided their next state. This mimics FLAMEGPU's staged execution.


class CulturalGameModel(mesa.Model):
    """
    The Cultural Game model.
    """
    def __init__(self, L, initial_coop_ratio, b, K, K_C, p_update_C, p_mut_culture, p_mut_strategy,
                 C_dist, mu, sigma, seed=None):
        super().__init__(seed=seed) 
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.random.seed(seed) # Mesa's internal random number generator

        self.L = L
        self.num_agents = L * L
        self.initial_coop_ratio = initial_coop_ratio
        self.b = b
        self.K = K
        self.K_C = K_C
        self.p_update_C = p_update_C
        self.p_mut_culture = p_mut_culture
        self.p_mut_strategy = p_mut_strategy
        self.C_dist = C_dist
        self.mu = mu
        self.sigma = sigma

        self.grid = mesa.space.SingleGrid(L, L, torus=True) # Toroidal grid for wrap-around
        self.schedule = mesa.time.RandomActivation(self) # Agents activate in random order

        # Payoff matrix for Prisoner's Dilemma (R=1, S=0, T=b, P=0)
        # Row player's payoff: payoff[my_strategy][opponent_strategy]
        # C=1, D=0
        #       C   D
        # C   [[1,  0],
        # D    [b,  0]]
        self.payoff_matrix = np.array([
            [0.0, self.b],  # 我的策略 0 (D): (D,D) 收益 0, (D,C) 收益 b
            [0.0, 1.0]   # 我的策略 1 (C): (C,D) 收益 0, (C,C) 收益 1
        ])

        self.running = True # For Mesa's batch runner

        # Create agents
        agent_id = 0
        for x in range(self.L):
            for y in range(self.L):
                initial_strategy = 1 if self.random.random() < self.initial_coop_ratio else 0

                C_value = 0.0
                if self.C_dist == "uniform":
                    C_value = self.random.random()
                elif self.C_dist == "bimodal":
                    C_value = 0.0 if self.random.random() < (1.0 - self.mu) else 1.0
                elif self.C_dist == "normal":
                    C_value = np.clip(self.random.normal(self.mu, self.sigma), 0, 1)
                elif self.C_dist == "fixed":
                    C_value = self.mu
                else:
                    C_value = self.random.random() # Default to uniform

                a = CulturalAgent(agent_id, self, (x, y), initial_strategy, C_value)
                self.schedule.add(a)
                self.grid.place_agent(a, (x, y))
                agent_id += 1

        # Data Collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "CooperationRate": lambda m: sum(a.strategy for a in m.schedule.agents) / m.num_agents,
                "AvgC": lambda m: sum(a.C for a in m.schedule.agents) / m.num_agents,
                "StdC": lambda m: np.std([a.C for a in m.schedule.agents]),
                "SegregationIndexStrategy": self.get_segregation_index_strategy,
                "SegregationIndexType": self.get_segregation_index_type,
                "BoundaryFraction": self.get_boundary_fraction,
                "BoundaryCoopRate": self.get_boundary_coop_rate,
                "BulkCoopRate": self.get_bulk_coop_rate,
            },
            agent_reporters={
                "strategy": "strategy",
                "C": "C",
                "ix": lambda a: a.pos[0],
                "iy": lambda a: a.pos[1],
                "utility": "current_utility",
                "same_strategy_neighbors": "same_strategy_neighbors",
                "same_type_neighbors": "same_type_neighbors",
                "total_neighbors_count": "total_neighbors_count",
                "is_boundary_agent": "is_boundary_agent",
                "is_boundary_cooperator": "is_boundary_cooperator",
                "is_bulk_cooperator": "is_bulk_cooperator",
                "agent_type": "agent_type"
            }
        )
        self.datacollector.collect(self) # Collect initial state

    def get_segregation_index_strategy(self): # <-- 签名已正确
        # 将 'model.schedule.agents' 改为 'self.schedule.agents'
        total_same_strategy_neighbors = sum(a.same_strategy_neighbors for a in self.schedule.agents)
        total_neighbors_sum = sum(a.total_neighbors_count for a in self.schedule.agents)
        if total_neighbors_sum > 0:
            return (2.0 * total_same_strategy_neighbors - total_neighbors_sum) / total_neighbors_sum
        return 0.0
    def get_segregation_index_type(self): # <-- 签名已正确
        # 将 'model.schedule.agents' 改为 'self.schedule.agents'
        total_same_type_neighbors = sum(a.same_type_neighbors for a in self.schedule.agents)
        total_neighbors_sum = sum(a.total_neighbors_count for a in self.schedule.agents)
        if total_neighbors_sum > 0:
            return float(total_same_type_neighbors) / total_neighbors_sum
        return 0.0
    def get_boundary_fraction(self): # <-- 签名已正确
        # 将 'model.schedule.agents' 改为 'self.schedule.agents'
        total_boundary_agents = sum(a.is_boundary_agent for a in self.schedule.agents)
        # 将 'model.num_agents' 改为 'self.num_agents'
        return float(total_boundary_agents) / self.num_agents if self.num_agents > 0 else 0.0
    def get_boundary_coop_rate(self): # <-- 签名已正确
        # 将 'model.schedule.agents' 改为 'self.schedule.agents'
        total_boundary_agents = sum(a.is_boundary_agent for a in self.schedule.agents)
        total_boundary_cooperators = sum(a.is_boundary_cooperator for a in self.schedule.agents)
        return float(total_boundary_cooperators) / total_boundary_agents if total_boundary_agents > 0 else 0.0
    def get_bulk_coop_rate(self): # <-- 签名已正确
        # 将 'model.schedule.agents' 改为 'self.schedule.agents'
        total_bulk_agents = sum(1 for a in self.schedule.agents if a.is_boundary_agent == 0)
        total_bulk_cooperators = sum(a.is_bulk_cooperator for a in self.schedule.agents)
        return float(total_bulk_cooperators) / total_bulk_agents if total_bulk_agents > 0 else 0.0


    def get_grid_state_for_snapshot(self):
        """
        Returns a 2D numpy array representing the grid state for plotting.
        Maps agent (strategy, C) to a single integer state.
        """
        grid_states = np.full((self.L, self.L), -1, dtype=int) # -1 for empty/error



        for agent in self.schedule.agents:
            x, y = agent.pos
            strategy = agent.strategy
            C = agent.C

            state = -1
            if 0.0 <= C < 0.5:
                state = 0 if strategy == 0 else 1
            elif 0.5 <= C <= 1.0 :
                state = 2 if strategy == 0 else 3

            
            grid_states[y][x] = state # Note: Mesa grid (x,y) maps to numpy array (row=y, col=x)
        
        return grid_states

    def step(self):
        """
        Advance the model by one step.
        """
        # All agents calculate their next state based on current state
        self.schedule.step() 
        
        # After all agents have decided, update their actual state
        for agent in self.schedule.agents:
            agent.advance()

        self.datacollector.collect(self)

