# agent_functions.py (或者放在主脚本)

import pyflamegpu
from pyflamegpu import * # Easier access to FLAMEGPU object methods
import math # For exp function in Fermi rule



# Assume MAX_NEIGHBORS is defined globally (e.g., MAX_NEIGHBORS = 8)
MAX_NEIGHBORS = 8 # Ensure this is defined/imported

# -------------------------------------
# Agent Function 1: Output State
# -------------------------------------
# Purpose: Broadcasts the agent's current strategy and C value to neighbors.
# Corresponds to: Preparation for utility calculation.

@pyflamegpu.agent_function
def output_state(message_out: pyflamegpu.MessageSpatial2D.Write, agent: pyflamegpu.Agent):
    """Agent function to output current state (strategy, C) via StateMessage."""
    agent_id = agent.getID()
    strategy = agent.getVariableInt32("strategy")
    C_value = agent.getVariableFloat("C")

    # Set message variables
    message_out.setVariableID("id", agent_id)
    message_out.setVariableInt32("strategy", strategy)
    message_out.setVariableFloat("C", C_value)

    # The location is automatically handled by Spatial messages

    return pyflamegpu.ALIVE # Agent continues to exist


# -------------------------------------
# Agent Function 2: Calculate Utility
# -------------------------------------
# Purpose: Reads state messages from neighbors and calculates the agent's total utility.
# Corresponds to: Mesa Agent's calculate_utility() method.

@pyflamegpu.agent_function
def calculate_utility(message_in: pyflamegpu.MessageSpatial2D.Read, agent: pyflamegpu.Agent):
    """Agent function to calculate utility based on neighbor interactions."""

    # Get my own state
    my_strategy = agent.getVariableInt32("strategy")
    my_C = agent.getVariableFloat("C")
    my_id = agent.getID() # For debugging or skipping self-messaging if needed

    # Get payoff parameters from environment
    T = FLAMEGPU.environment.getPropertyFloat("T_PAYOFF") # Temptation (b)
    R = FLAMEGPU.environment.getPropertyFloat("R_PAYOFF") # Reward
    S = FLAMEGPU.environment.getPropertyFloat("S_PAYOFF") # Sucker
    P = FLAMEGPU.environment.getPropertyFloat("P_PAYOFF") # Punishment

    total_utility = 0.0
    neighbor_count_actual = 0 # Keep track of actual neighbors processed

    # Iterate through messages from neighbors within the radius
    for neighbor_message in message_in:
        # Optional: Skip message from self if radius captures self (unlikely with radius 1.5 on grid)
        neighbor_id = neighbor_message.getVariableID("id")
        if neighbor_id == my_id:
             continue

        neighbor_strategy = neighbor_message.getVariableInt32("strategy")
        # neighbor_C = neighbor_message.getVariableFloat("C") # Not needed for this agent's utility calc

        # Calculate payoffs based on the interaction (my_strategy, neighbor_strategy)
        my_payoff = 0.0
        neighbor_payoff = 0.0

        if my_strategy == 1 and neighbor_strategy == 1: # C-C
            my_payoff = R
            neighbor_payoff = R
        elif my_strategy == 1 and neighbor_strategy == 0: # C-D
            my_payoff = S
            neighbor_payoff = T # Neighbor gets Temptation
        elif my_strategy == 0 and neighbor_strategy == 1: # D-C
            my_payoff = T # I get Temptation
            neighbor_payoff = S
        elif my_strategy == 0 and neighbor_strategy == 0: # D-D
            my_payoff = P
            neighbor_payoff = P
        else:
            # Handle potential invalid strategy values?
            # print(f"Warning: Agent {my_id} encountered invalid strategies ({my_strategy}, {neighbor_strategy})")
            pass # Assign 0 payoff or handle as error

        # Calculate the cultural utility contribution from this neighbor
        utility_contribution = (1.0 - my_C) * my_payoff + my_C * neighbor_payoff
        total_utility += utility_contribution
        neighbor_count_actual += 1

    # Store the calculated total utility
    agent.setVariableFloat("utility", total_utility)
    # We could also store the actual neighbor count if needed elsewhere,
    # but the decision function will get its own count from Utility messages.
    # agent.setVariableUInt32("neighbor_count", neighbor_count_actual) # Not strictly needed here

    # Initialize next_strategy and next_C for the upcoming decision steps
    # This mirrors Mesa's default assumption before decision functions run
    agent.setVariableInt32("next_strategy", my_strategy)
    agent.setVariableFloat("next_C", my_C)


    return pyflamegpu.ALIVE






# -------------------------------------
# Agent Function 3: Output Utility
# -------------------------------------
# Purpose: Broadcasts the agent's calculated utility, strategy, and C value.
# Corresponds to: Preparation for decision making (strategy/culture update).

@pyflamegpu.agent_function
def output_utility(message_out: pyflamegpu.MessageSpatial2D.Write, agent: pyflamegpu.Agent):
    """Agent function to output calculated utility and state via UtilityMessage."""
    agent_id = agent.getID()
    strategy = agent.getVariableInt32("strategy")
    C_value = agent.getVariableFloat("C")
    utility = agent.getVariableFloat("utility") # Use the utility calculated in the previous step

    # Set message variables
    message_out.setVariableID("id", agent_id)
    message_out.setVariableInt32("strategy", strategy)
    message_out.setVariableFloat("C", C_value)
    message_out.setVariableFloat("utility", utility)

    # Location handled by Spatial messages

    return pyflamegpu.ALIVE # Agent continues to exist


# -------------------------------------
# Agent Function 4: Process Neighbor Utility
# -------------------------------------
# Purpose: Reads utility messages from neighbors and stores their relevant info
#          (strategy, C, utility) in the agent's internal arrays for later decision.
# Corresponds to: Gathering information needed for decide_strategy_update() and decide_culture_update().

@pyflamegpu.agent_function
def process_neighbor_utility(message_in: pyflamegpu.MessageSpatial2D.Read, agent: pyflamegpu.Agent):
    """Agent function to read neighbor utilities and store them."""

    my_id = agent.getID()
    neighbor_count = 0

    # Iterate through messages from neighbors
    for neighbor_message in message_in:
        # Optional: Skip message from self
        neighbor_id = neighbor_message.getVariableID("id")
        if neighbor_id == my_id:
            continue

        # Check if we have space to store this neighbor's info
        if neighbor_count < MAX_NEIGHBORS:
            # Get neighbor data from the message
            n_strategy = neighbor_message.getVariableInt32("strategy")
            n_C = neighbor_message.getVariableFloat("C")
            n_utility = neighbor_message.getVariableFloat("utility")

            # Store the data in the agent's array variables at the current index
            agent.setVariableInt32Array("neighbor_strategies", neighbor_count, n_strategy)
            agent.setVariableFloatArray("neighbor_Cs", neighbor_count, n_C)
            agent.setVariableFloatArray("neighbor_utilities", neighbor_count, n_utility)

            neighbor_count += 1 # Increment the count of stored neighbors
        else:
            # Optional: Warn if more neighbors than storage space
            # print(f"Warning: Agent {my_id} encountered more than {MAX_NEIGHBORS} neighbors. Ignoring excess.")
            pass # Ignore neighbors beyond the array capacity

    # Store the final count of neighbors whose data was stored
    agent.setVariableUInt32("neighbor_count", neighbor_count)

    return pyflamegpu.ALIVE



@pyflamegpu.agent_function
def decide_strategy_update(agent: pyflamegpu.Agent):
    """Agent function to decide the next strategy using Fermi rule."""

    my_utility = agent.getVariableFloat("utility")
    neighbor_count = agent.getVariableUInt32("neighbor_count")

    # Only proceed if there are neighbors to compare with
    if neighbor_count > 0:
        # Get noise parameter K from environment
        K = FLAMEGPU.environment.getPropertyFloat("K")

        # Randomly select one neighbor from the stored list
        # FLAMEGPU.random.uniformUInt(min, max) generates in [min, max]
        # FLAMEGPU.random.uniformInt(min, max) generates in [min, max] -> Check docs
        # Let's assume uniformInt(0, count) is correct for index [0, count-1] if count is uint32
        # Safer: generate [0, count-1] using uniformUInt
        # rand_idx = FLAMEGPU.random.uniformUInt(0, neighbor_count - 1) # Generates in [0, count-1]

        # Using uniformInt(min, max) which generates an integer between min and max (inclusive).
        # So we need index 0 to neighbor_count - 1.
        if neighbor_count == 1:
             rand_idx = 0
        else:
             # Generate random index from 0 to neighbor_count-1 inclusive
             rand_idx = FLAMEGPU.random.uniformInt(0, neighbor_count - 1)


        # Get the chosen neighbor's data from the agent's arrays
        neighbor_strategy = agent.getVariableInt32Array("neighbor_strategies", rand_idx)
        neighbor_utility = agent.getVariableFloatArray("neighbor_utilities", rand_idx)

        # Calculate utility difference
        delta_utility = neighbor_utility - my_utility

        prob_adopt = 0.0
        # Apply Fermi rule
        if K < 1e-9: # Handle K close to zero (deterministic choice)
            prob_adopt = 1.0 if delta_utility > 0 else 0.0
        else:
            argument = -delta_utility / K
            # Overflow protection for exp()
            if argument > 700: # exp(700) is already huge
                prob_adopt = 0.0 # Very unlikely to adopt if neighbor is much worse
            elif argument < -700: # exp(-700) is effectively 0
                prob_adopt = 1.0 # Very likely to adopt if neighbor is much better
            else:
                try:
                    prob_adopt = 1.0 / (1.0 + math.exp(argument))
                except OverflowError:
                    # Fallback in case of unexpected overflow
                    prob_adopt = 0.0 if argument > 0 else 1.0


        # Decide whether to adopt based on probability
        if FLAMEGPU.random.uniformFloat() < prob_adopt:
            # Adopt the neighbor's strategy for the next step
            agent.setVariableInt32("next_strategy", neighbor_strategy)
        # Else: Keep the current strategy (already set in next_strategy during calculate_utility)

    # If neighbor_count is 0, next_strategy remains the current strategy

    return pyflamegpu.ALIVE


# -------------------------------------
# Agent Function 6: Decide Culture Update (Fermi Rule)
# -------------------------------------
# Purpose: With probability p_update_C, compares utility with a randomly chosen
#          neighbor and decides whether to adopt the neighbor's C value based
#          on the Fermi rule (using K_C). Updates next_C.
# Corresponds to: Mesa Agent's decide_culture_update() method.

@pyflamegpu.agent_function
def decide_culture_update(agent: pyflamegpu.Agent):
    """Agent function to potentially decide the next C value using Fermi rule."""

    p_update_C = FLAMEGPU.environment.getPropertyFloat("p_update_C")

    # Only attempt update with a certain probability
    if FLAMEGPU.random.uniformFloat() < p_update_C:
        my_utility = agent.getVariableFloat("utility")
        neighbor_count = agent.getVariableUInt32("neighbor_count")

        # Only proceed if there are neighbors to compare with
        if neighbor_count > 0:
            # Get cultural noise parameter K_C from environment
            K_C = FLAMEGPU.environment.getPropertyFloat("K_C")

            # Randomly select one neighbor (same logic as strategy update)
            if neighbor_count == 1:
                 rand_idx = 0
            else:
                 rand_idx = FLAMEGPU.random.uniformInt(0, neighbor_count - 1)


            # Get the chosen neighbor's C value and utility
            neighbor_C = agent.getVariableFloatArray("neighbor_Cs", rand_idx)
            neighbor_utility = agent.getVariableFloatArray("neighbor_utilities", rand_idx)

            # Calculate utility difference
            delta_utility = neighbor_utility - my_utility

            prob_adopt_culture = 0.0
            # Apply Fermi rule using K_C
            if K_C < 1e-9: # Handle K_C close to zero
                prob_adopt_culture = 1.0 if delta_utility > 0 else 0.0
            else:
                argument = -delta_utility / K_C
                # Overflow protection
                if argument > 700:
                    prob_adopt_culture = 0.0
                elif argument < -700:
                    prob_adopt_culture = 1.0
                else:
                    try:
                        prob_adopt_culture = 1.0 / (1.0 + math.exp(argument))
                    except OverflowError:
                        prob_adopt_culture = 0.0 if argument > 0 else 1.0


            # Decide whether to adopt based on probability
            if FLAMEGPU.random.uniformFloat() < prob_adopt_culture:
                # Adopt the neighbor's C value for the next step
                agent.setVariableFloat("next_C", neighbor_C)
            # Else: Keep the current C value (already set in next_C during calculate_utility)

    # If neighbor_count is 0 or random check fails, next_C remains the current C

    return pyflamegpu.ALIVE




@pyflamegpu.agent_function
def mutate(agent: pyflamegpu.Agent):
    """Agent function to apply random mutations to next_strategy and next_C."""

    # Get mutation rates from environment
    mutation_rate_strategy = FLAMEGPU.environment.getPropertyFloat("mutation_rate_strategy")
    mutation_rate_C = FLAMEGPU.environment.getPropertyFloat("mutation_rate_C")

    # --- Strategy Mutation ---
    if FLAMEGPU.random.uniformFloat() < mutation_rate_strategy:
        # Flip the current next_strategy
        current_next_strategy = agent.getVariableInt32("next_strategy")
        mutated_strategy = 1 - current_next_strategy # Flip 0 to 1, 1 to 0
        agent.setVariableInt32("next_strategy", mutated_strategy)

    # --- Culture (C value) Mutation ---
    if FLAMEGPU.random.uniformFloat() < mutation_rate_C:
        # Assign a new random C value between 0.0 and 1.0
        mutated_C = FLAMEGPU.random.uniformFloat() # Generates float in [0.0, 1.0)
        agent.setVariableFloat("next_C", mutated_C)

    return pyflamegpu.ALIVE


# -------------------------------------
# Agent Function 8: Advance State
# -------------------------------------
# Purpose: Updates the agent's main state variables (strategy, C) with the
#          values determined during the decision and mutation phases (next_strategy, next_C).
# Corresponds to: Mesa Agent's advance() method.

@pyflamegpu.agent_function
def advance(agent: pyflamegpu.Agent):
    """Agent function to update the agent's state for the next step."""

    # Get the finalized next state values
    final_next_strategy = agent.getVariableInt32("next_strategy")
    final_next_C = agent.getVariableFloat("next_C")

    # Update the main state variables
    agent.setVariableInt32("strategy", final_next_strategy)
    agent.setVariableFloat("C", final_next_C)

    # Reset utility for the next calculation cycle (optional but good practice)
    # agent.setVariableFloat("utility", 0.0)
    # Reset neighbor count (optional, as it's recalculated anyway)
    # agent.setVariableUInt32("neighbor_count", 0)

    return pyflamegpu.ALIVE
