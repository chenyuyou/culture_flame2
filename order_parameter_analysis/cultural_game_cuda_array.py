# cultural_game_cuda_array.py

output_state_func = r"""
FLAMEGPU_AGENT_FUNCTION(output_state_func, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    // Output state to the agent's specific grid cell in the agent_state_message
    const int ix = FLAMEGPU->getVariable<int>("ix");
    const int iy = FLAMEGPU->getVariable<int>("iy");
    FLAMEGPU->message_out.setIndex(ix, iy);
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<unsigned int>("strategy", FLAMEGPU->getVariable<unsigned int>("strategy"));
    FLAMEGPU->message_out.setVariable<float>("C", FLAMEGPU->getVariable<float>("C"));
    // We don't need to pass next_strategy/next_C forward here,
    // they are agent variables accessible in later layers.
    return flamegpu::ALIVE;
}
"""

calculate_utility_func = r"""
FLAMEGPU_AGENT_FUNCTION(calculate_utility_func, flamegpu::MessageArray2D, flamegpu::MessageArray2D) {
    // Input message: agent_state_message (Strategy & C from t-1)
    // Output message: utility_message (Calculated utility @ t, plus Strategy & C for adoption)

    // Get own state @ t-1
    // const flamegpu::id_t self_id = FLAMEGPU->getID(); // May not be needed if wrap() guarantees no self-message
    const int self_ix = FLAMEGPU->getVariable<int>("ix");
    const int self_iy = FLAMEGPU->getVariable<int>("iy");
    const unsigned int self_strategy = FLAMEGPU->getVariable<unsigned int>("strategy");
    const float self_C = FLAMEGPU->getVariable<float>("C");
    auto payoff = FLAMEGPU->environment.getMacroProperty<float, 2, 2>("payoff");

    float total_utility = 0.0f;

    // --- Iterate through neighbors using wrap(self_ix, self_iy, 1) ---
    // Get the iterator for the Moore neighborhood (radius 1) around the agent's own position.
    // This automatically handles wrapping and excludes the agent's own cell.
    for (const auto& neighbor_msg : FLAMEGPU->message_in.wrap(self_ix, self_iy, 1)) {
        // Get neighbor's state @ t-1 from the message
        // Note: We no longer need the self_id check because wrap() excludes the origin cell.
        // We also don't need the dx, dy loops or coordinate calculations.

        unsigned int neighbor_strategy = neighbor_msg.getVariable<unsigned int>("strategy");
        // float neighbor_C = neighbor_msg.getVariable<float>("C"); // Not directly needed for payoff calc

        // Calculate payoffs for THIS interaction
        float my_payoff = payoff[self_strategy][neighbor_strategy];
        float neighbor_payoff = payoff[neighbor_strategy][self_strategy]; // Payoff neighbor gets FROM ME

        // Accumulate weighted utility based on MY C value
        total_utility += (1.0f - self_C) * my_payoff + self_C * neighbor_payoff;

        // Important Consideration: If a neighbor cell could contain *multiple* agents/messages,
        // this loop will process *all* of them. The original code with `break` only processed one.
        // If the assumption is strictly one agent per cell for Array2D messaging, this behavior is fine.
    }

    // --- Store calculated utility in an agent variable ---
    FLAMEGPU->setVariable<float>("current_utility", total_utility);


    FLAMEGPU->message_out.setIndex(self_ix, self_iy); // <--- 确保这行代码存在且正确

    // --- Output utility AND state needed for adoption later ---
    // Output to the utility_message at the agent's own location
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getID()); // Use getID() here
    FLAMEGPU->message_out.setVariable<float>("utility", total_utility);
    FLAMEGPU->message_out.setVariable<unsigned int>("strategy", self_strategy); // Pass current strategy
    FLAMEGPU->message_out.setVariable<float>("C", self_C);               // Pass current C

    return flamegpu::ALIVE;
}
"""

# Add this new CUDA function string
calculate_local_metrics_func = r"""
#include <cmath> // For fabsf
FLAMEGPU_AGENT_FUNCTION(calculate_local_metrics_func, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    // Input message: agent_state_message (Strategy & C from t-1)
    // Note: This function runs after calculate_utility_func, but uses the SAME input message
    // which contains state from t-1. This is correct for calculating metrics based on the grid state.
    const int self_ix = FLAMEGPU->getVariable<int>("ix");
    const int self_iy = FLAMEGPU->getVariable<int>("iy");
    const unsigned int self_strategy = FLAMEGPU->getVariable<unsigned int>("strategy");
    const float self_C = FLAMEGPU->getVariable<float>("C");
    const float C_threshold = 0.5f; // Use a fixed threshold for A/B typing
    // Determine own type (A or B)
    const unsigned int self_type = (self_C < C_threshold) ? 1u : 2u; // 1 for A, 2 for B
    FLAMEGPU->setVariable<unsigned int>("agent_type", self_type); // Store for potential later use/logging

    unsigned int same_strategy_count = 0;
    unsigned int same_type_count = 0;
    unsigned int total_neighbors = 0;
    bool has_different_neighbor = false;

    // Variables for group-specific metrics
    unsigned int is_type_A = (self_type == 1u) ? 1u : 0u;
    unsigned int is_type_B = (self_type == 2u) ? 1u : 0u;
    unsigned int is_cooperator = (self_strategy == 1u) ? 1u : 0u;

    // Iterate through neighbors using wrap(self_ix, self_iy, 1)
    for (const auto& neighbor_msg : FLAMEGPU->message_in.wrap(self_ix, self_iy, 1)) {
        total_neighbors++;
        unsigned int neighbor_strategy = neighbor_msg.getVariable<unsigned int>("strategy");
        float neighbor_C = neighbor_msg.getVariable<float>("C");

        // Check strategy similarity
        if (neighbor_strategy == self_strategy) {
            same_strategy_count++;
        }

        // Check cultural type similarity
        const unsigned int neighbor_type = (neighbor_C < C_threshold) ? 1u : 2u;
        if (neighbor_type == self_type) {
            same_type_count++;
        } else {
            has_different_neighbor = true;
        }
    }

    // Store counts
    FLAMEGPU->setVariable<unsigned int>("same_strategy_neighbors", same_strategy_count);
    FLAMEGPU->setVariable<unsigned int>("same_type_neighbors", same_type_count);
    FLAMEGPU->setVariable<unsigned int>("total_neighbors_count", total_neighbors);

    // Determine if boundary agent
    FLAMEGPU->setVariable<unsigned int>("is_boundary_agent", has_different_neighbor ? 1u : 0u);

    // Determine if boundary/bulk cooperator
    if (is_cooperator) { // If agent is a cooperator
        if (has_different_neighbor) {
            FLAMEGPU->setVariable<unsigned int>("is_boundary_cooperator", 1u);
            FLAMEGPU->setVariable<unsigned int>("is_bulk_cooperator", 0u);
        } else {
            FLAMEGPU->setVariable<unsigned int>("is_boundary_cooperator", 0u);
            FLAMEGPU->setVariable<unsigned int>("is_bulk_cooperator", 1u);
        }
    } else { // If agent is a defector
        FLAMEGPU->setVariable<unsigned int>("is_boundary_cooperator", 0u);
        FLAMEGPU->setVariable<unsigned int>("is_bulk_cooperator", 0u);
    }

    // --- Accumulate group-specific metrics for reduction ---
    // We need to sum these up in the StepFunction
    // Sum of cooperators per type
    FLAMEGPU->setVariable<unsigned int>("cooperator_A", is_type_A * is_cooperator);
    FLAMEGPU->setVariable<unsigned int>("cooperator_B", is_type_B * is_cooperator);

    // Sum of agents per type (for normalization)
    FLAMEGPU->setVariable<unsigned int>("count_type_A", is_type_A);
    FLAMEGPU->setVariable<unsigned int>("count_type_B", is_type_B);

    // Sum of same_type_neighbors per type
    FLAMEGPU->setVariable<unsigned int>("same_type_neighbors_A", is_type_A * same_type_count);
    FLAMEGPU->setVariable<unsigned int>("same_type_neighbors_B", is_type_B * same_type_count);

    // Sum of total_neighbors_count per type
    FLAMEGPU->setVariable<unsigned int>("total_neighbors_count_A", is_type_A * total_neighbors);
    FLAMEGPU->setVariable<unsigned int>("total_neighbors_count_B", is_type_B * total_neighbors);


    return flamegpu::ALIVE;
}
"""

decide_updates_func = r"""
#include <cmath>  // For expf, fmaxf, fminf
// No <vector> needed
// Define NeighborUtilityInfo struct
struct NeighborUtilityInfo {
    float utility;
    unsigned int strategy;
    float C;
};
FLAMEGPU_AGENT_FUNCTION(decide_updates_func, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    // ... (Get self state, utility, environment parameters - same as before) ...
    const int self_ix = FLAMEGPU->getVariable<int>("ix");
    const int self_iy = FLAMEGPU->getVariable<int>("iy");
    const float self_utility = FLAMEGPU->getVariable<float>("current_utility"); // Assuming utility was stored
    const unsigned int current_strategy = FLAMEGPU->getVariable<unsigned int>("strategy");
    const float current_C = FLAMEGPU->getVariable<float>("C");
    unsigned int next_strategy = current_strategy;
    float next_C = current_C;
    const float K = FLAMEGPU->environment.getProperty<float>("K");
    const float K_C = FLAMEGPU->environment.getProperty<float>("K_C");
    const float p_update_C = FLAMEGPU->environment.getProperty<float>("p_update_C");
    // --- Collect neighbor utility info using a fixed-size array ---
    const unsigned int MAX_NEIGHBORS = 8; // Max for Moore radius 1
    NeighborUtilityInfo neighbors_utility_info[MAX_NEIGHBORS];
    unsigned int neighbor_count = 0; // Track how many neighbors we actually found
    // Use wrap with radius 1 around self position
    for (const auto& neighbor_msg : FLAMEGPU->message_in.wrap(self_ix, self_iy, 1)) {
        // Ensure we don't exceed array bounds (shouldn't happen with radius 1)
        if (neighbor_count < MAX_NEIGHBORS) {
            neighbors_utility_info[neighbor_count].utility = neighbor_msg.getVariable<float>("utility");
            neighbors_utility_info[neighbor_count].strategy = neighbor_msg.getVariable<unsigned int>("strategy");
            neighbors_utility_info[neighbor_count].C = neighbor_msg.getVariable<float>("C");
            neighbor_count++;
        }
        // Optional: break if neighbor_count == MAX_NEIGHBORS, though wrap should handle it.
    }
    // --- Select ONE random neighbor to compare against ---
    if (neighbor_count > 0) {
        // Generate random index based on the *actual* number of neighbors found
        unsigned int random_neighbor_index = FLAMEGPU->random.uniform<unsigned int>(0u, neighbor_count - 1);
        const NeighborUtilityInfo& selected_neighbor = neighbors_utility_info[random_neighbor_index];
        // --- Calculate Delta Utility ---
        float neighbor_utility = selected_neighbor.utility;
        float delta_utility = neighbor_utility - self_utility;
        // --- Decide Strategy Update (Fermi Rule) ---
        unsigned int neighbor_strategy_to_adopt = selected_neighbor.strategy;
        float argument = (K > 1e-9f) ? fmaxf(-70.0f, fminf(70.0f, -delta_utility / K)) : 0.0f;
        float prob_adopt_strategy = (K > 1e-9f) ? (1.0f / (1.0f + expf(argument))) : (delta_utility > 0.0f ? 1.0f : 0.0f);
        if (FLAMEGPU->random.uniform<float>() < prob_adopt_strategy) {
            next_strategy = neighbor_strategy_to_adopt;
        }
        // --- Decide Culture Update (Fermi Rule, conditional) ---
        if (FLAMEGPU->random.uniform<float>() < p_update_C) {
             float neighbor_C_to_adopt = selected_neighbor.C;
             float argument_C = (K_C > 1e-9f) ? fmaxf(-70.0f, fminf(70.0f, -delta_utility / K_C)) : 0.0f;
             float prob_adopt_culture = (K_C > 1e-9f) ? (1.0f / (1.0f + expf(argument_C))) : (delta_utility > 0.0f ? 1.0f : 0.0f);
             if (FLAMEGPU->random.uniform<float>() < prob_adopt_culture) {
                 next_C = neighbor_C_to_adopt;
                 next_C = fmaxf(0.0f, fminf(1.0f, next_C));
             }
        }
    } // else (no neighbors found)
    // --- Store the final decided next state ---
    FLAMEGPU->setVariable<unsigned int>("next_strategy", next_strategy);
    FLAMEGPU->setVariable<float>("next_C", next_C);
    return flamegpu::ALIVE;
}
"""

mutate_func = r"""
FLAMEGPU_AGENT_FUNCTION(mutate, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float p_mut_strategy = FLAMEGPU->environment.getProperty<float>("p_mut_strategy");
    const float p_mut_culture = FLAMEGPU->environment.getProperty<float>("p_mut_culture");
    unsigned int current_next_strategy = FLAMEGPU->getVariable<unsigned int>("next_strategy");
    float current_next_C = FLAMEGPU->getVariable<float>("next_C");

    if (FLAMEGPU->random.uniform<float>() < p_mut_strategy) {
        current_next_strategy = FLAMEGPU->random.uniform<unsigned int>(0, 1);
    }
    if (FLAMEGPU->random.uniform<float>() < p_mut_culture) {
        current_next_C = FLAMEGPU->random.uniform<float>(0.0f, 1.0f);
    }

    FLAMEGPU->setVariable<unsigned int>("next_strategy", current_next_strategy);
    FLAMEGPU->setVariable<float>("next_C", current_next_C);
    return flamegpu::ALIVE;
}
"""


advance_func = r"""
FLAMEGPU_AGENT_FUNCTION(advance_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("strategy", FLAMEGPU->getVariable<unsigned int>("next_strategy"));
    FLAMEGPU->setVariable<float>("C", FLAMEGPU->getVariable<float>("next_C"));
    return flamegpu::ALIVE;
}
"""
