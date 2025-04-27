# cuda_kernels.py
# Contains C++ agent function implementations as RTC strings

import pyflamegpu # For flamegpu:: namespace access if needed, though typically not directly used in strings

# Note: MAX_NEIGHBORS is hardcoded here based on the Python definition.
# If it needs to be dynamic, consider passing it via environment properties.
MAX_NEIGHBORS = 8

# Include necessary C++ headers within the string if needed (e.g., <cmath> for math functions)

output_state_cpp = r'''
#include <cmath> // For potential future use, not strictly needed here

FLAMEGPU_AGENT_FUNCTION(output_state, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    // Agent ID is implicitly handled by the framework for spatial messages,
    // but we include it in the message for potential use by receivers.
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("strategy", FLAMEGPU->getVariable<int>("strategy"));
    FLAMEGPU->message_out.setVariable<float>("C", FLAMEGPU->getVariable<float>("C"));
    // Location is automatically handled by MessageSpatial2D output
    return flamegpu::ALIVE;
}
'''

calculate_utility_cpp = r'''
#include <cmath> // For potential future use, not strictly needed here

FLAMEGPU_AGENT_FUNCTION(calculate_utility, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    // Get my own state
    int my_strategy = FLAMEGPU->getVariable<int>("strategy");
    float my_C = FLAMEGPU->getVariable<float>("C");
    flamegpu::id_t my_id = FLAMEGPU->getID(); // For skipping self-message

    // Get payoff parameters from environment
    // Note: Ensure environment properties use matching types (e.g., float here)
    const float T = FLAMEGPU->environment.getProperty<float>("T_PAYOFF");
    const float R = FLAMEGPU->environment.getProperty<float>("R_PAYOFF");
    const float S = FLAMEGPU->environment.getProperty<float>("S_PAYOFF");
    const float P = FLAMEGPU->environment.getProperty<float>("P_PAYOFF");

    float total_utility = 0.0f;
    unsigned int neighbor_count_actual = 0; // Use unsigned int for counts

    // Iterate through messages from neighbors within the radius
    for (const auto& neighbor_message : FLAMEGPU->message_in) {
        // Skip message from self
        if (neighbor_message.getVariable<flamegpu::id_t>("id") == my_id) {
            continue;
        }

        int neighbor_strategy = neighbor_message.getVariable<int>("strategy");
        // float neighbor_C = neighbor_message.getVariable<float>("C"); // Not needed here

        // Calculate payoffs based on the interaction (my_strategy, neighbor_strategy)
        float my_payoff = 0.0f;
        float neighbor_payoff = 0.0f; // Payoff the *neighbor* receives from *me*

        if (my_strategy == 1 && neighbor_strategy == 1) { // C-C
            my_payoff = R;
            neighbor_payoff = R;
        } else if (my_strategy == 1 && neighbor_strategy == 0) { // C-D
            my_payoff = S;
            neighbor_payoff = T; // Neighbor gets Temptation
        } else if (my_strategy == 0 && neighbor_strategy == 1) { // D-C
            my_payoff = T; // I get Temptation
            neighbor_payoff = S;
        } else { // D-D (my_strategy == 0 && neighbor_strategy == 0)
            my_payoff = P;
            neighbor_payoff = P;
        }

        // Calculate the cultural utility contribution from this neighbor
        // utility = (1-C)*my_payoff + C*neighbor_payoff (where neighbor_payoff is the payoff *they* get from *me*)
        float utility_contribution = (1.0f - my_C) * my_payoff + my_C * neighbor_payoff;
        total_utility += utility_contribution;
        neighbor_count_actual++; // Increment count after processing a valid neighbor
    }

    // Store the calculated total utility
    FLAMEGPU->setVariable<float>("utility", total_utility);

    // Initialize next_strategy and next_C for the upcoming decision steps
    FLAMEGPU->setVariable<int>("next_strategy", my_strategy);
    FLAMEGPU->setVariable<float>("next_C", my_C);

    // Optional: Reset neighbor count used for decision making here,
    // although process_neighbor_utility will overwrite it.
    // FLAMEGPU->setVariable<unsigned int>("neighbor_count", 0);

    return flamegpu::ALIVE;
}
'''

output_utility_cpp = r'''
#include <cmath>

FLAMEGPU_AGENT_FUNCTION(output_utility, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("strategy", FLAMEGPU->getVariable<int>("strategy"));
    FLAMEGPU->message_out.setVariable<float>("C", FLAMEGPU->getVariable<float>("C"));
    FLAMEGPU->message_out.setVariable<float>("utility", FLAMEGPU->getVariable<float>("utility"));
    // Location is automatic
    return flamegpu::ALIVE;
}
'''

process_neighbor_utility_cpp = r'''
#include <cmath>

// Assumes MAX_NEIGHBORS is defined or hardcoded (using 8 here)
#define MAX_NEIGHBORS 8

FLAMEGPU_AGENT_FUNCTION(process_neighbor_utility, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    flamegpu::id_t my_id = FLAMEGPU->getID();
    unsigned int neighbor_count = 0; // Index for storing neighbor data

    // Iterate through utility messages from neighbors
    for (const auto& neighbor_message : FLAMEGPU->message_in) {
        // Optional: Skip message from self
        if (neighbor_message.getVariable<flamegpu::id_t>("id") == my_id) {
            continue;
        }

        // Check if we have space to store this neighbor's info
        if (neighbor_count < MAX_NEIGHBORS) {
            // Get neighbor data from the message
            int n_strategy = neighbor_message.getVariable<int>("strategy");
            float n_C = neighbor_message.getVariable<float>("C");
            float n_utility = neighbor_message.getVariable<float>("utility");

            // Store the data in the agent's array variables at the current index
            // Note: C++ API uses index directly in setVariable for arrays
            FLAMEGPU->setVariable<int>("neighbor_strategies", neighbor_count, n_strategy);
            FLAMEGPU->setVariable<float>("neighbor_Cs", neighbor_count, n_C);
            FLAMEGPU->setVariable<float>("neighbor_utilities", neighbor_count, n_utility);

            neighbor_count++; // Increment the count of stored neighbors
        } else {
            // Optional: Warn if more neighbors than storage space (cannot print easily in RTC)
            // Break if you only care about the first MAX_NEIGHBORS found
             break;
        }
    }

    // Store the final count of neighbors whose data was stored
    FLAMEGPU->setVariable<unsigned int>("neighbor_count", neighbor_count);

    return flamegpu::ALIVE;
}
'''

decide_strategy_update_cpp = r'''
#include <cmath> // For expf
#include <limits> // For infinity

// Assumes MAX_NEIGHBORS is defined or hardcoded (using 8 here)
// #define MAX_NEIGHBORS 8 // Defined globally or per-function if needed

FLAMEGPU_AGENT_FUNCTION(decide_strategy_update, flamegpu::MessageNone, flamegpu::MessageNone) {
    float my_utility = FLAMEGPU->getVariable<float>("utility");
    unsigned int neighbor_count = FLAMEGPU->getVariable<unsigned int>("neighbor_count");

    // Only proceed if there are neighbors to compare with
    if (neighbor_count > 0) {
        // Get noise parameter K from environment
        const float K = FLAMEGPU->environment.getProperty<float>("K");

        // Randomly select one neighbor index from the stored list [0, neighbor_count - 1]
        // FLAMEGPU->random.uniform<unsigned int>() % neighbor_count; gives [0, count-1]
        unsigned int rand_idx = FLAMEGPU->random.uniform<unsigned int>() % neighbor_count;


        // Get the chosen neighbor's data from the agent's arrays using the index
        int neighbor_strategy = FLAMEGPU->getVariable<int>("neighbor_strategies", rand_idx);
        float neighbor_utility = FLAMEGPU->getVariable<float>("neighbor_utilities", rand_idx);

        // Calculate utility difference
        float delta_utility = neighbor_utility - my_utility;

        float prob_adopt = 0.0f;
        // Apply Fermi rule
        if (K < 1e-9f) { // Handle K close to zero (deterministic choice)
            prob_adopt = (delta_utility > 0.0f) ? 1.0f : 0.0f;
        } else {
            float exponent_arg = -delta_utility / K;
            // Overflow protection for expf()
            // Use appropriate limits for float if available, or large numbers
             const float exp_max_arg = 80.0f; // Around expf(80) overflows float
             if (exponent_arg > exp_max_arg) {
                 prob_adopt = 0.0f; // Denominator becomes huge -> probability near 0
             } else if (exponent_arg < -exp_max_arg) {
                 prob_adopt = 1.0f; // expf(arg) becomes near 0 -> probability near 1
             } else {
                 prob_adopt = 1.0f / (1.0f + expf(exponent_arg));
             }
        }

        // Decide whether to adopt based on probability
        // FLAMEGPU->random.uniform<float>() generates in [0.0, 1.0)
        if (FLAMEGPU->random.uniform<float>() < prob_adopt) {
            // Adopt the neighbor's strategy for the next step
            FLAMEGPU->setVariable<int>("next_strategy", neighbor_strategy);
        }
        // Else: Keep the current strategy (already set in next_strategy during calculate_utility)
    }
    // If neighbor_count is 0, next_strategy remains the current strategy

    return flamegpu::ALIVE;
}
'''

decide_culture_update_cpp = r'''
#include <cmath> // For expf
#include <limits> // For infinity

// Assumes MAX_NEIGHBORS is defined or hardcoded (using 8 here)
// #define MAX_NEIGHBORS 8

FLAMEGPU_AGENT_FUNCTION(decide_culture_update, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float p_update_C = FLAMEGPU->environment.getProperty<float>("p_update_C");

    // Only attempt update with a certain probability
    if (FLAMEGPU->random.uniform<float>() < p_update_C) {
        float my_utility = FLAMEGPU->getVariable<float>("utility");
        unsigned int neighbor_count = FLAMEGPU->getVariable<unsigned int>("neighbor_count");

        // Only proceed if there are neighbors to compare with
        if (neighbor_count > 0) {
            // Get cultural noise parameter K_C from environment
            const float K_C = FLAMEGPU->environment.getProperty<float>("K_C");

            // Randomly select one neighbor index [0, neighbor_count - 1]
             unsigned int rand_idx = FLAMEGPU->random.uniform<unsigned int>() % neighbor_count;

            // Get the chosen neighbor's C value and utility
            float neighbor_C = FLAMEGPU->getVariable<float>("neighbor_Cs", rand_idx);
            float neighbor_utility = FLAMEGPU->getVariable<float>("neighbor_utilities", rand_idx);

            // Calculate utility difference
            float delta_utility = neighbor_utility - my_utility;

            float prob_adopt_culture = 0.0f;
            // Apply Fermi rule using K_C
            if (K_C < 1e-9f) { // Handle K_C close to zero
                prob_adopt_culture = (delta_utility > 0.0f) ? 1.0f : 0.0f;
            } else {
                 float exponent_arg = -delta_utility / K_C;
                 const float exp_max_arg = 80.0f; // Check float limits / empirical value
                 if (exponent_arg > exp_max_arg) {
                     prob_adopt_culture = 0.0f;
                 } else if (exponent_arg < -exp_max_arg) {
                     prob_adopt_culture = 1.0f;
                 } else {
                     prob_adopt_culture = 1.0f / (1.0f + expf(exponent_arg));
                 }
            }

            // Decide whether to adopt based on probability
            if (FLAMEGPU->random.uniform<float>() < prob_adopt_culture) {
                // Adopt the neighbor's C value for the next step
                FLAMEGPU->setVariable<float>("next_C", neighbor_C);
            }
            // Else: Keep the current C value (already set in next_C)
        }
    }
    // If random check fails or neighbor_count is 0, next_C remains the current C

    return flamegpu::ALIVE;
}
'''

mutate_cpp = r'''
#include <cmath> // For potential future use

FLAMEGPU_AGENT_FUNCTION(mutate, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Get mutation rates from environment
    const float mutation_rate_strategy = FLAMEGPU->environment.getProperty<float>("mutation_rate_strategy");
    const float mutation_rate_C = FLAMEGPU->environment.getProperty<float>("mutation_rate_C");

    // --- Strategy Mutation ---
    if (FLAMEGPU->random.uniform<float>() < mutation_rate_strategy) {
        // Flip the current next_strategy
        int current_next_strategy = FLAMEGPU->getVariable<int>("next_strategy");
        FLAMEGPU->setVariable<int>("next_strategy", 1 - current_next_strategy); // Flip 0 to 1, 1 to 0
    }

    // --- Culture (C value) Mutation ---
    if (FLAMEGPU->random.uniform<float>() < mutation_rate_C) {
        // Assign a new random C value between 0.0 and 1.0
        // FLAMEGPU->random.uniform<float>() generates float in [0.0, 1.0)
        FLAMEGPU->setVariable<float>("next_C", FLAMEGPU->random.uniform<float>());
    }

    return flamegpu::ALIVE;
}
'''

advance_cpp = r'''
#include <cmath> // For potential future use

FLAMEGPU_AGENT_FUNCTION(advance, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Get the finalized next state values
    int final_next_strategy = FLAMEGPU->getVariable<int>("next_strategy");
    float final_next_C = FLAMEGPU->getVariable<float>("next_C");

    // Update the main state variables
    FLAMEGPU->setVariable<int>("strategy", final_next_strategy);
    FLAMEGPU->setVariable<float>("C", final_next_C);

    // Reset utility and neighbor count (optional, good practice if not reset elsewhere)
    // FLAMEGPU->setVariable<float>("utility", 0.0f);
    // FLAMEGPU->setVariable<unsigned int>("neighbor_count", 0); // Reset for next iteration's accumulation

    return flamegpu::ALIVE;
}
'''
