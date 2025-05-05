# cultural_game_cuda.py

output_state = r"""
FLAMEGPU_AGENT_FUNCTION(output_state, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    // Output current state needed for interaction/comparison by others
    const flamegpu::id_t id = FLAMEGPU->getID();
    const unsigned int strategy = FLAMEGPU->getVariable<unsigned int>("strategy");
    const float C = FLAMEGPU->getVariable<float>("C");
    // We don't output utility directly, as it depends on the observer in Mesa's logic.
    // We'll calculate hypothetical utilities during interaction.

    FLAMEGPU->message_out.setVariable<int>("id", id); // Use int for ID matching message example
    FLAMEGPU->message_out.setVariable<unsigned int>("strategy", strategy);
    FLAMEGPU->message_out.setVariable<float>("C", C);

    // Pass forward the next state decided in the *previous* step's update/mutate phases
    // This ensures consistency if multiple layers modify next_state before advance
    FLAMEGPU->setVariable<unsigned int>("next_strategy", FLAMEGPU->getVariable<unsigned int>("next_strategy"));
    FLAMEGPU->setVariable<float>("next_C", FLAMEGPU->getVariable<float>("next_C"));

    return flamegpu::ALIVE;
}
"""

# Combines Mesa's calculate_utility (partially), decide_strategy_update, decide_culture_update
# Based on interaction with ONE random agent, not all neighbors.
calculate_utility_and_update = r"""
#include <cmath> // For expf

FLAMEGPU_AGENT_FUNCTION(calculate_utility_and_update, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    // Get own state
    const flamegpu::id_t id = FLAMEGPU->getID();
    const unsigned int self_strategy = FLAMEGPU->getVariable<unsigned int>("strategy");
    const float self_C = FLAMEGPU->getVariable<float>("C");
    unsigned int next_strategy = self_strategy; // Start with current strategy
    float next_C = self_C;                 // Start with current C

    // Environment parameters for Fermi update
    const float K = FLAMEGPU->environment.getProperty<float>("K");
    const float K_C = FLAMEGPU->environment.getProperty<float>("K_C");
    const float p_update_C = FLAMEGPU->environment.getProperty<float>("p_update_C");

    // Payoff Matrix (b is payoff for D vs C)
    // payoff[MyStrategy][OtherStrategy] = MyRawPayoff
    // P(C,C) = 1; P(C,D) = 0; P(D,C) = b; P(D,D) = 0
    // Assuming the raw payoff matrix is stored directly:
    // payoff[0][0] = 0 (D-D)
    // payoff[0][1] = b (D-C)
    // payoff[1][0] = 0 (C-D)
    // payoff[1][1] = 1 (C-C)
    auto payoff = FLAMEGPU->environment.getMacroProperty<float, 2, 2>("payoff");

    // Get total number of agents for random selection
    // Note: Agent count might change if agents can die/reproduce. Assuming fixed for now.
    const unsigned int N = FLAMEGPU->environment.getProperty<unsigned int>("agent_count"); // Needs to be set correctly

    // Find a random agent to interact with (excluding self)
    int partner_id = -1;
    unsigned int partner_strategy = 0; // Default values
    float partner_C = 0.0f;

    if (N > 1) {
        int selected_agent_index = FLAMEGPU->random.uniform<int>(0, N - 2); // Select random index from 0 to N-2

        // Iterate through messages to find the agent at the selected index
        // This is inefficient but follows the BruteForce pattern.
        // A better approach might involve different messaging or data structures if performance critical.
        int current_index = 0;
        for (const auto& msg : FLAMEGPU->message_in) {
            int msg_id = msg.getVariable<int>("id");
            if (msg_id != id) { // If not self
                if (current_index == selected_agent_index) {
                    partner_id = msg_id;
                    partner_strategy = msg.getVariable<unsigned int>("strategy");
                    partner_C = msg.getVariable<float>("C");
                    break; // Found the partner
                }
                current_index++;
            }
        }
    }


    // If a partner was successfully found (N > 1 and message found)
    if (partner_id != -1) {
        // Calculate HYPOTHETICAL utilities based on this specific interaction
        // Utility = (1-C) * MyRawPayoff + C * PartnerRawPayoff

        // 1. My utility if I interact with partner
        float my_raw_payoff = payoff[self_strategy][partner_strategy];
        float partner_raw_payoff_vs_me = payoff[partner_strategy][self_strategy];
        float U_self = (1.0f - self_C) * my_raw_payoff + self_C * partner_raw_payoff_vs_me;

        // 2. Partner's utility if they interact with me
        float partner_raw_payoff = payoff[partner_strategy][self_strategy];
        float my_raw_payoff_vs_partner = payoff[self_strategy][partner_strategy];
        float U_partner = (1.0f - partner_C) * partner_raw_payoff + partner_C * my_raw_payoff_vs_partner;

        // 3. Calculate Utility Difference
        float delta_utility = U_partner - U_self;

        // 4. Decide Strategy Update (Fermi Rule)
        if (K > 1e-9f) { // Avoid division by zero/very small K
            float prob_adopt_strategy = 1.0f / (1.0f + expf(-delta_utility / K));
            if (FLAMEGPU->random.uniform<float>() < prob_adopt_strategy) {
                next_strategy = partner_strategy;
            }
        } else { // Deterministic update for K=0
            if (delta_utility > 0.0f) {
                next_strategy = partner_strategy;
            }
        }

        // 5. Decide Culture Update (Fermi Rule, conditional on p_update_C)
        if (FLAMEGPU->random.uniform<float>() < p_update_C) {
            if (K_C > 1e-9f) { // Avoid division by zero/very small K_C
                 float prob_adopt_culture = 1.0f / (1.0f + expf(-delta_utility / K_C));
                 if (FLAMEGPU->random.uniform<float>() < prob_adopt_culture) {
                    next_C = partner_C;
                 }
            } else { // Deterministic update for K_C=0
                if (delta_utility > 0.0f) {
                     next_C = partner_C;
                }
            }
        }
         // Ensure C stays within [0, 1] - should naturally happen if adopting, but good practice
         next_C = fmaxf(0.0f, fminf(1.0f, next_C));

    } // else (no partner found / N=1), next_strategy and next_C remain unchanged

    // Store the decided next state
    FLAMEGPU->setVariable<unsigned int>("next_strategy", next_strategy);
    FLAMEGPU->setVariable<float>("next_C", next_C);

    return flamegpu::ALIVE;
}
"""

mutate = r"""
#include <cmath> // For fmaxf, fminf

FLAMEGPU_AGENT_FUNCTION(mutate, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Get mutation probabilities from environment
    const float p_mut_strategy = FLAMEGPU->environment.getProperty<float>("p_mut_strategy");
    const float p_mut_culture = FLAMEGPU->environment.getProperty<float>("p_mut_culture");

    // Get the tentative next state decided by the update function
    unsigned int current_next_strategy = FLAMEGPU->getVariable<unsigned int>("next_strategy");
    float current_next_C = FLAMEGPU->getVariable<float>("next_C");

    // Mutate Strategy
    if (FLAMEGPU->random.uniform<float>() < p_mut_strategy) {
        // Flip the strategy (0 becomes 1, 1 becomes 0) or choose randomly
        // Random choice is simpler:
        current_next_strategy = FLAMEGPU->random.uniform<unsigned int>(0, 1);
    }

    // Mutate Culture
    if (FLAMEGPU->random.uniform<float>() < p_mut_culture) {
        current_next_C = FLAMEGPU->random.uniform<float>(0.0f, 1.0f); // New random C value
    }

    // Set the potentially mutated next state
    FLAMEGPU->setVariable<unsigned int>("next_strategy", current_next_strategy);
    FLAMEGPU->setVariable<float>("next_C", current_next_C); // Already ensures [0,1] if mutated

    return flamegpu::ALIVE;
}
"""

advance = r"""
FLAMEGPU_AGENT_FUNCTION(advance, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Update the agent's main state variables from the 'next_' variables
    // These 'next_' variables hold the results from update and mutate phases
    FLAMEGPU->setVariable<unsigned int>("strategy", FLAMEGPU->getVariable<unsigned int>("next_strategy"));
    FLAMEGPU->setVariable<float>("C", FLAMEGPU->getVariable<float>("next_C"));

    return flamegpu::ALIVE;
}
"""
