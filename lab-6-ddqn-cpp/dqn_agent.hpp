#ifndef DQN_AGENT_HPP
#define DQN_AGENT_HPP

#include "neural_network.hpp"
#include "replay_buffer.hpp"
#include <random>
#include <algorithm>
#include <cmath>

class DQNAgent {
private:
    NeuralNetwork policy_net;
    NeuralNetwork target_net;
    ReplayBuffer memory;
    
    int state_dim;
    int action_dim;
    
    // Hiperparámetros
    double gamma;           // Factor de descuento
    double epsilon;         // Probabilidad de exploración
    double epsilon_min;     // Epsilon mínimo
    double epsilon_decay;   // Decay de epsilon
    double tau;             // Para soft update
    int batch_size;
    
    bool use_double_dqn;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;

public:
    DQNAgent(int state_dim, int action_dim, bool double_dqn = false,
             double lr = 0.001, double gamma = 0.99,
             double eps_start = 1.0, double eps_end = 0.01, double eps_decay = 0.995,
             int batch_size = 64, int memory_size = 10000, double tau = 0.005,
             unsigned int seed = 42)
        : policy_net({state_dim, 128, 128, action_dim}, lr, seed),
          target_net({state_dim, 128, 128, action_dim}, lr, seed + 1),
          memory(memory_size, seed + 2),
          state_dim(state_dim),
          action_dim(action_dim),
          gamma(gamma),
          epsilon(eps_start),
          epsilon_min(eps_end),
          epsilon_decay(eps_decay),
          tau(tau),
          batch_size(batch_size),
          use_double_dqn(double_dqn),
          rng(seed + 3),
          uniform_dist(0.0, 1.0) {
        
        // Inicializar target network con los mismos pesos
        target_net.copyWeightsFrom(policy_net);
    }
    
    // Seleccionar acción (epsilon-greedy)
    int selectAction(const std::vector<double>& state, bool training = true) {
        if (training && uniform_dist(rng) < epsilon) {
            // Acción aleatoria
            std::uniform_int_distribution<int> action_dist(0, action_dim - 1);
            return action_dist(rng);
        }
        
        // Acción greedy
        std::vector<double> q_values = policy_net.forward(state);
        return policy_net.argmax(q_values);
    }
    
    // Almacenar transición
    void storeTransition(const std::vector<double>& state, int action, double reward,
                         const std::vector<double>& next_state, bool done) {
        memory.push(state, action, reward, next_state, done);
    }
    
    // Paso de entrenamiento
    double trainStep() {
        if (!memory.canSample(batch_size)) {
            return -1.0;
        }
        
        std::vector<Transition> batch = memory.sample(batch_size);
        double total_loss = 0.0;
        
        for (const auto& trans : batch) {
            // Q-values actuales
            std::vector<double> current_q = policy_net.forward(trans.state);
            
            // Calcular target
            double target_value;
            if (trans.done) {
                target_value = trans.reward;
            } else {
                std::vector<double> next_q_target = target_net.forward(trans.next_state);
                
                if (use_double_dqn) {
                    // Double DQN: seleccionar acción con policy, evaluar con target
                    std::vector<double> next_q_policy = policy_net.forward(trans.next_state);
                    int best_action = policy_net.argmax(next_q_policy);
                    target_value = trans.reward + gamma * next_q_target[best_action];
                } else {
                    // DQN estándar
                    double max_next_q = *std::max_element(next_q_target.begin(), next_q_target.end());
                    target_value = trans.reward + gamma * max_next_q;
                }
            }
            
            // Crear target vector (solo actualizar la acción tomada)
            std::vector<double> target_q = current_q;
            target_q[trans.action] = target_value;
            
            // Forward y backward
            policy_net.forward(trans.state);
            policy_net.backward(target_q);
            
            // Calcular loss
            double diff = current_q[trans.action] - target_value;
            total_loss += diff * diff;
        }
        
        // Soft update de target network
        target_net.softUpdate(policy_net, tau);
        
        // Decay epsilon
        epsilon = std::max(epsilon_min, epsilon * epsilon_decay);
        
        return total_loss / batch_size;
    }
    
    // Getters
    double getEpsilon() const { return epsilon; }
    size_t getMemorySize() const { return memory.size(); }
    std::string getAlgorithmName() const { return use_double_dqn ? "Double DQN" : "DQN"; }
    
    // Evaluar sin exploración
    int getBestAction(const std::vector<double>& state) {
        std::vector<double> q_values = policy_net.forward(state);
        return policy_net.argmax(q_values);
    }
};

#endif // DQN_AGENT_HPP
