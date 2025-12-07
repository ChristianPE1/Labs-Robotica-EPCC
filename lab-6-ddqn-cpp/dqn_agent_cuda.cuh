#ifndef DQN_AGENT_CUDA_CUH
#define DQN_AGENT_CUDA_CUH

#include "neural_network_cuda.cuh"
#include "replay_buffer.hpp"
#include <random>
#include <algorithm>
#include <cmath>

class DQNAgentCUDA {
private:
    NeuralNetworkCUDA policy_net;
    NeuralNetworkCUDA target_net;
    ReplayBuffer memory;
    
    int state_dim;
    int action_dim;
    int batch_size;
    
    // Hiperparámetros
    float gamma;
    float epsilon;
    float epsilon_min;
    float epsilon_decay;
    float tau;
    
    bool use_double_dqn;
    std::mt19937 rng;
    std::uniform_real_distribution<float> uniform_dist;
    
    // Memoria GPU para batch training
    float* d_states;
    float* d_next_states;
    float* d_targets;
    int* d_actions;
    float* d_rewards;
    bool* d_dones;

public:
    DQNAgentCUDA(int state_dim, int action_dim, bool double_dqn = false,
                 float lr = 0.001f, float gamma = 0.99f,
                 float eps_start = 1.0f, float eps_end = 0.01f, float eps_decay = 0.995f,
                 int batch_size = 64, int memory_size = 10000, float tau = 0.005f,
                 unsigned int seed = 42)
        : policy_net({state_dim, 128, 128, action_dim}, lr, batch_size, seed),
          target_net({state_dim, 128, 128, action_dim}, lr, batch_size, seed + 1),
          memory(memory_size, seed + 2),
          state_dim(state_dim),
          action_dim(action_dim),
          batch_size(batch_size),
          gamma(gamma),
          epsilon(eps_start),
          epsilon_min(eps_end),
          epsilon_decay(eps_decay),
          tau(tau),
          use_double_dqn(double_dqn),
          rng(seed + 3),
          uniform_dist(0.0f, 1.0f) {
        
        // Inicializar target network con los mismos pesos
        target_net.copyWeightsFrom(policy_net);
        
        // Alojar memoria GPU para batch
        CUDA_CHECK(cudaMalloc(&d_states, batch_size * state_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_next_states, batch_size * state_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_targets, batch_size * action_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_actions, batch_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_rewards, batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dones, batch_size * sizeof(bool)));
    }
    
    ~DQNAgentCUDA() {
        cudaFree(d_states);
        cudaFree(d_next_states);
        cudaFree(d_targets);
        cudaFree(d_actions);
        cudaFree(d_rewards);
        cudaFree(d_dones);
    }
    
    // Seleccionar acción (epsilon-greedy)
    int selectAction(const std::vector<double>& state, bool training = true) {
        if (training && uniform_dist(rng) < epsilon) {
            std::uniform_int_distribution<int> action_dist(0, action_dim - 1);
            return action_dist(rng);
        }
        
        // Convertir a float y obtener Q-values
        std::vector<float> state_f(state.begin(), state.end());
        std::vector<float> q_values = policy_net.forward(state_f);
        return policy_net.argmax(q_values);
    }
    
    // Almacenar transición
    void storeTransition(const std::vector<double>& state, int action, double reward,
                         const std::vector<double>& next_state, bool done) {
        memory.push(state, action, reward, next_state, done);
    }
    
    // Paso de entrenamiento en GPU
    float trainStep() {
        if (!memory.canSample(batch_size)) {
            return -1.0f;
        }
        
        std::vector<Transition> batch = memory.sample(batch_size);
        
        // Preparar datos del batch en CPU
        std::vector<float> h_states(batch_size * state_dim);
        std::vector<float> h_next_states(batch_size * state_dim);
        std::vector<int> h_actions(batch_size);
        std::vector<float> h_rewards(batch_size);
        std::vector<bool> h_dones(batch_size);
        
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < state_dim; ++j) {
                h_states[i * state_dim + j] = (float)batch[i].state[j];
                h_next_states[i * state_dim + j] = (float)batch[i].next_state[j];
            }
            h_actions[i] = batch[i].action;
            h_rewards[i] = (float)batch[i].reward;
            h_dones[i] = batch[i].done;
        }
        
        // Calcular targets en CPU (más simple para Double DQN)
        std::vector<float> h_targets(batch_size * action_dim);
        float total_loss = 0.0f;
        
        for (int i = 0; i < batch_size; ++i) {
            // Forward actual state
            std::vector<float> state_i(h_states.begin() + i * state_dim, 
                                       h_states.begin() + (i + 1) * state_dim);
            std::vector<float> current_q = policy_net.forward(state_i);
            
            // Calcular target
            float target_value;
            if (h_dones[i]) {
                target_value = h_rewards[i];
            } else {
                std::vector<float> next_state_i(h_next_states.begin() + i * state_dim,
                                                 h_next_states.begin() + (i + 1) * state_dim);
                std::vector<float> next_q_target = target_net.forward(next_state_i);
                
                if (use_double_dqn) {
                    std::vector<float> next_q_policy = policy_net.forward(next_state_i);
                    int best_action = policy_net.argmax(next_q_policy);
                    target_value = h_rewards[i] + gamma * next_q_target[best_action];
                } else {
                    float max_next_q = *std::max_element(next_q_target.begin(), next_q_target.end());
                    target_value = h_rewards[i] + gamma * max_next_q;
                }
            }
            
            // Crear target (copiar current_q y actualizar solo acción tomada)
            for (int j = 0; j < action_dim; ++j) {
                h_targets[i * action_dim + j] = current_q[j];
            }
            h_targets[i * action_dim + h_actions[i]] = target_value;
            
            // Calcular loss
            float diff = current_q[h_actions[i]] - target_value;
            total_loss += diff * diff;
        }
        
        // Copiar a GPU y entrenar
        CUDA_CHECK(cudaMemcpy(policy_net.getActivationsDevice(0), h_states.data(),
                              batch_size * state_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(),
                              batch_size * action_dim * sizeof(float), cudaMemcpyHostToDevice));
        
        // Forward y backward en GPU
        policy_net.forwardBatch(batch_size);
        policy_net.backwardBatch(d_targets, batch_size);
        
        // Soft update
        target_net.softUpdate(policy_net, tau);
        
        // Decay epsilon
        epsilon = std::max(epsilon_min, epsilon * epsilon_decay);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        return total_loss / batch_size;
    }
    
    // Getters
    float getEpsilon() const { return epsilon; }
    size_t getMemorySize() const { return memory.size(); }
    std::string getAlgorithmName() const { return use_double_dqn ? "Double DQN" : "DQN"; }
    
    int getBestAction(const std::vector<double>& state) {
        std::vector<float> state_f(state.begin(), state.end());
        std::vector<float> q_values = policy_net.forward(state_f);
        return policy_net.argmax(q_values);
    }
};

#endif // DQN_AGENT_CUDA_CUH
