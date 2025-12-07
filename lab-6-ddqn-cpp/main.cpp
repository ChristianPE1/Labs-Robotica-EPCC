#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include "cartpole.hpp"
#include "dqn_agent.hpp"

// Estructura para almacenar métricas
struct TrainingMetrics {
    std::vector<double> rewards;
    std::vector<int> lengths;
    std::vector<double> losses;
    std::vector<double> epsilons;
    int success_count = 0;
};

// Entrenar un agente
TrainingMetrics trainAgent(bool use_double_dqn, int num_episodes = 500, bool verbose = true) {
    std::string algo_name = use_double_dqn ? "Double DQN" : "DQN";
    
    if (verbose) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "Entrenando " << algo_name << std::endl;
        std::cout << "============================================================" << std::endl;
    }
    
    // Crear entorno y agente
    CartPole env(42);
    DQNAgent agent(CartPole::STATE_DIM, CartPole::ACTION_DIM, use_double_dqn,
                   0.001,   // learning_rate
                   0.99,    // gamma
                   1.0,     // epsilon_start
                   0.05,    // epsilon_end (mantener algo de exploración)
                   0.9995,  // epsilon_decay (muy lento)
                   64,      // batch_size
                   50000,   // memory_size (grande)
                   0.001);  // tau (suave)
    
    TrainingMetrics metrics;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        auto state_arr = env.reset();
        std::vector<double> state(state_arr.begin(), state_arr.end());
        
        double episode_reward = 0.0;
        double episode_loss = 0.0;
        int loss_count = 0;
        
        while (!env.isDone()) {
            // Seleccionar acción
            int action = agent.selectAction(state, true);
            
            // Ejecutar acción
            auto [next_state_arr, reward, done] = env.step(action);
            std::vector<double> next_state(next_state_arr.begin(), next_state_arr.end());
            
            // Almacenar transición
            agent.storeTransition(state, action, reward, next_state, done);
            
            // Entrenar
            double loss = agent.trainStep();
            if (loss >= 0) {
                episode_loss += loss;
                loss_count++;
            }
            
            state = next_state;
            episode_reward += reward;
        }
        
        // Registrar métricas
        metrics.rewards.push_back(episode_reward);
        metrics.lengths.push_back(env.getSteps());
        metrics.losses.push_back(loss_count > 0 ? episode_loss / loss_count : 0.0);
        metrics.epsilons.push_back(agent.getEpsilon());
        
        if (env.getSteps() >= CartPole::MAX_STEPS) {
            metrics.success_count++;
        }
        
        // Imprimir progreso
        if (verbose && (episode + 1) % 50 == 0) {
            int window = std::min(100, episode + 1);
            double avg_reward = std::accumulate(metrics.rewards.end() - window, 
                                                metrics.rewards.end(), 0.0) / window;
            double success_rate = (double)metrics.success_count / (episode + 1) * 100.0;
            
            std::cout << "Ep " << std::setw(4) << (episode + 1) << "/" << num_episodes
                      << " | Reward: " << std::setw(3) << (int)episode_reward
                      << " | Avg: " << std::fixed << std::setprecision(1) << std::setw(6) << avg_reward
                      << " | Success: " << std::setprecision(1) << std::setw(5) << success_rate << "%"
                      << " | ε: " << std::setprecision(3) << agent.getEpsilon()
                      << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    if (verbose) {
        double avg_reward = std::accumulate(metrics.rewards.begin(), metrics.rewards.end(), 0.0) / metrics.rewards.size();
        double success_rate = (double)metrics.success_count / num_episodes * 100.0;
        
        std::cout << "\n" << algo_name << " completado en " << duration.count() << " segundos" << std::endl;
        std::cout << "Episodios: " << num_episodes << std::endl;
        std::cout << "Recompensa promedio: " << std::fixed << std::setprecision(2) << avg_reward << std::endl;
        std::cout << "Tasa de éxito: " << std::setprecision(2) << success_rate << "%" << std::endl;
    }
    
    return metrics;
}

// Guardar métricas en CSV
void saveMetricsCSV(const TrainingMetrics& metrics, const std::string& filename) {
    std::ofstream file(filename);
    file << "episode,reward,length,loss,epsilon\n";
    
    for (size_t i = 0; i < metrics.rewards.size(); ++i) {
        file << (i + 1) << ","
             << metrics.rewards[i] << ","
             << metrics.lengths[i] << ","
             << metrics.losses[i] << ","
             << metrics.epsilons[i] << "\n";
    }
    
    file.close();
    std::cout << "Métricas guardadas en: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "============================================================" << std::endl;
    std::cout << "  Comparación DQN vs Double DQN - Implementación C++" << std::endl;
    std::cout << "  Entorno: CartPole" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    int num_episodes = 500;
    if (argc > 1) {
        num_episodes = std::atoi(argv[1]);
    }
    
    std::cout << "Episodios de entrenamiento: " << num_episodes << std::endl;
    
    // Entrenar DQN
    auto dqn_metrics = trainAgent(false, num_episodes, true);
    saveMetricsCSV(dqn_metrics, "dqn_metrics.csv");
    
    // Entrenar Double DQN
    auto ddqn_metrics = trainAgent(true, num_episodes, true);
    saveMetricsCSV(ddqn_metrics, "ddqn_metrics.csv");
    
    // Resumen comparativo
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  RESUMEN COMPARATIVO" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    double dqn_avg = std::accumulate(dqn_metrics.rewards.begin(), dqn_metrics.rewards.end(), 0.0) / dqn_metrics.rewards.size();
    double ddqn_avg = std::accumulate(ddqn_metrics.rewards.begin(), ddqn_metrics.rewards.end(), 0.0) / ddqn_metrics.rewards.size();
    
    double dqn_max = *std::max_element(dqn_metrics.rewards.begin(), dqn_metrics.rewards.end());
    double ddqn_max = *std::max_element(ddqn_metrics.rewards.begin(), ddqn_metrics.rewards.end());
    
    double dqn_success = (double)dqn_metrics.success_count / num_episodes * 100.0;
    double ddqn_success = (double)ddqn_metrics.success_count / num_episodes * 100.0;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n" << std::setw(25) << "Métrica" 
              << std::setw(15) << "DQN" 
              << std::setw(15) << "Double DQN" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    std::cout << std::setw(25) << "Recompensa promedio" 
              << std::setw(15) << dqn_avg 
              << std::setw(15) << ddqn_avg << std::endl;
    std::cout << std::setw(25) << "Recompensa máxima" 
              << std::setw(15) << dqn_max 
              << std::setw(15) << ddqn_max << std::endl;
    std::cout << std::setw(25) << "Tasa de éxito (%)" 
              << std::setw(15) << dqn_success 
              << std::setw(15) << ddqn_success << std::endl;
    std::cout << std::setw(25) << "Episodios exitosos" 
              << std::setw(15) << dqn_metrics.success_count 
              << std::setw(15) << ddqn_metrics.success_count << std::endl;
    
    std::cout << "\nArchivos generados:" << std::endl;
    std::cout << "  - dqn_metrics.csv" << std::endl;
    std::cout << "  - ddqn_metrics.csv" << std::endl;
    std::cout << "\nUsa 'python visualize.py' para generar gráficos." << std::endl;
    
    return 0;
}
