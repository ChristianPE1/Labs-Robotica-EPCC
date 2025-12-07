#ifndef CARTPOLE_HPP
#define CARTPOLE_HPP

#include <cmath>
#include <random>
#include <array>

// Simulación del entorno CartPole
// Basado en la implementación de OpenAI Gym
class CartPole {
public:
    static constexpr int STATE_DIM = 4;
    static constexpr int ACTION_DIM = 2;
    static constexpr int MAX_STEPS = 500;
    
    // Parámetros físicos
    static constexpr double GRAVITY = 9.8;
    static constexpr double CART_MASS = 1.0;
    static constexpr double POLE_MASS = 0.1;
    static constexpr double TOTAL_MASS = CART_MASS + POLE_MASS;
    static constexpr double POLE_LENGTH = 0.5;
    static constexpr double POLE_MASS_LENGTH = POLE_MASS * POLE_LENGTH;
    static constexpr double FORCE_MAG = 10.0;
    static constexpr double TAU = 0.02;  // Tiempo entre updates
    
    // Límites
    static constexpr double X_THRESHOLD = 2.4;
    static constexpr double THETA_THRESHOLD = 12.0 * M_PI / 180.0;  // 12 grados en radianes

private:
    std::array<double, STATE_DIM> state;
    int steps;
    bool done;
    std::mt19937 rng;
    std::uniform_real_distribution<double> init_dist;

public:
    CartPole(unsigned int seed = 42) 
        : steps(0), done(false), rng(seed), init_dist(-0.05, 0.05) {
        reset();
    }
    
    std::array<double, STATE_DIM> reset() {
        // Estado inicial aleatorio pequeño
        for (int i = 0; i < STATE_DIM; ++i) {
            state[i] = init_dist(rng);
        }
        steps = 0;
        done = false;
        return state;
    }
    
    // Ejecutar una acción (0: izquierda, 1: derecha)
    std::tuple<std::array<double, STATE_DIM>, double, bool> step(int action) {
        double x = state[0];
        double x_dot = state[1];
        double theta = state[2];
        double theta_dot = state[3];
        
        // Fuerza aplicada
        double force = (action == 1) ? FORCE_MAG : -FORCE_MAG;
        
        // Física del péndulo invertido
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        
        double temp = (force + POLE_MASS_LENGTH * theta_dot * theta_dot * sin_theta) / TOTAL_MASS;
        double theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / 
                          (POLE_LENGTH * (4.0/3.0 - POLE_MASS * cos_theta * cos_theta / TOTAL_MASS));
        double x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;
        
        // Integración de Euler
        x = x + TAU * x_dot;
        x_dot = x_dot + TAU * x_acc;
        theta = theta + TAU * theta_dot;
        theta_dot = theta_dot + TAU * theta_acc;
        
        // Actualizar estado
        state[0] = x;
        state[1] = x_dot;
        state[2] = theta;
        state[3] = theta_dot;
        
        steps++;
        
        // Verificar terminación
        done = (x < -X_THRESHOLD || x > X_THRESHOLD ||
                theta < -THETA_THRESHOLD || theta > THETA_THRESHOLD ||
                steps >= MAX_STEPS);
        
        // Recompensa: +1 por cada paso que sobrevive
        double reward = done ? 0.0 : 1.0;
        
        // Si llegó al máximo de pasos, es éxito
        if (steps >= MAX_STEPS) {
            reward = 1.0;
        }
        
        return {state, reward, done};
    }
    
    bool isDone() const { return done; }
    int getSteps() const { return steps; }
    const std::array<double, STATE_DIM>& getState() const { return state; }
};

#endif // CARTPOLE_HPP
