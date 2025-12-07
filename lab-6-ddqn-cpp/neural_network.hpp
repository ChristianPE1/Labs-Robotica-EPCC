#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

// Red neuronal simple con backpropagation
class NeuralNetwork {
private:
    std::vector<int> layers;
    std::vector<std::vector<std::vector<double>>> weights;  // [capa][neurona_destino][neurona_origen]
    std::vector<std::vector<double>> biases;                 // [capa][neurona]
    std::vector<std::vector<double>> activations;            // Para forward pass
    std::vector<std::vector<double>> z_values;               // Pre-activaciones
    double learning_rate;
    std::mt19937 rng;

public:
    NeuralNetwork(const std::vector<int>& layer_sizes, double lr = 0.001, unsigned int seed = 42)
        : layers(layer_sizes), learning_rate(lr), rng(seed) {
        
        // Inicializar pesos con Xavier initialization
        std::normal_distribution<double> dist(0.0, 1.0);
        
        for (size_t i = 1; i < layers.size(); ++i) {
            int fan_in = layers[i-1];
            int fan_out = layers[i];
            double std_dev = std::sqrt(2.0 / (fan_in + fan_out));
            
            std::vector<std::vector<double>> layer_weights;
            std::vector<double> layer_biases;
            
            for (int j = 0; j < layers[i]; ++j) {
                std::vector<double> neuron_weights;
                for (int k = 0; k < layers[i-1]; ++k) {
                    neuron_weights.push_back(dist(rng) * std_dev);
                }
                layer_weights.push_back(neuron_weights);
                layer_biases.push_back(0.0);
            }
            
            weights.push_back(layer_weights);
            biases.push_back(layer_biases);
        }
        
        // Inicializar vectores de activaciones
        for (size_t i = 0; i < layers.size(); ++i) {
            activations.push_back(std::vector<double>(layers[i], 0.0));
            z_values.push_back(std::vector<double>(layers[i], 0.0));
        }
    }
    
    // Función de activación ReLU
    static double relu(double x) {
        return std::max(0.0, x);
    }
    
    static double relu_derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
    
    // Forward pass
    std::vector<double> forward(const std::vector<double>& input) {
        activations[0] = input;
        z_values[0] = input;
        
        for (size_t layer = 1; layer < layers.size(); ++layer) {
            for (int j = 0; j < layers[layer]; ++j) {
                double sum = biases[layer-1][j];
                for (int k = 0; k < layers[layer-1]; ++k) {
                    sum += weights[layer-1][j][k] * activations[layer-1][k];
                }
                z_values[layer][j] = sum;
                
                // ReLU para capas ocultas, lineal para salida
                if (layer < layers.size() - 1) {
                    activations[layer][j] = relu(sum);
                } else {
                    activations[layer][j] = sum;  // Salida lineal para Q-values
                }
            }
        }
        
        return activations.back();
    }
    
    // Backward pass con MSE loss
    void backward(const std::vector<double>& target) {
        int num_layers = layers.size();
        std::vector<std::vector<double>> deltas(num_layers);
        
        // Calcular delta de la capa de salida (derivada de MSE)
        deltas[num_layers-1].resize(layers[num_layers-1]);
        for (int j = 0; j < layers[num_layers-1]; ++j) {
            deltas[num_layers-1][j] = activations[num_layers-1][j] - target[j];
        }
        
        // Backpropagate deltas
        for (int layer = num_layers - 2; layer >= 1; --layer) {
            deltas[layer].resize(layers[layer]);
            for (int j = 0; j < layers[layer]; ++j) {
                double error = 0.0;
                for (int k = 0; k < layers[layer+1]; ++k) {
                    error += deltas[layer+1][k] * weights[layer][k][j];
                }
                deltas[layer][j] = error * relu_derivative(z_values[layer][j]);
            }
        }
        
        // Actualizar pesos y biases
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            for (int j = 0; j < layers[layer+1]; ++j) {
                for (int k = 0; k < layers[layer]; ++k) {
                    weights[layer][j][k] -= learning_rate * deltas[layer+1][j] * activations[layer][k];
                }
                biases[layer][j] -= learning_rate * deltas[layer+1][j];
            }
        }
    }
    
    // Entrenar con un batch
    double trainBatch(const std::vector<std::vector<double>>& inputs,
                      const std::vector<std::vector<double>>& targets) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> output = forward(inputs[i]);
            
            // Calcular loss MSE
            for (size_t j = 0; j < output.size(); ++j) {
                double diff = output[j] - targets[i][j];
                total_loss += diff * diff;
            }
            
            backward(targets[i]);
        }
        
        return total_loss / inputs.size();
    }
    
    // Copiar pesos de otra red
    void copyWeightsFrom(const NeuralNetwork& other) {
        weights = other.weights;
        biases = other.biases;
    }
    
    // Soft update: θ_target = τ*θ + (1-τ)*θ_target
    void softUpdate(const NeuralNetwork& other, double tau) {
        for (size_t l = 0; l < weights.size(); ++l) {
            for (size_t j = 0; j < weights[l].size(); ++j) {
                for (size_t k = 0; k < weights[l][j].size(); ++k) {
                    weights[l][j][k] = tau * other.weights[l][j][k] + (1.0 - tau) * weights[l][j][k];
                }
                biases[l][j] = tau * other.biases[l][j] + (1.0 - tau) * biases[l][j];
            }
        }
    }
    
    // Obtener índice de la acción con mayor Q-value
    int argmax(const std::vector<double>& q_values) const {
        return std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end()));
    }
    
    void setLearningRate(double lr) { learning_rate = lr; }
    double getLearningRate() const { return learning_rate; }
};

#endif // NEURAL_NETWORK_HPP
