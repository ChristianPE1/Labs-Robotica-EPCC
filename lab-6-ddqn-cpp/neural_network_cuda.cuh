#ifndef NEURAL_NETWORK_CUDA_CUH
#define NEURAL_NETWORK_CUDA_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

// Macro para verificar errores CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel para inicialización Xavier
__global__ void initWeightsKernel(float* weights, int size, float std_dev, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = curand_normal(&state) * std_dev;
    }
}

// Kernel para forward pass de una capa (matmul + bias + ReLU)
__global__ void forwardLayerKernel(const float* input, const float* weights, const float* bias,
                                   float* output, int input_size, int output_size, bool apply_relu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = bias[idx];
        for (int i = 0; i < input_size; ++i) {
            sum += weights[idx * input_size + i] * input[i];
        }
        output[idx] = apply_relu ? fmaxf(0.0f, sum) : sum;
    }
}

// Kernel para batch forward (múltiples muestras)
__global__ void batchForwardKernel(const float* input, const float* weights, const float* bias,
                                    float* output, float* z_values,
                                    int batch_size, int input_size, int output_size, bool apply_relu) {
    int batch_idx = blockIdx.y;
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && neuron_idx < output_size) {
        float sum = bias[neuron_idx];
        for (int i = 0; i < input_size; ++i) {
            sum += weights[neuron_idx * input_size + i] * input[batch_idx * input_size + i];
        }
        
        int out_idx = batch_idx * output_size + neuron_idx;
        z_values[out_idx] = sum;
        output[out_idx] = apply_relu ? fmaxf(0.0f, sum) : sum;
    }
}

// Kernel para calcular delta de salida (derivada MSE)
__global__ void outputDeltaKernel(const float* output, const float* target, float* delta,
                                   int batch_size, int output_size) {
    int batch_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && idx < output_size) {
        int global_idx = batch_idx * output_size + idx;
        delta[global_idx] = output[global_idx] - target[global_idx];
    }
}

// Kernel para backpropagation de deltas
__global__ void backpropDeltaKernel(const float* delta_next, const float* weights, const float* z_values,
                                     float* delta, int batch_size, int current_size, int next_size) {
    int batch_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && idx < current_size) {
        float error = 0.0f;
        for (int j = 0; j < next_size; ++j) {
            error += delta_next[batch_idx * next_size + j] * weights[j * current_size + idx];
        }
        int global_idx = batch_idx * current_size + idx;
        // ReLU derivative
        delta[global_idx] = z_values[global_idx] > 0 ? error : 0.0f;
    }
}

// Kernel para actualizar pesos
__global__ void updateWeightsKernel(float* weights, const float* deltas, const float* activations,
                                     int batch_size, int input_size, int output_size, float lr) {
    int out_idx = blockIdx.y;
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx < output_size && in_idx < input_size) {
        float grad = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad += deltas[b * output_size + out_idx] * activations[b * input_size + in_idx];
        }
        weights[out_idx * input_size + in_idx] -= lr * grad / batch_size;
    }
}

// Kernel para actualizar biases
__global__ void updateBiasesKernel(float* biases, const float* deltas,
                                    int batch_size, int output_size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size) {
        float grad = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad += deltas[b * output_size + idx];
        }
        biases[idx] -= lr * grad / batch_size;
    }
}

// Kernel para soft update
__global__ void softUpdateKernel(float* target, const float* source, int size, float tau) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        target[idx] = tau * source[idx] + (1.0f - tau) * target[idx];
    }
}

// Kernel para copiar pesos
__global__ void copyWeightsKernel(float* dst, const float* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// Kernel para encontrar argmax
__global__ void argmaxKernel(const float* values, int* result, int size) {
    // Simple single-thread argmax (suficiente para pequeños vectores)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int max_idx = 0;
        float max_val = values[0];
        for (int i = 1; i < size; ++i) {
            if (values[i] > max_val) {
                max_val = values[i];
                max_idx = i;
            }
        }
        *result = max_idx;
    }
}

// Red neuronal con CUDA
class NeuralNetworkCUDA {
private:
    std::vector<int> layers;
    std::vector<float*> d_weights;      // Pesos en GPU
    std::vector<float*> d_biases;       // Biases en GPU
    std::vector<float*> d_activations;  // Activaciones en GPU
    std::vector<float*> d_z_values;     // Pre-activaciones en GPU
    std::vector<float*> d_deltas;       // Deltas para backprop
    
    float learning_rate;
    int max_batch_size;
    
    // Tamaños totales para cada capa
    std::vector<int> weight_sizes;
    std::vector<int> bias_sizes;

public:
    NeuralNetworkCUDA(const std::vector<int>& layer_sizes, float lr = 0.001f, 
                      int max_batch = 64, unsigned int seed = 42)
        : layers(layer_sizes), learning_rate(lr), max_batch_size(max_batch) {
        
        // Calcular tamaños
        for (size_t i = 1; i < layers.size(); ++i) {
            weight_sizes.push_back(layers[i] * layers[i-1]);
            bias_sizes.push_back(layers[i]);
        }
        
        // Alojar memoria en GPU
        for (size_t i = 1; i < layers.size(); ++i) {
            float* w, *b;
            CUDA_CHECK(cudaMalloc(&w, weight_sizes[i-1] * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&b, bias_sizes[i-1] * sizeof(float)));
            d_weights.push_back(w);
            d_biases.push_back(b);
            
            // Inicializar pesos con Xavier
            float std_dev = sqrtf(2.0f / (layers[i-1] + layers[i]));
            int blocks = (weight_sizes[i-1] + 255) / 256;
            initWeightsKernel<<<blocks, 256>>>(w, weight_sizes[i-1], std_dev, seed + i);
            
            // Inicializar biases a 0
            CUDA_CHECK(cudaMemset(b, 0, bias_sizes[i-1] * sizeof(float)));
        }
        
        // Alojar activaciones, z_values y deltas para batch
        for (size_t i = 0; i < layers.size(); ++i) {
            float *act, *z, *delta;
            int size = max_batch_size * layers[i];
            CUDA_CHECK(cudaMalloc(&act, size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&z, size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&delta, size * sizeof(float)));
            d_activations.push_back(act);
            d_z_values.push_back(z);
            d_deltas.push_back(delta);
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    ~NeuralNetworkCUDA() {
        for (auto w : d_weights) cudaFree(w);
        for (auto b : d_biases) cudaFree(b);
        for (auto a : d_activations) cudaFree(a);
        for (auto z : d_z_values) cudaFree(z);
        for (auto d : d_deltas) cudaFree(d);
    }
    
    // Forward pass para un solo estado (retorna Q-values en CPU)
    std::vector<float> forward(const std::vector<float>& input) {
        // Copiar input a GPU
        CUDA_CHECK(cudaMemcpy(d_activations[0], input.data(), 
                              layers[0] * sizeof(float), cudaMemcpyHostToDevice));
        
        // Forward por cada capa
        for (size_t i = 1; i < layers.size(); ++i) {
            bool apply_relu = (i < layers.size() - 1);  // ReLU excepto última capa
            int blocks = (layers[i] + 255) / 256;
            
            forwardLayerKernel<<<blocks, 256>>>(
                d_activations[i-1], d_weights[i-1], d_biases[i-1],
                d_activations[i], layers[i-1], layers[i], apply_relu
            );
        }
        
        // Copiar resultado a CPU
        std::vector<float> output(layers.back());
        CUDA_CHECK(cudaMemcpy(output.data(), d_activations.back(),
                              layers.back() * sizeof(float), cudaMemcpyDeviceToHost));
        
        return output;
    }
    
    // Forward pass para batch (datos ya en GPU)
    void forwardBatch(int batch_size) {
        for (size_t i = 1; i < layers.size(); ++i) {
            bool apply_relu = (i < layers.size() - 1);
            
            dim3 blocks((layers[i] + 255) / 256, batch_size);
            batchForwardKernel<<<blocks, 256>>>(
                d_activations[i-1], d_weights[i-1], d_biases[i-1],
                d_activations[i], d_z_values[i],
                batch_size, layers[i-1], layers[i], apply_relu
            );
        }
    }
    
    // Backward pass para batch
    void backwardBatch(float* d_targets, int batch_size) {
        int num_layers = layers.size();
        
        // Delta de salida
        dim3 blocks1((layers.back() + 255) / 256, batch_size);
        outputDeltaKernel<<<blocks1, 256>>>(
            d_activations.back(), d_targets, d_deltas.back(),
            batch_size, layers.back()
        );
        
        // Backpropagate deltas
        for (int i = num_layers - 2; i >= 1; --i) {
            dim3 blocks((layers[i] + 255) / 256, batch_size);
            backpropDeltaKernel<<<blocks, 256>>>(
                d_deltas[i+1], d_weights[i], d_z_values[i],
                d_deltas[i], batch_size, layers[i], layers[i+1]
            );
        }
        
        // Actualizar pesos y biases
        for (size_t i = 0; i < d_weights.size(); ++i) {
            // Pesos
            dim3 wblocks((layers[i] + 31) / 32, layers[i+1]);
            updateWeightsKernel<<<wblocks, 32>>>(
                d_weights[i], d_deltas[i+1], d_activations[i],
                batch_size, layers[i], layers[i+1], learning_rate
            );
            
            // Biases
            int bblocks = (layers[i+1] + 255) / 256;
            updateBiasesKernel<<<bblocks, 256>>>(
                d_biases[i], d_deltas[i+1], batch_size, layers[i+1], learning_rate
            );
        }
    }
    
    // Soft update desde otra red
    void softUpdate(const NeuralNetworkCUDA& other, float tau) {
        for (size_t i = 0; i < d_weights.size(); ++i) {
            int wblocks = (weight_sizes[i] + 255) / 256;
            softUpdateKernel<<<wblocks, 256>>>(d_weights[i], other.d_weights[i], weight_sizes[i], tau);
            
            int bblocks = (bias_sizes[i] + 255) / 256;
            softUpdateKernel<<<bblocks, 256>>>(d_biases[i], other.d_biases[i], bias_sizes[i], tau);
        }
    }
    
    // Copiar pesos desde otra red
    void copyWeightsFrom(const NeuralNetworkCUDA& other) {
        for (size_t i = 0; i < d_weights.size(); ++i) {
            CUDA_CHECK(cudaMemcpy(d_weights[i], other.d_weights[i], 
                                  weight_sizes[i] * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_biases[i], other.d_biases[i],
                                  bias_sizes[i] * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
    
    // Argmax de Q-values
    int argmax(const std::vector<float>& values) const {
        return std::distance(values.begin(), std::max_element(values.begin(), values.end()));
    }
    
    // Getters
    float* getActivationsDevice(int layer) { return d_activations[layer]; }
    const std::vector<int>& getLayers() const { return layers; }
    void setLearningRate(float lr) { learning_rate = lr; }
};

#endif // NEURAL_NETWORK_CUDA_CUH
