# Lab 6: DQN vs Double DQN - Implementación C++

Comparación de Deep Q-Network (DQN) y Double DQN implementados en C++ puro, con soporte opcional para CUDA.

## Estructura del Proyecto

```
lab-6-ddqn-cpp/
├── cartpole.hpp           # Simulación del entorno CartPole
├── neural_network.hpp     # Red neuronal (CPU)
├── neural_network_cuda.cuh # Red neuronal (CUDA)
├── replay_buffer.hpp      # Buffer de experiencias
├── dqn_agent.hpp          # Agente DQN/DDQN (CPU)
├── dqn_agent_cuda.cuh     # Agente DQN/DDQN (CUDA)
├── main.cpp               # Programa principal (CPU)
├── main_cuda.cu           # Programa principal (CUDA)
├── Makefile               # Compilación CPU
├── Makefile.cuda          # Compilación CUDA
├── visualize.py           # Visualización de resultados
└── DQN_CUDA_Colab.ipynb   # Notebook para Google Colab
```

## Compilación y Ejecución

### Versión CPU (local)

```bash
make
./train 1000  # 1000 episodios
```

### Versión CUDA (Google Colab)

1. Subir el notebook `DQN_CUDA_Colab.ipynb` a Google Colab
2. Activar GPU: Runtime > Change runtime type > GPU
3. Ejecutar todas las celdas

O manualmente:

```bash
# Compilar
nvcc -std=c++17 -O3 -arch=sm_75 -o train_cuda main_cuda.cu

# Ejecutar (5000 episodios recomendado para CUDA)
./train_cuda 5000
```

## Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Learning Rate | 0.001 |
| Gamma | 0.99 |
| Epsilon Start | 1.0 |
| Epsilon End | 0.05 |
| Epsilon Decay | 0.9995 |
| Batch Size | 64 |
| Memory Size | 50000 |
| Tau (soft update) | 0.001 |
| Hidden Layers | [128, 128] |

## Visualización

```bash
python3 visualize.py
```

Genera:
- `comparison_rewards.png` - Comparación de recompensas
- `comparison_success.png` - Tasas de éxito
- `comparison_loss.png` - Pérdidas durante entrenamiento

## Resultados

Ver los archivos CSV generados:
- `dqn_metrics.csv` - Métricas de entrenamiento DQN
- `ddqn_metrics.csv` - Métricas de entrenamiento Double DQN

## Autor

Christian Peñaranda - Universidad Nacional de San Agustín
