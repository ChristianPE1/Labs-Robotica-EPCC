# Lab 6: DQN vs Double DQN

Implementación en C++ de Deep Q-Network (DQN) y Double DQN para el entorno CartPole.

## Estructura

```
lab-6-ddqn-cpp/
├── cartpole.hpp        # Simulación del entorno CartPole
├── neural_network.hpp  # Red neuronal con backpropagation
├── replay_buffer.hpp   # Buffer de experiencias
├── dqn_agent.hpp       # Agente DQN/Double DQN
├── main.cpp            # Programa principal
├── Makefile            # Compilación
```

## Compilación

```bash
make
```

## Ejecución

```bash
./train 1000
```

El argumento especifica el número de episodios de entrenamiento.

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
| Tau | 0.001 |
| Hidden Layers | [128, 128] |


## Archivos generados

- `dqn_metrics.csv`: Métricas de entrenamiento DQN
- `ddqn_metrics.csv`: Métricas de entrenamiento Double DQN

## Autor

Christian Pardavé Espinoza - UNSA
