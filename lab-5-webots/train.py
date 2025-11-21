# Script de entrenamiento DQN para entornos de Gymnasium
# Este script entrena un agente DQN en un entorno de Gymnasium con soporte para CUDA.
# El progreso del entrenamiento y los modelos se guardan periódicamente

import gymnasium as gym
import numpy as np
import sys
import os

import config
from dqn_agent import DQNAgent
from utils import save_checkpoint, save_metrics, compute_statistics, get_device_info


def train_dqn():
    # Entrenar agente DQN en el entorno configurado de Gymnasium
    # Mostrar información del dispositivo
    device_info = get_device_info()
    print("Información del dispositivo:")
    print(f"  Usando dispositivo: {config.DEVICE}")
    print(f"  CUDA disponible: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        print(f"  GPU: {device_info['device_name']}")
    print()

    # Crear entorno
    env = gym.make(config.ENV_NAME)

    # Obtener dimensiones del entorno
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Entorno: {config.ENV_NAME}")
    print(f"Dimensión del estado: {state_dim}")
    print(f"Dimensión de la acción: {action_dim}")
    print(f"Episodios: {config.NUM_EPISODES}")
    print()

    # Inicializar agente
    agent = DQNAgent(state_dim, action_dim, config, config.DEVICE)

    # Métricas de entrenamiento
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    success_count = 0

    # Bucle de entrenamiento
    print("Iniciando entrenamiento...")
    print("-" * 60)

    for episode in range(config.NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss_values = []

        done = False
        truncated = False

        while not (done or truncated):
            # Seleccionar acción
            action = agent.select_action(state, training=True)

            # Ejecutar acción
            next_state, reward, done, truncated, info = env.step(action)

            # Almacenar transición
            agent.store_transition(state, action, reward, next_state, done or truncated)

            # Entrenar agente
            loss = agent.train_step()
            if loss is not None:
                episode_loss_values.append(loss)

            # Actualizar estado y métricas
            state = next_state
            episode_reward += reward
            episode_length += 1

        # Registrar métricas del episodio
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        avg_loss = np.mean(episode_loss_values) if episode_loss_values else 0.0
        episode_losses.append(avg_loss)

        # Contar éxitos (para CartPole, éxito es alcanzar pasos máximos)
        if episode_length >= 500:
            success_count += 1

        # Calcular promedios móviles
        window = min(100, episode + 1)
        avg_reward = np.mean(episode_rewards[-window:])
        avg_length = np.mean(episode_lengths[-window:])

        # Imprimir progreso
        if (episode + 1) % config.PRINT_EVERY == 0:
            success_rate = (success_count / (episode + 1)) * 100
            print(f"Episodio {episode + 1}/{config.NUM_EPISODES}")
            print(f"  Recompensa: {episode_reward:.2f} | Promedio: {avg_reward:.2f}")
            print(f"  Longitud: {episode_length} | Promedio: {avg_length:.1f}")
            print(f"  Pérdida: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f}")
            print(f"  Tasa de éxito: {success_rate:.2f}%")
            print("-" * 60)

        # Guardar checkpoint
        if (episode + 1) % config.SAVE_EVERY == 0:
            checkpoint_path = f"checkpoints/dqn_episode_{episode + 1}.pt"
            metrics = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'episode_losses': episode_losses,
                'episode': episode + 1
            }
            save_checkpoint(agent, metrics, checkpoint_path)
            print(f"Checkpoint guardado: {checkpoint_path}")
            print("-" * 60)

    # Guardar modelo final y métricas
    print("\nEntrenamiento completado!")

    final_checkpoint = "checkpoints/dqn_final.pt"
    final_metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_losses': episode_losses,
        'episode': config.NUM_EPISODES
    }
    save_checkpoint(agent, final_metrics, final_checkpoint)
    print(f"Modelo final guardado: {final_checkpoint}")

    metrics_file = "metrics/training_metrics.pkl"
    save_metrics(final_metrics, metrics_file)
    print(f"Métricas de entrenamiento guardadas: {metrics_file}")

    # Imprimir estadísticas finales
    print("\nEstadísticas de entrenamiento:")
    print(f"  Recompensa promedio: {np.mean(episode_rewards):.2f}")
    print(f"  Longitud promedio del episodio: {np.mean(episode_lengths):.1f}")
    print(f"  Tasa de éxito: {(success_count / config.NUM_EPISODES) * 100:.2f}%")
    print(f"  Epsilon final: {agent.epsilon:.4f}")

    env.close()


if __name__ == "__main__":
    # Create directories for checkpoints and metrics
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    train_dqn()
