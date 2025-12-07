import gymnasium as gym
import numpy as np
import os
import argparse

import config
from dqn_agent import DQNAgent
from utils import (
    save_checkpoint, save_metrics, set_random_seed,
    print_training_progress, get_device_info
)


def train_agent(use_double_dqn=False, verbose=True):
    algorithm_name = "Double DQN" if use_double_dqn else "DQN"
    algorithm_prefix = "ddqn" if use_double_dqn else "dqn"
    
    # Configurar semilla para reproducibilidad
    set_random_seed(config.RANDOM_SEED)
    
    # Información del dispositivo
    if verbose:
        print(f"Entrenando {algorithm_name}")
        device_info = get_device_info()
        print(f"Dispositivo: {device_info['device_name']}")
        print(f"CUDA disponible: {device_info['cuda_available']}")
        if device_info['cuda_available']:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    
    # Crear entorno
    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if verbose:
        print(f"Entorno: {config.ENV_NAME}")
        print(f"Dimensión del estado: {state_dim}")
        print(f"Dimensión de la acción: {action_dim}")
        print(f"Episodios: {config.NUM_EPISODES}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Memory size: {config.MEMORY_SIZE}")
        print(f"Learning rate: {config.LEARNING_RATE}")
    
    # Inicializar agente
    agent = DQNAgent(
        state_dim, action_dim, config, config.DEVICE,
        use_double_dqn=use_double_dqn
    )
    
    # Métricas de entrenamiento
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    episode_q_values = []  # Para comparar sobreestimación
    success_count = 0
    
    # Early stopping
    early_stop_counter = 0
    
    if verbose:
        print("Iniciando entrenamiento...")
        print("Llenando buffer de experiencias...")
    
    # Llenar buffer inicialmente con acciones aleatorias
    state, _ = env.reset()
    for _ in range(config.BATCH_SIZE):
        action = env.action_space.sample()  # Acción aleatoria
        next_state, reward, done, truncated, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done or truncated)
        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state
    
    if verbose:
        print(f"Buffer inicial lleno con {len(agent.memory)} transiciones.")
        print(f"Red en dispositivo: {next(agent.policy_net.parameters()).device}")
        print(f"Optimizador en dispositivo: {agent.optimizer.param_groups[0]['params'][0].device}")
    
    
    # Bucle de entrenamiento
    for episode in range(config.NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss_values = []
        episode_q_vals = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Obtener Q-values para análisis
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)
                q_values = agent.policy_net(state_tensor)
                max_q = q_values.max().item()
                episode_q_vals.append(max_q)
            
            # Seleccionar acción
            action = agent.select_action(state, training=True)
            
            # Ejecutar acción
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Almacenar transición
            agent.store_transition(state, action, reward, next_state, done or truncated)
            
            # Entrenar
            loss = agent.train_step()
            if loss is not None:
                episode_loss_values.append(loss)
                if episode % 100 == 0:  # Debug cada 100 episodios
                    print(f"Debug: Loss = {loss:.4f}, Buffer size = {len(agent.memory)}, Epsilon = {agent.epsilon:.4f}")
            
            # Actualizar estado y métricas
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Registrar métricas
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        avg_loss = np.mean(episode_loss_values) if episode_loss_values else 0.0
        episode_losses.append(avg_loss)
        
        avg_q = np.mean(episode_q_vals) if episode_q_vals else 0.0
        episode_q_values.append(avg_q)
        
        # Contar éxitos
        if episode_length >= 500:
            success_count += 1
        
        # Actualizar red objetivo periódicamente
        if (episode + 1) % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        # Calcular promedios móviles
        window = min(100, episode + 1)
        avg_reward = np.mean(episode_rewards[-window:])
        avg_length = np.mean(episode_lengths[-window:])
        
        # Early stopping
        if avg_reward >= config.EARLY_STOP_THRESHOLD:
            early_stop_counter += 1
            if early_stop_counter >= config.EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"\nEarly stopping en episodio {episode + 1}")
                    print(f"Recompensa promedio: {avg_reward:.2f}")
                break
        else:
            early_stop_counter = 0
        
        # Imprimir progreso
        if verbose and (episode + 1) % config.PRINT_EVERY == 0:
            success_rate = (success_count / (episode + 1)) * 100
            
            # Calcular uso de memoria GPU
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1e6  # MB
                mem_reserved = torch.cuda.memory_reserved() / 1e6  # MB
                gpu_info = f"| GPU: {mem_allocated:.0f}/{mem_reserved:.0f}MB"
            else:
                gpu_info = ""
            
            print(f"Ep {episode+1:>4}/{config.NUM_EPISODES} | "
                  f"R: {episode_reward:>3.0f} | "
                  f"AvgR: {avg_reward:>6.1f} | "
                  f"L: {episode_length:>3} | "
                  f"AvgL: {avg_length:>6.1f} | "
                  f"Loss: {avg_loss:>6.4f} | "
                  f"ε: {agent.epsilon:.4f} | "
                  f"Success: {success_rate:>5.1f}% {gpu_info}")
        
        # Guardar checkpoint
        if (episode + 1) % config.SAVE_EVERY == 0:
            checkpoint_path = f"checkpoints/{algorithm_prefix}_episode_{episode + 1}.pt"
            metrics = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'episode_losses': episode_losses,
                'episode_q_values': episode_q_values,
                'episode': episode + 1
            }
            save_checkpoint(agent.policy_net, agent.optimizer, episode + 1, metrics, checkpoint_path)
            if verbose:
                print(f"Checkpoint guardado: {checkpoint_path}")
    
    env.close()
    
    # Guardar modelo y métricas finales
    total_episodes = len(episode_rewards)
    
    final_checkpoint = f"checkpoints/{algorithm_prefix}_final.pt"
    final_metrics = {
        'algorithm': algorithm_name,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_losses': episode_losses,
        'episode_q_values': episode_q_values,
        'total_episodes': total_episodes,
        'success_count': success_count,
        'final_epsilon': agent.epsilon
    }
    save_checkpoint(agent.policy_net, agent.optimizer, total_episodes, final_metrics, final_checkpoint)
    
    metrics_file = f"metrics/{algorithm_prefix}_metrics.pkl"
    save_metrics(final_metrics, metrics_file)
    
    if verbose:
        print(f"\nEntrenamiento {algorithm_name} completado!")
        print(f"Episodios entrenados: {total_episodes}")
        print(f"Recompensa promedio: {np.mean(episode_rewards):.2f}")
        print(f"Longitud promedio: {np.mean(episode_lengths):.1f}")
        print(f"Tasa de éxito: {(success_count / total_episodes) * 100:.2f}%")
        print(f"Modelo guardado: {final_checkpoint}")
        print(f"Métricas guardadas: {metrics_file}")
    
    return final_metrics


# Importar torch aquí para evitar problemas de importación circular
import torch


def main():
    """Función principal para entrenar ambos algoritmos."""
    parser = argparse.ArgumentParser(description='Entrenar DQN o Double DQN')
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'ddqn', 'both'],
                        default='both', help='Algoritmo a entrenar')
    args = parser.parse_args()
    
    # Crear directorios
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    if args.algorithm == 'dqn':
        train_agent(use_double_dqn=False)
    elif args.algorithm == 'ddqn':
        train_agent(use_double_dqn=True)
    else:
        # Entrenar ambos
        print("\nENTRENAMIENTO COMPARATIVO: DQN vs Double DQN")
        
        # Entrenar DQN
        dqn_metrics = train_agent(use_double_dqn=False)
        
        # Entrenar Double DQN
        ddqn_metrics = train_agent(use_double_dqn=True)
        
        # Resumen comparativo
        print("\nRESUMEN COMPARATIVO")
        dqn_success = (dqn_metrics['success_count'] / dqn_metrics['total_episodes']) * 100
        ddqn_success = (ddqn_metrics['success_count'] / ddqn_metrics['total_episodes']) * 100
        
        print(f"\n{'Métrica':<30} {'DQN':>15} {'Double DQN':>15}")
        print(f"{'Episodios entrenados':<30} {dqn_metrics['total_episodes']:>15} {ddqn_metrics['total_episodes']:>15}")
        print(f"{'Recompensa promedio':<30} {np.mean(dqn_metrics['episode_rewards']):>15.2f} {np.mean(ddqn_metrics['episode_rewards']):>15.2f}")
        print(f"{'Recompensa máxima':<30} {np.max(dqn_metrics['episode_rewards']):>15.2f} {np.max(ddqn_metrics['episode_rewards']):>15.2f}")
        print(f"{'Longitud promedio':<30} {np.mean(dqn_metrics['episode_lengths']):>15.1f} {np.mean(ddqn_metrics['episode_lengths']):>15.1f}")
        print(f"{'Tasa de éxito':<30} {dqn_success:>14.2f}% {ddqn_success:>14.2f}%")
        print(f"{'Q-value promedio':<30} {np.mean(dqn_metrics['episode_q_values']):>15.2f} {np.mean(ddqn_metrics['episode_q_values']):>15.2f}")


if __name__ == "__main__":
    main()
