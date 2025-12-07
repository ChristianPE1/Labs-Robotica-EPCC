import os
import pickle
import numpy as np
import torch


def save_checkpoint(model, optimizer, episode, metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode'], checkpoint['metrics']


def save_metrics(metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(metrics, f)


def load_metrics(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def compute_moving_average(data, window=100):
    # Calcular promedio móvil
    if len(data) < window:
        return np.array(data)
    
    moving_avg = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        window_data = data[start:i+1]
        moving_avg.append(np.mean(window_data))
    
    return np.array(moving_avg)


def compute_statistics(data, window=100):
    # Calcular estadísticas con ventana móvil
    moving_avg = compute_moving_average(data, window)
    
    moving_std = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        window_data = data[start:i+1]
        moving_std.append(np.std(window_data))
    
    return moving_avg, np.array(moving_std)


def get_device_info():
    if torch.cuda.is_available():
        return {
            'cuda_available': True,
            'device_name': torch.cuda.get_device_name(0),
            'device_memory': torch.cuda.get_device_properties(0).total_memory / 1e9
        }
    return {
        'cuda_available': False,
        'device_name': 'CPU',
        'device_memory': 0.0
    }


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def print_training_progress(episode, total_episodes, reward, avg_reward, 
                            length, avg_length, loss, epsilon, success_rate, algorithm):
    print(f"[{algorithm}] Episodio {episode}/{total_episodes}")
    print(f"  Recompensa: {reward:.2f} | Promedio: {avg_reward:.2f}")
    print(f"  Longitud: {length} | Promedio: {avg_length:.1f}")
    print(f"  Pérdida: {loss:.4f} | Epsilon: {epsilon:.4f}")
    print(f"  Tasa de éxito: {success_rate:.2f}%\n")


def evaluate_agent(agent, env, num_episodes=10):
    rewards = []
    lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
    
    return {
        'rewards': rewards,
        'lengths': lengths,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'success_rate': sum(1 for l in lengths if l >= 500) / num_episodes * 100
    }
