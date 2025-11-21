import numpy as np
import matplotlib.pyplot as plt
import os

from utils import load_metrics, compute_statistics


def plot_reward_curve(episode_rewards, save_path='plots/reward_curve.png'):
    # Graficar recompensas por episodio con promedio móvil
    episodes = np.arange(1, len(episode_rewards) + 1)

    # Calcular promedio móvil
    window = 100
    moving_avg, _ = compute_statistics(episode_rewards, window)

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Recompensa del Episodio')
    plt.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'Promedio Móvil de {window} Episodios')

    plt.xlabel('Episodio', fontsize=12)
    plt.ylabel('Recompensa', fontsize=12)
    plt.title('Entrenamiento DQN: Recompensas por Episodio', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")


def plot_episode_length(episode_lengths, save_path='plots/episode_length.png'):
    # Graficar longitudes de episodio con promedio móvil
    episodes = np.arange(1, len(episode_lengths) + 1)

    # Calcular promedio móvil
    window = 100
    moving_avg, _ = compute_statistics(episode_lengths, window)

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_lengths, alpha=0.3, color='green', label='Longitud del Episodio')
    plt.plot(episodes[window-1:], moving_avg, color='orange', linewidth=2, label=f'Promedio Móvil de {window} Episodios')

    plt.xlabel('Episodio', fontsize=12)
    plt.ylabel('Longitud (Pasos)', fontsize=12)
    plt.title('Entrenamiento DQN: Longitudes de Episodio', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")


def plot_loss_curve(episode_losses, save_path='plots/loss_curve.png'):
    # Graficar pérdida de entrenamiento con promedio móvil
    episodes = np.arange(1, len(episode_losses) + 1)

    # Calcular promedio móvil
    window = 100
    moving_avg, _ = compute_statistics(episode_losses, window)

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_losses, alpha=0.3, color='purple', label='Pérdida del Episodio')
    plt.plot(episodes[window-1:], moving_avg, color='darkred', linewidth=2, label=f'Promedio Móvil de {window} Episodios')

    plt.xlabel('Episodio', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.title('Entrenamiento DQN: Curva de Pérdida', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")


def plot_success_rate(episode_lengths, threshold=500, save_path='plots/success_rate.png'):
    # Graficar tasa de éxito a lo largo del tiempo (episodios que alcanzan longitud umbral)
    # episode_lengths: Lista de longitudes de episodio
    # threshold: Longitud mínima para considerar éxito
    # save_path: Ruta para guardar la gráfica
    episodes = np.arange(1, len(episode_lengths) + 1)

    # Calcular tasa de éxito con ventana móvil
    window = 100
    success_rates = []

    for i in range(len(episode_lengths)):
        start = max(0, i - window + 1)
        window_lengths = episode_lengths[start:i+1]
        success_count = sum(1 for length in window_lengths if length >= threshold)
        success_rate = (success_count / len(window_lengths)) * 100
        success_rates.append(success_rate)

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, success_rates, color='teal', linewidth=2)
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% Éxito')

    plt.xlabel('Episodio', fontsize=12)
    plt.ylabel('Tasa de Éxito (%)', fontsize=12)
    plt.title(f'Entrenamiento DQN: Tasa de Éxito (Longitud >= {threshold})', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")


def plot_all_metrics(metrics):
    # Generar todas las gráficas de entrenamiento
    print("Generando visualizaciones de entrenamiento...")
    print("-" * 60)

    episode_rewards = metrics['episode_rewards']
    episode_lengths = metrics['episode_lengths']
    episode_losses = metrics['episode_losses']

    # Generar gráficas
    plot_reward_curve(episode_rewards)
    plot_episode_length(episode_lengths)
    plot_loss_curve(episode_losses)
    plot_success_rate(episode_lengths)

    print("-" * 60)
    print("Todas las gráficas generadas exitosamente!")


def print_summary_statistics(metrics):
    # Imprimir estadísticas resumidas del entrenamiento
    episode_rewards = metrics['episode_rewards']
    episode_lengths = metrics['episode_lengths']
    episode_losses = metrics['episode_losses']

    print("\nEstadísticas Resumidas de Entrenamiento:")
    print("-" * 60)

    print(f"Episodios entrenados: {len(episode_rewards)}")

    print("\nRecompensas:")
    print(f"  Media: {np.mean(episode_rewards):.2f}")
    print(f"  Desviación estándar: {np.std(episode_rewards):.2f}")
    print(f"  Mínimo: {np.min(episode_rewards):.2f}")
    print(f"  Máximo: {np.max(episode_rewards):.2f}")

    print("\nLongitudes de Episodio:")
    print(f"  Media: {np.mean(episode_lengths):.2f}")
    print(f"  Desviación estándar: {np.std(episode_lengths):.2f}")
    print(f"  Mínimo: {np.min(episode_lengths)}")
    print(f"  Máximo: {np.max(episode_lengths)}")

    print("\nPérdidas:")
    print(f"  Media: {np.mean(episode_losses):.4f}")
    print(f"  Desviación estándar: {np.std(episode_losses):.4f}")
    print(f"  Mínimo: {np.min(episode_losses):.4f}")
    print(f"  Máximo: {np.max(episode_losses):.4f}")

    # Tasa de éxito
    success_count = sum(1 for length in episode_lengths if length >= 500)
    success_rate = (success_count / len(episode_lengths)) * 100
    print(f"\nTasa de Éxito (Longitud >= 500): {success_rate:.2f}%")

    # Estadísticas de los últimos 100 episodios
    if len(episode_rewards) >= 100:
        last_100_rewards = episode_rewards[-100:]
        last_100_lengths = episode_lengths[-100:]
        last_100_success = sum(1 for length in last_100_lengths if length >= 500)

        print("\nÚltimos 100 Episodios:")
        print(f"  Recompensa media: {np.mean(last_100_rewards):.2f}")
        print(f"  Longitud media: {np.mean(last_100_lengths):.2f}")
        print(f"  Tasa de éxito: {(last_100_success / 100) * 100:.2f}%")


if __name__ == "__main__":
    metrics_file = "metrics/training_metrics.pkl"

    if not os.path.exists(metrics_file):
        print(f"Archivo de métricas no encontrado: {metrics_file}")
        exit(1)

    print(f"Cargando métricas desde: {metrics_file}")
    metrics = load_metrics(metrics_file)

    # Imprimir estadísticas resumidas
    print_summary_statistics(metrics)

    # Generar gráficas
    plot_all_metrics(metrics)

    print("\nVisualización completada!")
