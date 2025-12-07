import numpy as np
import matplotlib.pyplot as plt
import os

from utils import load_metrics, compute_moving_average


# Configuración global de matplotlib
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def plot_reward_comparison(dqn_metrics, ddqn_metrics, save_path='plots/reward_comparison.png'):
    # Graficar comparación de recompensas entre DQN y Double DQN.

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica 1: Recompensas por episodio
    ax1 = axes[0]
    
    dqn_rewards = dqn_metrics['episode_rewards']
    ddqn_rewards = ddqn_metrics['episode_rewards']
    
    dqn_episodes = np.arange(1, len(dqn_rewards) + 1)
    ddqn_episodes = np.arange(1, len(ddqn_rewards) + 1)
    
    # Promedios móviles
    dqn_ma = compute_moving_average(dqn_rewards, 50)
    ddqn_ma = compute_moving_average(ddqn_rewards, 50)
    
    ax1.plot(dqn_episodes, dqn_rewards, alpha=0.2, color='blue')
    ax1.plot(dqn_episodes, dqn_ma, color='blue', linewidth=2, label='DQN')
    
    ax1.plot(ddqn_episodes, ddqn_rewards, alpha=0.2, color='red')
    ax1.plot(ddqn_episodes, ddqn_ma, color='red', linewidth=2, label='Double DQN')
    
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Curva de Aprendizaje: Recompensas')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Box plot de recompensas
    ax2 = axes[1]
    
    box_data = [dqn_rewards, ddqn_rewards]
    bp = ax2.boxplot(box_data, labels=['DQN', 'Double DQN'], patch_artist=True)
    
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Recompensa')
    ax2.set_title('Distribución de Recompensas')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")


def plot_q_value_comparison(dqn_metrics, ddqn_metrics, save_path='plots/q_value_comparison.png'):
    # Graficar comparación de Q-values para analizar sobreestimación.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    dqn_q_values = dqn_metrics['episode_q_values']
    ddqn_q_values = ddqn_metrics['episode_q_values']
    
    dqn_episodes = np.arange(1, len(dqn_q_values) + 1)
    ddqn_episodes = np.arange(1, len(ddqn_q_values) + 1)
    
    # Gráfica 1: Q-values por episodio
    ax1 = axes[0]
    
    dqn_ma = compute_moving_average(dqn_q_values, 50)
    ddqn_ma = compute_moving_average(ddqn_q_values, 50)
    
    ax1.plot(dqn_episodes, dqn_q_values, alpha=0.2, color='blue')
    ax1.plot(dqn_episodes, dqn_ma, color='blue', linewidth=2, label='DQN')
    
    ax1.plot(ddqn_episodes, ddqn_q_values, alpha=0.2, color='red')
    ax1.plot(ddqn_episodes, ddqn_ma, color='red', linewidth=2, label='Double DQN')
    
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Q-Value Promedio')
    ax1.set_title('Evolución de Q-Values (Análisis de Sobreestimación)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Histograma de Q-values
    ax2 = axes[1]
    
    ax2.hist(dqn_q_values, bins=30, alpha=0.5, color='blue', label='DQN', density=True)
    ax2.hist(ddqn_q_values, bins=30, alpha=0.5, color='red', label='Double DQN', density=True)
    
    ax2.axvline(np.mean(dqn_q_values), color='blue', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(ddqn_q_values), color='red', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Q-Value')
    ax2.set_ylabel('Densidad')
    ax2.set_title('Distribución de Q-Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")


def plot_loss_comparison(dqn_metrics, ddqn_metrics, save_path='plots/loss_comparison.png'):
    # Graficar comparación de pérdidas entre DQN y Double DQN.
    fig, ax = plt.subplots(figsize=(12, 5))
    
    dqn_losses = dqn_metrics['episode_losses']
    ddqn_losses = ddqn_metrics['episode_losses']
    
    dqn_episodes = np.arange(1, len(dqn_losses) + 1)
    ddqn_episodes = np.arange(1, len(ddqn_losses) + 1)
    
    # Promedios móviles
    dqn_ma = compute_moving_average(dqn_losses, 50)
    ddqn_ma = compute_moving_average(ddqn_losses, 50)
    
    ax.plot(dqn_episodes, dqn_losses, alpha=0.2, color='blue')
    ax.plot(dqn_episodes, dqn_ma, color='blue', linewidth=2, label='DQN')
    
    ax.plot(ddqn_episodes, ddqn_losses, alpha=0.2, color='red')
    ax.plot(ddqn_episodes, ddqn_ma, color='red', linewidth=2, label='Double DQN')
    
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Pérdida')
    ax.set_title('Curva de Pérdida: DQN vs Double DQN')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Escala logarítmica para mejor visualización
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")


def plot_success_rate_comparison(dqn_metrics, ddqn_metrics, save_path='plots/success_rate_comparison.png'):
    # Graficar comparación de tasas de éxito.
    fig, ax = plt.subplots(figsize=(12, 5))
    
    def compute_success_rate(lengths, window=100):
        # Calcular tasa de éxito con ventana móvil.
        success_rates = []
        for i in range(len(lengths)):
            start = max(0, i - window + 1)
            window_lengths = lengths[start:i+1]
            success_count = sum(1 for l in window_lengths if l >= 500)
            success_rate = (success_count / len(window_lengths)) * 100
            success_rates.append(success_rate)
        return success_rates
    
    dqn_success = compute_success_rate(dqn_metrics['episode_lengths'])
    ddqn_success = compute_success_rate(ddqn_metrics['episode_lengths'])
    
    dqn_episodes = np.arange(1, len(dqn_success) + 1)
    ddqn_episodes = np.arange(1, len(ddqn_success) + 1)
    
    ax.plot(dqn_episodes, dqn_success, color='blue', linewidth=2, label='DQN')
    ax.plot(ddqn_episodes, ddqn_success, color='red', linewidth=2, label='Double DQN')
    
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% Éxito')
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% Éxito')
    
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Tasa de Éxito (%)')
    ax.set_title('Tasa de Éxito: DQN vs Double DQN')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")


def plot_episode_length_comparison(dqn_metrics, ddqn_metrics, save_path='plots/length_comparison.png'):
    # Graficar comparación de longitudes de episodio.
    fig, ax = plt.subplots(figsize=(12, 5))
    
    dqn_lengths = dqn_metrics['episode_lengths']
    ddqn_lengths = ddqn_metrics['episode_lengths']
    
    dqn_episodes = np.arange(1, len(dqn_lengths) + 1)
    ddqn_episodes = np.arange(1, len(ddqn_lengths) + 1)
    
    # Promedios móviles
    dqn_ma = compute_moving_average(dqn_lengths, 50)
    ddqn_ma = compute_moving_average(ddqn_lengths, 50)
    
    ax.plot(dqn_episodes, dqn_lengths, alpha=0.2, color='blue')
    ax.plot(dqn_episodes, dqn_ma, color='blue', linewidth=2, label='DQN')
    
    ax.plot(ddqn_episodes, ddqn_lengths, alpha=0.2, color='red')
    ax.plot(ddqn_episodes, ddqn_ma, color='red', linewidth=2, label='Double DQN')
    
    ax.axhline(y=500, color='green', linestyle='--', alpha=0.5, label='Máximo (500)')
    
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Longitud del Episodio')
    ax.set_title('Longitud de Episodios: DQN vs Double DQN')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")


def plot_convergence_comparison(dqn_metrics, ddqn_metrics, save_path='plots/convergence_comparison.png'):
    # Graficar análisis de convergencia.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Recompensas con bandas de confianza
    ax1 = axes[0, 0]
    
    for metrics, label, color in [(dqn_metrics, 'DQN', 'blue'), (ddqn_metrics, 'Double DQN', 'red')]:
        rewards = metrics['episode_rewards']
        episodes = np.arange(1, len(rewards) + 1)
        
        ma = compute_moving_average(rewards, 50)
        
        # Calcular banda de confianza
        std_band = []
        for i in range(len(rewards)):
            start = max(0, i - 50 + 1)
            std_band.append(np.std(rewards[start:i+1]))
        std_band = np.array(std_band)
        
        ax1.fill_between(episodes, ma - std_band, ma + std_band, alpha=0.2, color=color)
        ax1.plot(episodes, ma, color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Convergencia con Bandas de Confianza')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Tiempo para alcanzar umbral
    ax2 = axes[0, 1]
    
    thresholds = [100, 200, 300, 400, 450]
    dqn_times = []
    ddqn_times = []
    
    for threshold in thresholds:
        # DQN
        dqn_ma = compute_moving_average(dqn_metrics['episode_rewards'], 50)
        dqn_idx = next((i for i, r in enumerate(dqn_ma) if r >= threshold), len(dqn_ma))
        dqn_times.append(dqn_idx)
        
        # Double DQN
        ddqn_ma = compute_moving_average(ddqn_metrics['episode_rewards'], 50)
        ddqn_idx = next((i for i, r in enumerate(ddqn_ma) if r >= threshold), len(ddqn_ma))
        ddqn_times.append(ddqn_idx)
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    ax2.bar(x - width/2, dqn_times, width, label='DQN', color='blue', alpha=0.7)
    ax2.bar(x + width/2, ddqn_times, width, label='Double DQN', color='red', alpha=0.7)
    
    ax2.set_xlabel('Umbral de Recompensa')
    ax2.set_ylabel('Episodios para Alcanzar')
    ax2.set_title('Velocidad de Convergencia')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(t) for t in thresholds])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Estabilidad (varianza de recompensas)
    ax3 = axes[1, 0]
    
    for metrics, label, color in [(dqn_metrics, 'DQN', 'blue'), (ddqn_metrics, 'Double DQN', 'red')]:
        rewards = metrics['episode_rewards']
        episodes = np.arange(1, len(rewards) + 1)
        
        # Calcular varianza móvil
        variance = []
        for i in range(len(rewards)):
            start = max(0, i - 50 + 1)
            variance.append(np.var(rewards[start:i+1]))
        
        ax3.plot(episodes, variance, color=color, linewidth=2, label=label, alpha=0.7)
    
    ax3.set_xlabel('Episodio')
    ax3.set_ylabel('Varianza de Recompensas')
    ax3.set_title('Estabilidad del Aprendizaje')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Resumen estadístico
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calcular estadísticas
    dqn_stats = {
        'Media': np.mean(dqn_metrics['episode_rewards']),
        'Máximo': np.max(dqn_metrics['episode_rewards']),
        'Desv. Std.': np.std(dqn_metrics['episode_rewards']),
        'Tasa Éxito': (dqn_metrics['success_count'] / dqn_metrics['total_episodes']) * 100,
        'Q-Value Prom.': np.mean(dqn_metrics['episode_q_values'])
    }
    
    ddqn_stats = {
        'Media': np.mean(ddqn_metrics['episode_rewards']),
        'Máximo': np.max(ddqn_metrics['episode_rewards']),
        'Desv. Std.': np.std(ddqn_metrics['episode_rewards']),
        'Tasa Éxito': (ddqn_metrics['success_count'] / ddqn_metrics['total_episodes']) * 100,
        'Q-Value Prom.': np.mean(ddqn_metrics['episode_q_values'])
    }
    
    # Crear tabla
    table_data = []
    for key in dqn_stats.keys():
        table_data.append([key, f"{dqn_stats[key]:.2f}", f"{ddqn_stats[key]:.2f}"])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Métrica', 'DQN', 'Double DQN'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    ax4.set_title('Resumen Estadístico', fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")


def generate_all_comparisons(dqn_metrics_path='metrics/dqn_metrics.pkl',
                             ddqn_metrics_path='metrics/ddqn_metrics.pkl'):
    print("Generando visualizaciones comparativas")
    
    # Cargar métricas
    dqn_metrics = load_metrics(dqn_metrics_path)
    ddqn_metrics = load_metrics(ddqn_metrics_path)
    
    print(f"\nMétricas DQN cargadas: {len(dqn_metrics['episode_rewards'])} episodios")
    print(f"Métricas Double DQN cargadas: {len(ddqn_metrics['episode_rewards'])} episodios")
    print()
    
    # Generar gráficas
    plot_reward_comparison(dqn_metrics, ddqn_metrics)
    plot_q_value_comparison(dqn_metrics, ddqn_metrics)
    plot_loss_comparison(dqn_metrics, ddqn_metrics)
    plot_success_rate_comparison(dqn_metrics, ddqn_metrics)
    plot_episode_length_comparison(dqn_metrics, ddqn_metrics)
    plot_convergence_comparison(dqn_metrics, ddqn_metrics)
    
    print("\nTodas las visualizaciones generadas exitosamente!")


def print_comparison_summary(dqn_metrics, ddqn_metrics):
    print("\nRESUMEN COMPARATIVO: DQN vs Double DQN")
    metrics_names = [
        ('Episodios entrenados', 'total_episodes', lambda x: f"{x}"),
        ('Recompensa promedio', 'episode_rewards', lambda x: f"{np.mean(x):.2f}"),
        ('Recompensa máxima', 'episode_rewards', lambda x: f"{np.max(x):.2f}"),
        ('Desviación estándar', 'episode_rewards', lambda x: f"{np.std(x):.2f}"),
        ('Longitud promedio', 'episode_lengths', lambda x: f"{np.mean(x):.1f}"),
        ('Q-Value promedio', 'episode_q_values', lambda x: f"{np.mean(x):.2f}"),
        ('Pérdida promedio', 'episode_losses', lambda x: f"{np.mean(x):.4f}"),
    ]
    
    print(f"\n{'Métrica':<25} {'DQN':>15} {'Double DQN':>15} {'Diferencia':>15}")
    
    for name, key, fmt in metrics_names:
        dqn_val = dqn_metrics[key] if key in ['total_episodes'] else dqn_metrics[key]
        ddqn_val = ddqn_metrics[key] if key in ['total_episodes'] else ddqn_metrics[key]
        
        if key == 'total_episodes':
            diff = ddqn_val - dqn_val
            print(f"{name:<25} {dqn_val:>15} {ddqn_val:>15} {diff:>+15}")
        else:
            dqn_mean = np.mean(dqn_val)
            ddqn_mean = np.mean(ddqn_val)
            diff = ddqn_mean - dqn_mean
            diff_pct = (diff / dqn_mean * 100) if dqn_mean != 0 else 0
            print(f"{name:<25} {fmt(dqn_val):>15} {fmt(ddqn_val):>15} {diff_pct:>+14.1f}%")
    
    # Tasa de éxito
    dqn_success = (dqn_metrics['success_count'] / dqn_metrics['total_episodes']) * 100
    ddqn_success = (ddqn_metrics['success_count'] / ddqn_metrics['total_episodes']) * 100
    diff = ddqn_success - dqn_success
    print(f"{'Tasa de éxito':<25} {dqn_success:>14.2f}% {ddqn_success:>14.2f}% {diff:>+14.1f}%")
    

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--summary':
        # Solo imprimir resumen
        dqn_metrics = load_metrics('metrics/dqn_metrics.pkl')
        ddqn_metrics = load_metrics('metrics/ddqn_metrics.pkl')
        print_comparison_summary(dqn_metrics, ddqn_metrics)
    else:
        # Generar todas las gráficas
        generate_all_comparisons()
