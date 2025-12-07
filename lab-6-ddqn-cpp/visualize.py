#!/usr/bin/env python3
"""
Visualización de resultados DQN vs Double DQN
Genera gráficos comparativos a partir de los CSV generados por el programa C++
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(data, window=50):
    """Calcular promedio móvil"""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def load_metrics(filename):
    """Cargar métricas desde CSV"""
    if not os.path.exists(filename):
        print(f"Error: No se encuentra {filename}")
        return None
    return pd.read_csv(filename)

def plot_comparison(dqn_df, ddqn_df, output_dir='.'):
    """Generar gráficos comparativos"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparación DQN vs Double DQN - CartPole', fontsize=14, fontweight='bold')
    
    # 1. Recompensas por episodio
    ax1 = axes[0, 0]
    ax1.plot(dqn_df['episode'], dqn_df['reward'], alpha=0.3, color='blue', label='DQN (raw)')
    ax1.plot(ddqn_df['episode'], ddqn_df['reward'], alpha=0.3, color='red', label='DDQN (raw)')
    
    # Promedio móvil
    window = 50
    if len(dqn_df) > window:
        dqn_ma = moving_average(dqn_df['reward'], window)
        ddqn_ma = moving_average(ddqn_df['reward'], window)
        ax1.plot(range(window, len(dqn_df)+1), dqn_ma, color='blue', linewidth=2, label=f'DQN (MA-{window})')
        ax1.plot(range(window, len(ddqn_df)+1), ddqn_ma, color='red', linewidth=2, label=f'DDQN (MA-{window})')
    
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Recompensas por Episodio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=500, color='green', linestyle='--', alpha=0.5, label='Máximo (500)')
    
    # 2. Longitud de episodios
    ax2 = axes[0, 1]
    ax2.plot(dqn_df['episode'], dqn_df['length'], alpha=0.3, color='blue')
    ax2.plot(ddqn_df['episode'], ddqn_df['length'], alpha=0.3, color='red')
    
    if len(dqn_df) > window:
        dqn_len_ma = moving_average(dqn_df['length'], window)
        ddqn_len_ma = moving_average(ddqn_df['length'], window)
        ax2.plot(range(window, len(dqn_df)+1), dqn_len_ma, color='blue', linewidth=2, label='DQN')
        ax2.plot(range(window, len(ddqn_df)+1), ddqn_len_ma, color='red', linewidth=2, label='DDQN')
    
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Pasos')
    ax2.set_title('Longitud de Episodios')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=500, color='green', linestyle='--', alpha=0.5)
    
    # 3. Pérdida (Loss)
    ax3 = axes[1, 0]
    ax3.plot(dqn_df['episode'], dqn_df['loss'], alpha=0.5, color='blue', label='DQN')
    ax3.plot(ddqn_df['episode'], ddqn_df['loss'], alpha=0.5, color='red', label='DDQN')
    ax3.set_xlabel('Episodio')
    ax3.set_ylabel('Loss')
    ax3.set_title('Pérdida durante Entrenamiento')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Epsilon decay
    ax4 = axes[1, 1]
    ax4.plot(dqn_df['episode'], dqn_df['epsilon'], color='blue', label='DQN')
    ax4.plot(ddqn_df['episode'], ddqn_df['epsilon'], color='red', linestyle='--', label='DDQN')
    ax4.set_xlabel('Episodio')
    ax4.set_ylabel('Epsilon')
    ax4.set_title('Decaimiento de Epsilon')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_plots.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'comparison_plots.pdf'), bbox_inches='tight')
    print(f"Gráficos guardados en: comparison_plots.png/pdf")
    plt.close()
    
    # Gráfico adicional: Tasa de éxito acumulada
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    dqn_success = (dqn_df['length'] >= 500).cumsum() / (dqn_df['episode']) * 100
    ddqn_success = (ddqn_df['length'] >= 500).cumsum() / (ddqn_df['episode']) * 100
    
    ax.plot(dqn_df['episode'], dqn_success, color='blue', linewidth=2, label='DQN')
    ax.plot(ddqn_df['episode'], ddqn_success, color='red', linewidth=2, label='Double DQN')
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Tasa de Éxito Acumulada (%)')
    ax.set_title('Tasa de Éxito Acumulada - DQN vs Double DQN')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'success_rate.pdf'), bbox_inches='tight')
    print(f"Tasa de éxito guardada en: success_rate.png/pdf")
    plt.close()

def print_statistics(dqn_df, ddqn_df):
    """Imprimir estadísticas comparativas"""
    print("\n" + "="*60)
    print("ESTADÍSTICAS COMPARATIVAS")
    print("="*60)
    
    metrics = ['reward', 'length']
    names = ['Recompensa', 'Longitud']
    
    print(f"\n{'Métrica':<25} {'DQN':>15} {'Double DQN':>15}")
    print("-"*55)
    
    for metric, name in zip(metrics, names):
        dqn_mean = dqn_df[metric].mean()
        ddqn_mean = ddqn_df[metric].mean()
        print(f"{name + ' (promedio)':<25} {dqn_mean:>15.2f} {ddqn_mean:>15.2f}")
        
        dqn_std = dqn_df[metric].std()
        ddqn_std = ddqn_df[metric].std()
        print(f"{name + ' (std)':<25} {dqn_std:>15.2f} {ddqn_std:>15.2f}")
        
        dqn_max = dqn_df[metric].max()
        ddqn_max = ddqn_df[metric].max()
        print(f"{name + ' (máximo)':<25} {dqn_max:>15.2f} {ddqn_max:>15.2f}")
    
    # Tasa de éxito
    dqn_success = (dqn_df['length'] >= 500).sum()
    ddqn_success = (ddqn_df['length'] >= 500).sum()
    dqn_success_rate = dqn_success / len(dqn_df) * 100
    ddqn_success_rate = ddqn_success / len(ddqn_df) * 100
    
    print(f"{'Episodios exitosos':<25} {dqn_success:>15} {ddqn_success:>15}")
    print(f"{'Tasa de éxito (%)':<25} {dqn_success_rate:>15.2f} {ddqn_success_rate:>15.2f}")
    
    # Últimos 100 episodios
    print("\n" + "-"*55)
    print("Últimos 100 episodios:")
    dqn_last100 = dqn_df.tail(100)['reward'].mean()
    ddqn_last100 = ddqn_df.tail(100)['reward'].mean()
    print(f"{'Recompensa promedio':<25} {dqn_last100:>15.2f} {ddqn_last100:>15.2f}")
    
    dqn_last100_success = (dqn_df.tail(100)['length'] >= 500).sum()
    ddqn_last100_success = (ddqn_df.tail(100)['length'] >= 500).sum()
    print(f"{'Episodios exitosos':<25} {dqn_last100_success:>15} {ddqn_last100_success:>15}")

def main():
    print("Cargando métricas...")
    
    dqn_df = load_metrics('dqn_metrics.csv')
    ddqn_df = load_metrics('ddqn_metrics.csv')
    
    if dqn_df is None or ddqn_df is None:
        print("Error: Primero ejecuta el programa de entrenamiento (./train)")
        return
    
    print(f"DQN: {len(dqn_df)} episodios cargados")
    print(f"Double DQN: {len(ddqn_df)} episodios cargados")
    
    # Generar gráficos
    plot_comparison(dqn_df, ddqn_df)
    
    # Imprimir estadísticas
    print_statistics(dqn_df, ddqn_df)
    
    print("\n" + "="*60)
    print("Archivos generados:")
    print("  - comparison_plots.png/pdf")
    print("  - success_rate.png/pdf")
    print("="*60)

if __name__ == "__main__":
    main()
