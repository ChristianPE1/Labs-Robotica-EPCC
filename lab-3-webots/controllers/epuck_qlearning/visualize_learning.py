import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

def load_qtable():
    filename = 'qtable.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            q_table = pickle.load(f)
        return q_table
    else:
        print("No se encontró archivo qtable.pkl")
        return None

def analyze_qtable(q_table):
    if not q_table:
        return
    
    print("ANÁLISIS DE Q-TABLE")
    
    # Número de estados
    num_states = len(q_table)
    print(f"\nNúmero de estados explorados: {num_states}")
    
    # Valores Q
    all_q_values = []
    for state, actions in q_table.items():
        all_q_values.extend(actions)
    
    all_q_values = np.array(all_q_values)

    print(f"Estadísticas de valores Q:")
    print(f"   Máximo: {np.max(all_q_values):.2f}")
    print(f"   Mínimo: {np.min(all_q_values):.2f}")
    print(f"   Promedio: {np.mean(all_q_values):.2f}")
    print(f"   Desviación estándar: {np.std(all_q_values):.2f}")
    
    # Mejores estados
    print(f"\n5 estados con mejores valores Q:")
    state_max_q = [(state, np.max(actions)) for state, actions in q_table.items()]
    state_max_q.sort(key=lambda x: x[1], reverse=True)
    
    for i, (state, max_q) in enumerate(state_max_q[:5], 1):
        print(f"   {i}. Estado {state}: Q_max = {max_q:.2f}")
    
    # Distribución de acciones preferidas
    print(f"\nDistribución de acciones preferidas:")
    action_names = ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'BACK_LEFT', 'BACK_RIGHT']
    preferred_actions = [np.argmax(actions) for actions in q_table.values()]
    action_counts = [preferred_actions.count(i) for i in range(5)]
    
    for action_id, count in enumerate(action_counts):
        percentage = (count / num_states) * 100
        print(f"   {action_names[action_id]}: {count} estados ({percentage:.1f}%)")
    
    return all_q_values, action_counts, action_names

def plot_statistics(all_q_values, action_counts, action_names):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histograma de valores Q
        ax1.hist(all_q_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Valor Q', fontsize=12)
        ax1.set_ylabel('Frecuencia', fontsize=12)
        ax1.set_title('Distribución de Valores Q', fontsize=14, fontweight='bold')
        ax1.axvline(np.mean(all_q_values), color='red', linestyle='--', 
                   label=f'Media: {np.mean(all_q_values):.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de barras de acciones preferidas
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        bars = ax2.bar(action_names, action_counts, color=colors, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Acción', fontsize=12)
        ax2.set_ylabel('Número de Estados', fontsize=12)
        ax2.set_title('Acciones Preferidas por Estado', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('qtable_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nGráficos guardados en: qtable_analysis.png")
        plt.show()
        
    except Exception as e:
        print(f"\nNo se pudo generar el gráfico: {e}")

def show_best_policy(q_table):

    action_names = ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'BACK_LEFT', 'BACK_RIGHT']
    sensor_levels = {0: 'Libre', 1: 'Cerca', 2: 'Muy cerca'}
    
    # Mostrar algunos estados interesantes
    examples = []
    for state, actions in list(q_table.items())[:10]:
        best_action = np.argmax(actions)
        best_q = np.max(actions)
        
        # Interpretar sensores
        sensor_str = [f"{sensor_levels[s]}" for s in state]
        examples.append((sensor_str, action_names[best_action], best_q))
    
    for i, (sensors, action, q_value) in enumerate(examples, 1):
        print(f"   {i}. Sensores: {sensors}")
        print(f"      → Acción: {action} (Q={q_value:.2f})")

def main():
   
    # Cargar Q-table
    q_table = load_qtable()
    
    if q_table:
        # Analizar
        all_q_values, action_counts, action_names = analyze_qtable(q_table)
        
        # Mostrar política
        show_best_policy(q_table)
        
        # Generar gráficos
        print(f"\nGenerando visualizaciones...")
        plot_statistics(all_q_values, action_counts, action_names)
        
        print(f"\nAnálisis completado!")
    else:
        print("\nNo hay datos para analizar.")

if __name__ == "__main__":
    main()
