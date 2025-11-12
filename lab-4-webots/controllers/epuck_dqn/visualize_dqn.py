"""
Script para visualizar y analizar el entrenamiento de DQN con TensorFlow/Keras
Permite cargar modelos guardados y visualizar métricas
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from config import *
from epuck_dqn import create_dqn_model

def load_model_info(model_path):
    """Cargar información del modelo TensorFlow/Keras"""
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None

def plot_model_architecture():
    """Visualizar la arquitectura de la red DQN con Keras"""
    model = create_dqn_model(STATE_SIZE, HIDDEN_LAYERS, len(ACTIONS))

    print("\n" + "="*60)
    print("ARQUITECTURA DE LA RED NEURONAL DQN (TensorFlow/Keras)")
    print("="*60)
    print(model.summary())
    print("\nNúmero total de parámetros:", model.count_params())
    print("Parámetros entrenables:", sum([layer.count_params() for layer in model.layers if len(layer.trainable_weights) > 0]))
    print("="*60 + "\n")

def analyze_training_log(log_file='dqn_training_log.txt'):
    """Analizar archivo de log si existe"""
    if not os.path.exists(log_file):
        print(f"No se encontró archivo de log: {log_file}")
        return

    episodes = []
    rewards = []

    with open(log_file, 'r') as f:
        for line in f:
            if 'FIN EPISODIO' in line:
                parts = line.split()
                ep = int(parts[2])
                episodes.append(ep)
            elif 'Recompensa total:' in line and episodes:
                reward = float(line.split(':')[1].strip())

def visualize_q_values_distribution(model_path='best_dqn_model.h5'):
    """Visualizar distribución de Q-valores para diferentes estados"""
    model = load_model_info(model_path)
    if model is None:
        return

    print("📊 Generando distribución de Q-valores...")

    # Generar estados de prueba
    test_states = []

    # Estados cerca de la meta
    for dist in [0.1, 0.5, 1.0]:
        for angle in [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]:
            # Sensores limpios, distancia normalizada, ángulo
            sensors = [0.0] * 8  # Sin obstáculos
            state = sensors + [dist/3.0, angle/np.pi]
            test_states.append(state)

    # Estados con obstáculos
    for sensor_idx in range(8):
        sensors = [0.0] * 8
        sensors[sensor_idx] = 0.8  # Obstáculo cercano
        state = sensors + [1.0, 0.0]  # Lejos de meta
        test_states.append(state)

    test_states = np.array(test_states, dtype=np.float32)

    # Predecir Q-valores
    q_values = model.predict(test_states, verbose=0)

    # Visualizar
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Distribución general
    axes[0, 0].hist(q_values.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('Q-valor')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title('Distribución General de Q-valores')
    axes[0, 0].grid(True)

    # Q-valores por acción
    action_names = ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'SLIGHT_LEFT', 'SLIGHT_RIGHT']
    q_by_action = q_values.mean(axis=0)
    axes[0, 1].bar(action_names, q_by_action, color='green', alpha=0.7)
    axes[0, 1].set_ylabel('Q-valor promedio')
    axes[0, 1].set_title('Q-valores Promedio por Acción')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, axis='y')

    # Estados cerca de meta vs lejos
    near_goal = q_values[:5]  # Primeros 5 estados (cerca de meta)
    far_goal = q_values[5:10]  # Siguientes 5 estados (lejos de meta)

    axes[1, 0].boxplot([near_goal.flatten(), far_goal.flatten()],
                      labels=['Cerca de Meta', 'Lejos de Meta'])
    axes[1, 0].set_ylabel('Q-valor')
    axes[1, 0].set_title('Q-valores: Cerca vs Lejos de Meta')
    axes[1, 0].grid(True, axis='y')

    # Estados con obstáculos
    obstacle_states = q_values[10:]  # Estados con obstáculos
    axes[1, 1].boxplot(obstacle_states.flatten())
    axes[1, 1].set_ylabel('Q-valor')
    axes[1, 1].set_title('Q-valores en Estados con Obstáculos')
    axes[1, 1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('q_values_distribution.png', dpi=150, bbox_inches='tight')
    print("✅ Gráfica guardada: q_values_distribution.png")
    plt.close()

def test_model_policy(model_path='best_dqn_model.h5', num_tests=5):
    """Probar la política del modelo en escenarios específicos"""
    model = load_model_info(model_path)
    if model is None:
        return

    print(f"🎯 Probando política del modelo ({num_tests} escenarios)...")

    test_scenarios = [
        {
            'name': 'Cerca de meta, sin obstáculos',
            'state': [0.0]*8 + [0.1, 0.0],  # Sensores limpios, cerca de meta
            'expected': 'FORWARD'
        },
        {
            'name': 'Lejos de meta, sin obstáculos',
            'state': [0.0]*8 + [1.0, 0.0],  # Sensores limpios, lejos de meta
            'expected': 'FORWARD'
        },
        {
            'name': 'Obstáculo adelante',
            'state': [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.5, 0.0],
            'expected': 'TURN_LEFT o TURN_RIGHT'
        },
        {
            'name': 'Obstáculo a la derecha',
            'state': [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.5, 0.0],
            'expected': 'TURN_LEFT'
        },
        {
            'name': 'Meta a la izquierda',
            'state': [0.0]*8 + [0.3, -0.5],  # Meta a la izquierda
            'expected': 'TURN_LEFT o SLIGHT_LEFT'
        }
    ]

    action_names = ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'SLIGHT_LEFT', 'SLIGHT_RIGHT']

    print("\n" + "="*80)
    print("PRUEBA DE POLÍTICA DEL MODELO")
    print("="*80)

    for i, scenario in enumerate(test_scenarios[:num_tests]):
        state = np.array([scenario['state']], dtype=np.float32)
        q_values = model.predict(state, verbose=0)[0]

        best_action_idx = np.argmax(q_values)
        best_action = action_names[best_action_idx]
        best_q = q_values[best_action_idx]

        print(f"\n{i+1}. {scenario['name']}")
        print(f"   Estado: {scenario['state']}")
        print(f"   Acción elegida: {best_action} (Q={best_q:.3f})")
        print(f"   Esperado: {scenario['expected']}")

        # Mostrar todas las Q-valores
        q_str = "   Q-valores: "
        for j, (action, q) in enumerate(zip(action_names, q_values)):
            marker = " ←" if j == best_action_idx else ""
            q_str += f"{action}={q:.3f}{marker}  "
        print(q_str)

    print("\n" + "="*80)

def analyze_training_progress():
    """Analizar el progreso del entrenamiento desde archivos guardados"""
    print("📈 Analizando progreso del entrenamiento...")

    # Buscar archivos de modelos guardados
    model_files = []
    for file in os.listdir('.'):
        if file.startswith('dqn_model_ep') and file.endswith('.h5'):
            episode = int(file.split('ep')[1].split('.')[0])
            model_files.append((episode, file))

    if not model_files:
        print("❌ No se encontraron modelos guardados")
        return

    model_files.sort()
    episodes = [ep for ep, _ in model_files]
    print(f"📊 Encontrados {len(episodes)} modelos guardados: episodios {episodes}")

    # Cargar el último modelo
    last_ep, last_file = model_files[-1]
    print(f"🔍 Analizando modelo del episodio {last_ep}...")

    model = load_model_info(last_file)
    if model is None:
        return

    # Mostrar arquitectura
    plot_model_architecture()

    # Visualizar Q-valores
    visualize_q_values_distribution(last_file)

    # Probar política
    test_model_policy(last_file)

def main():
    """Función principal"""
    print("🤖 VISUALIZACIÓN Y ANÁLISIS DQN")
    print("="*50)

    # Verificar que estamos en el directorio correcto
    if not os.path.exists('config.py'):
        print("❌ Ejecuta este script desde controllers/epuck_dqn/")
        print("   cd controllers/epuck_dqn")
        print("   python3 visualize_dqn.py")
        return

    while True:
        print("\n" + "="*50)
        print("OPCIONES DE ANÁLISIS:")
        print("1. Mostrar arquitectura del modelo")
        print("2. Visualizar distribución de Q-valores")
        print("3. Probar política del modelo")
        print("4. Análisis completo del entrenamiento")
        print("5. Salir")
        print("="*50)

        try:
            choice = input("Selecciona una opción (1-5): ").strip()

            if choice == '1':
                plot_model_architecture()
            elif choice == '2':
                model_file = input("Archivo del modelo (default: best_dqn_model.h5): ").strip()
                if not model_file:
                    model_file = 'best_dqn_model.h5'
                visualize_q_values_distribution(model_file)
            elif choice == '3':
                model_file = input("Archivo del modelo (default: best_dqn_model.h5): ").strip()
                if not model_file:
                    model_file = 'best_dqn_model.h5'
                test_model_policy(model_file)
            elif choice == '4':
                analyze_training_progress()
            elif choice == '5':
                print("👋 ¡Hasta luego!")
                break
            else:
                print("❌ Opción no válida")

        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()