"""DQN training visualization and analysis utilities.

This module provides functions to load saved Keras models, inspect the
network architecture, and generate plots for Q-value distributions and
training diagnostics.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from config import *
from epuck_dqn import create_dqn_model

def load_model_info(model_path):
    """Load a TensorFlow/Keras model from disk.

    Returns the loaded model or None if loading fails.
    """
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def plot_model_architecture():
    """Print model summary and parameter counts.

    Uses the architecture defined in `config.py` to instantiate a model and
    prints its summary and parameter counts.
    """
    model = create_dqn_model(STATE_SIZE, HIDDEN_LAYERS, len(ACTIONS))

    print("\n" + "="*60)
    print("DQN model architecture (TensorFlow/Keras)")
    print("="*60)
    print(model.summary())
    print("\nTotal parameters:", model.count_params())
    print("Trainable parameters:", sum([layer.count_params() for layer in model.layers if len(layer.trainable_weights) > 0]))
    print("="*60 + "\n")

def analyze_training_log(log_file='dqn_training_log.txt'):
    """Parse a training log file if present and extract episode metrics."""
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
    """Generate and save plots of Q-value distributions for sample states."""
    model = load_model_info(model_path)
    if model is None:
        return

    print("Generating Q-values distribution...")

    # Generate sample states for analysis
    test_states = []

    # States near the goal
    for dist in [0.1, 0.5, 1.0]:
        for angle in [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]:
            # Sensores limpios, distancia normalizada, ángulo
            sensors = [0.0] * 8  # Sin obstáculos
            state = sensors + [dist/3.0, angle/np.pi]
            test_states.append(state)

    # States with obstacles
    for sensor_idx in range(8):
        sensors = [0.0] * 8
        sensors[sensor_idx] = 0.8  # Obstáculo cercano
        state = sensors + [1.0, 0.0]  # Lejos de meta
        test_states.append(state)

    test_states = np.array(test_states, dtype=np.float32)

    # Predict Q-values
    q_values = model.predict(test_states, verbose=0)

    # Visualizar
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Overall distribution
    axes[0, 0].hist(q_values.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('Q-value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Overall Q-value Distribution')
    axes[0, 0].grid(True)

    # Q-values per action
    action_names = ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'SLIGHT_LEFT', 'SLIGHT_RIGHT']
    q_by_action = q_values.mean(axis=0)
    axes[0, 1].bar(action_names, q_by_action, color='green', alpha=0.7)
    axes[0, 1].set_ylabel('Mean Q-value')
    axes[0, 1].set_title('Mean Q-value per Action')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, axis='y')

    # States near vs far from goal
    near_goal = q_values[:5]
    far_goal = q_values[5:10]

    axes[1, 0].boxplot([near_goal.flatten(), far_goal.flatten()], labels=['Near goal', 'Far from goal'])
    axes[1, 0].set_ylabel('Q-value')
    axes[1, 0].set_title('Q-values: Near vs Far from Goal')
    axes[1, 0].grid(True, axis='y')

    # States with obstacles
    obstacle_states = q_values[10:]
    axes[1, 1].boxplot(obstacle_states.flatten())
    axes[1, 1].set_ylabel('Q-value')
    axes[1, 1].set_title('Q-values for Obstacle States')
    axes[1, 1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('q_values_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved plot: q_values_distribution.png")
    plt.close()

def test_model_policy(model_path='best_dqn_model.h5', num_tests=5):
    """Evaluate model policy on a set of predefined scenarios."""
    model = load_model_info(model_path)
    if model is None:
        return

    print(f"Testing model policy ({num_tests} scenarios)...")

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
    print("MODEL POLICY TEST")
    print("="*80)

    for i, scenario in enumerate(test_scenarios[:num_tests]):
        state = np.array([scenario['state']], dtype=np.float32)
        q_values = model.predict(state, verbose=0)[0]

        best_action_idx = np.argmax(q_values)
        best_action = action_names[best_action_idx]
        best_q = q_values[best_action_idx]

        print(f"\n{i+1}. {scenario['name']}")
    print(f"   State: {scenario['state']}")
    print(f"   Selected action: {best_action} (Q={best_q:.3f})")
    print(f"   Expected: {scenario['expected']}")

        # Display all Q-values
        q_str = "   Q-values: "
        for j, (action, q) in enumerate(zip(action_names, q_values)):
            marker = " ←" if j == best_action_idx else ""
            q_str += f"{action}={q:.3f}{marker}  "
        print(q_str)

    print("\n" + "="*80)

def analyze_training_progress():
    """Analyze training progress from saved model checkpoints."""
    print("Analyzing training progress...")

    # Buscar archivos de modelos guardados
    model_files = []
    for file in os.listdir('.'):
        if file.startswith('dqn_model_ep') and file.endswith('.h5'):
            episode = int(file.split('ep')[1].split('.')[0])
            model_files.append((episode, file))

    if not model_files:
        print("No saved models found")
        return

    model_files.sort()
    episodes = [ep for ep, _ in model_files]
    print(f"Found {len(episodes)} saved models: episodes {episodes}")

    # Cargar el último modelo
    last_ep, last_file = model_files[-1]
    print(f"Analyzing model from episode {last_ep}...")

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
    print("DQN Visualization and Analysis")
    print("="*50)

    # Verificar que estamos en el directorio correcto
    if not os.path.exists('config.py'):
        print("Run this script from controllers/epuck_dqn/")
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
                print("Exit.")
                break
            else:
                print("Invalid option")

        except KeyboardInterrupt:
            print("\nExit.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()