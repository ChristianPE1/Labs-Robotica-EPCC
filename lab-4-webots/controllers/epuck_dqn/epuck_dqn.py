from controller import Robot, DistanceSensor, Motor, Lidar
import numpy as np
import os
import random
from collections import deque
import matplotlib.pyplot as plt

# Disable GPU and force CPU execution
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import tensorflow as tf
# Configure TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras
from tensorflow.keras import layers

# Importar configuración
from config import *

# ============================================================================
# RED NEURONAL DQN CON KERAS
# ============================================================================

def create_dqn_model(input_size, hidden_layers, output_size):
    """Crear red neuronal con Keras"""
    model = keras.Sequential()
    
    # Capa de entrada
    model.add(layers.Input(shape=(input_size,)))
    
    # Capas ocultas
    for hidden_size in hidden_layers:
        model.add(layers.Dense(hidden_size, activation='relu'))
    
    # Capa de salida (Q-valores por acción)
    model.add(layers.Dense(output_size, activation='linear'))
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse'
    )
    
    return model


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Memoria para almacenar experiencias (s, a, r, s', done)"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# AGENTE DQN
# ============================================================================

class DQNAgent:
    """Agente que utiliza DQN para aprender"""
    
    def __init__(self, input_size, hidden_layers, output_size):
        # Redes neuronales
        self.policy_net = create_dqn_model(input_size, hidden_layers, output_size)
        self.target_net = create_dqn_model(input_size, hidden_layers, output_size)
        self.target_net.set_weights(self.policy_net.get_weights())
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        # Epsilon (exploración)
        self.epsilon = EPSILON_START
        
        # Estadísticas
        self.training_step = 0
        self.losses = []
        self.num_actions = output_size
    
    def select_action(self, state, training=True):
        """Seleccionar acción usando epsilon-greedy"""
        if training and random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return random.randint(0, self.num_actions - 1)
        else:
            # Explotación: mejor acción según Q-valores
            state_tensor = np.array([state], dtype=np.float32)
            q_values = self.policy_net.predict(state_tensor, verbose=0)
            return np.argmax(q_values[0])
    
    def train_step(self):
        """Realizar un paso de entrenamiento"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return None
        
        # Muestrear batch del replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        
        # Calcular Q-valores actuales
        current_q_values = self.policy_net.predict(states, verbose=0)
        
        # Calcular Q-valores objetivo usando target network
        next_q_values = self.target_net.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1)
        
        # Actualizar Q-valores objetivo
        target_q_values = current_q_values.copy()
        for i in range(BATCH_SIZE):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + GAMMA * max_next_q_values[i]
        
        # Entrenar la red
        history = self.policy_net.fit(
            states, 
            target_q_values, 
            epochs=1, 
            verbose=0,
            batch_size=BATCH_SIZE
        )
        
        loss = history.history['loss'][0]
        self.losses.append(loss)
        self.training_step += 1
        
        return loss
    
    def update_target_network(self):
        """Copiar pesos de policy net a target net"""
        self.target_net.set_weights(self.policy_net.get_weights())
    
    def decay_epsilon(self):
        """Decrementar epsilon"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def save_model(self, filename='dqn_model.keras'):
        """Guardar modelo"""
        self.policy_net.save(filename)
    
    def load_model(self, filename='dqn_model.keras'):
        """Cargar modelo"""
        if os.path.exists(filename):
            self.policy_net = keras.models.load_model(filename)
            self.target_net.set_weights(self.policy_net.get_weights())
            return True
        return False


# ============================================================================
# CONTROLADOR DEL ROBOT
# ============================================================================

class DQNRobotController:
    """Controlador principal del robot con DQN"""
    
    def __init__(self):
        # Inicializar robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # POSICIÓN INICIAL Y META (del archivo .wbt)
        self.start_pos = np.array([-0.53, 0.09])  # Posición inicial del robot
        self.goal_pos = np.array([0.8, 0.8])      # Centro de la zona meta
        
        # Inicializar sensores
        self.init_sensors()
        
        # Inicializar motores
        self.init_motors()
        
        # Inicializar LIDAR
        self.init_lidar()
        
        # Wait several timesteps for LIDAR initialization
        for _ in range(10):
            self.robot.step(self.timestep)
        
        # Agente DQN
        input_size = INPUT_SIZE  # 8 sensores + 2 LIDAR + 2 posición relativa = 12
        output_size = len(ACTIONS)
        self.agent = DQNAgent(input_size, HIDDEN_LAYERS, output_size)
        
        # Estadísticas
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        
        print(f"[INICIO] Robot DQN inicializado - {input_size} inputs, {output_size} acciones")
        print(f"[INICIO] Red neuronal: {HIDDEN_LAYERS}")
        print(f"[INICIO] Meta en: {self.goal_pos}")
        print(f"[INICIO] Comenzando entrenamiento de {EPISODES} episodios...\n")
    
    def init_sensors(self):
        """Inicializar sensores de distancia"""
        self.sensors = []
        sensor_names = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
        
        for name in sensor_names:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.sensors.append(sensor)
    
    def init_motors(self):
        """Inicializar motores"""
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
    
    def init_lidar(self):
        """Inicializar LIDAR para detección de meta"""
        self.lidar = self.robot.getDevice('lidar')
        if self.lidar is None:
            print("ERROR: LIDAR no encontrado")
            exit(1)
        else:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
    
    def get_sensor_values(self):
        """Return normalized proximity sensor values. Thresholds tuned for a 2x2 m world."""
        values = []
        for sensor in self.sensors:
            raw_value = sensor.getValue()
            # Normalizar con umbral ajustado para mundo pequeño
            # Sensores e-puck: 0 = lejos, ~4000 = muy cerca
            # En mundo 2x2m, reducir umbral para mayor sensibilidad
            normalized = min(raw_value / 2500.0, 1.0)  # Reducido de 3500 a 2500
            values.append(normalized)
        return values
    
    def detect_goal_with_lidar(self):
        """
        Detectar si el robot está DENTRO de la zona meta (0.8, 0.8) con tamaño 0.4x0.4
        META = robot dentro del cuadrado verde
        """
        try:
            # Obtener posición actual
            current_pos = np.array(self.robot.getSelf().getPosition()[:2])
            
            # Calcular distancia al centro de la meta
            distance_to_goal = np.linalg.norm(current_pos - self.goal_pos)
            
            # La zona meta tiene tamaño 0.4x0.4, así que radio efectivo ~0.2
            GOAL_RADIUS = 0.2
            
            # META: robot dentro de la zona verde
            return distance_to_goal <= GOAL_RADIUS
            
        except:
            return False
    
    def get_state(self):
        """Construir vector de estado usando sensores + posición relativa a meta"""
        # Valores de sensores (8)
        sensor_values = self.get_sensor_values()
        
        # Información contextual basada en LIDAR (2 valores) - FILTRAR INFINITOS
        try:
            point_cloud = self.lidar.getPointCloud()
            if point_cloud and len(point_cloud) > 0:
                # Filtrar distancias válidas (no infinitas)
                distances = []
                for p in point_cloud:
                    dist = np.sqrt(p.x**2 + p.y**2)
                    if np.isfinite(dist) and dist < 10.0:
                        distances.append(dist)
                
                if len(distances) > 10:  # Suficientes puntos válidos
                    avg_distance = np.mean(distances)
                    min_distance = min(distances)
                    
                    # Normalizar (límite razonable)
                    avg_normalized = min(avg_distance / 2.0, 1.0)
                    min_normalized = min(min_distance / 1.0, 1.0)
                else:
                    avg_normalized = 0.0
                    min_normalized = 0.0
            else:
                avg_normalized = 0.0
                min_normalized = 0.0
        except:
            # Si LIDAR aún no está listo, usar valores por defecto
            avg_normalized = 0.0
            min_normalized = 0.0
        
        # POSICIÓN RELATIVA A LA META (2 valores) - NUEVO
        try:
            # Obtener posición actual del robot (como supervisor)
            current_pos = np.array(self.robot.getSelf().getPosition()[:2])  # Solo X, Y
            
            # Vector relativo a la meta
            relative_pos = self.goal_pos - current_pos
            
            # Normalizar por el tamaño del mundo (2x2)
            pos_x_normalized = relative_pos[0] / 2.0  # -1 a 1
            pos_y_normalized = relative_pos[1] / 2.0  # -1 a 1
            
        except:
            # Si no se puede obtener posición, usar valores por defecto
            pos_x_normalized = 0.0
            pos_y_normalized = 0.0
        
        state = sensor_values + [avg_normalized, min_normalized, pos_x_normalized, pos_y_normalized]
        return np.array(state, dtype=np.float32)
    
    def execute_action(self, action_idx):
        """Execute selected action applying conservative safety checks.

        Uses proximity sensors to determine whether to stop or apply an evasive
        maneuver before executing the action commanded by the policy.
        """
        sensor_values = self.get_sensor_values()
        max_sensor = max(sensor_values)

        # If proximity exceeds a critical threshold, stop the motors
        if max_sensor > 0.95:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            return

        # If proximity indicates a very close obstacle, apply a gentle evasive turn
        if max_sensor > 0.8:
            # Apply reduced-magnitude evasive velocities
            self.left_motor.setVelocity(-0.2 * MAX_SPEED)
            self.right_motor.setVelocity(0.5 * MAX_SPEED)
            return

        left_speed, right_speed = ACTIONS[action_idx]
        self.left_motor.setVelocity(left_speed * MAX_SPEED)
        self.right_motor.setVelocity(right_speed * MAX_SPEED)
    
    def calculate_reward(self, state, prev_state):
        """Reward function guiding the agent toward the goal.

        - Large positive reward for reaching the goal
        - Positive reward for reducing distance to the goal
        - Negative reward for proximity to obstacles
        - Small step penalty to encourage efficiency
        """
        reward = 0
        
        # 1. GRAN RECOMPENSA: Llegar a la zona meta
        if self.detect_goal_with_lidar():
            reward += 500  # Meta alcanzada
            return reward, True
        
        # 2. RECOMPENSA POR ACERCARSE A LA META
        if prev_state is not None and len(prev_state) >= 12 and len(state) >= 12:
            # Extraer posiciones relativas previas y actuales
            prev_rel_x, prev_rel_y = prev_state[-2], prev_state[-1]
            curr_rel_x, curr_rel_y = state[-2], state[-1]
            
            # Calcular distancia previa y actual a la meta
            prev_distance = np.sqrt(prev_rel_x**2 + prev_rel_y**2)
            curr_distance = np.sqrt(curr_rel_x**2 + curr_rel_y**2)
            
            # Recompensa por acercarse (distancia reducida)
            if curr_distance < prev_distance - 0.05:  # Se acercó significativamente
                reward += 10
            elif curr_distance < prev_distance - 0.02:  # Se acercó un poco
                reward += 3
            elif curr_distance > prev_distance + 0.05:  # Se alejó
                reward -= 5
        
        # 3. PENALIZACIÓN: Obstáculos cerca (sensores IR)
        sensor_values = state[:8]
        max_sensor = max(sensor_values)
        
        if max_sensor > 0.9:  # Colisión real
            reward -= 30
        elif max_sensor > 0.7:  # Muy cerca
            reward -= 10
        elif max_sensor > 0.5:  # Cerca
            reward -= 3
        elif max_sensor > 0.3:  # Moderadamente cerca
            reward -= 1
        
        # 4. BONUS: Espacio libre alrededor (usando LIDAR)
        if len(state) >= 10:
            avg_lidar_dist = state[8]  # Distancia promedio LIDAR
            if avg_lidar_dist > 0.7:  # Mucho espacio libre
                reward += 2
        
        # 5. Penalización por paso (incentiva eficiencia)
        reward -= 0.1
        
        return reward, False
    
    def train_episode(self, episode):
        """Entrenar un episodio completo"""
        # Resetear motores
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Esperar breve detención
        for _ in range(10):
            self.robot.step(self.timestep)
        
        state = self.get_state()
        episode_reward = 0
        steps = 0
        
        prev_state = None
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Seleccionar y ejecutar acción
            action = self.agent.select_action(state, training=True)
            self.execute_action(action)
            
            # Simular tiempo suficiente para que las acciones tengan efecto (8 pasos)
            for _ in range(8):
                if self.robot.step(self.timestep) == -1:
                    # Calcular estadísticas finales si se termina el episodio
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(steps)
                    recent_rewards = self.episode_rewards[-50:]
                    success_count = sum(1 for r in recent_rewards if r >= 200)
                    success_rate = success_count / len(recent_rewards) * 100 if recent_rewards else 0
                    self.success_rate.append(success_rate)
                    return episode_reward, steps, success_rate
            
            # Observar nuevo estado
            next_state = self.get_state()
            reward, done = self.calculate_reward(next_state, prev_state)
            
            # Guardar experiencia
            self.agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Entrenar
            loss = self.agent.train_step()
            
            # Actualizar
            state = next_state
            prev_state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                # Meta alcanzada
                print(f"Meta alcanzada en {steps} pasos con reward {episode_reward:.1f}")
                break
        
        # Decrementar epsilon
        self.agent.decay_epsilon()
        
        # Actualizar target network
        if episode % TARGET_UPDATE_FREQUENCY == 0:
            self.agent.update_target_network()
        
        # Guardar estadísticas
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(steps)
        
        # Calcular tasa de éxito (últimos 50 episodios)
        # ÉXITO = episodios donde se alcanzó la meta (reward >= 500)
        recent_rewards = self.episode_rewards[-50:]
        success_count = sum(1 for r in recent_rewards if r >= 500)  # Meta alcanzada = reward >= 500
        success_rate = success_count / len(recent_rewards) * 100 if recent_rewards else 0
        self.success_rate.append(success_rate)
        
        return episode_reward, steps, success_rate
    
    def celebrate_escape(self):
        """Perform a spin maneuver when the goal is reached."""
        print("Area open detected. Executing spin maneuver")
        
        # Detenerse completamente primero
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        for _ in range(20):  # Más tiempo para asegurar detención completa
            self.robot.step(self.timestep)
        
    print("Initiating 360-degree spin")
        
        # Girar 360 grados (celebración) - Más lento y visible
        spin_duration = 4.0  # Más tiempo (4 segundos)
        spin_steps = int(spin_duration * 1000 / self.timestep)
        
        for i in range(spin_steps):
            # Giro más lento para que sea visible
            self.left_motor.setVelocity(MAX_SPEED * 0.3)   # Más lento
            self.right_motor.setVelocity(-MAX_SPEED * 0.3) # Más lento
            self.robot.step(self.timestep)
            
            # Display progress periodically
            if i % int(1000 / self.timestep) == 0:
                progress = int((i / spin_steps) * 100)
                print(f"Spin progress: {progress}%")
        
        # Detenerse completamente de nuevo
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        for _ in range(20):
            self.robot.step(self.timestep)
        
    print("Spin maneuver completed; robot ready for next episode")
    
    def run_training(self):
        """Ejecutar entrenamiento completo"""
        best_reward = float('-inf')
        
        print("="*70)
        print(f"ENTRENAMIENTO DQN - {EPISODES} EPISODIOS")
        print("="*70)
        
        for episode in range(1, EPISODES + 1):
            reward, steps, success_rate = self.train_episode(episode)
            
            # Mostrar progreso SIEMPRE (cada episodio)
            avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else reward
            print(f"Ep {episode:3d} | Reward:{reward:7.1f} | Promedio:{avg_reward:7.1f} | "
                  f"Pasos:{steps:4d} | Epsilon:{self.agent.epsilon:.3f} | Exito:{success_rate:5.1f}%")
            
            # Guardar mejor modelo
            if reward > best_reward:
                best_reward = reward
                self.agent.save_model('best_dqn_model.keras')
                print(f"  -> Nuevo mejor modelo! Reward: {best_reward:.1f}")
            
            # Guardar modelo periódicamente
            if episode % 50 == 0:
                self.agent.save_model(f'dqn_model_ep{episode}.keras')
                self.plot_training_progress()
                print(f"  -> Checkpoint guardado (episodio {episode})")
                self.plot_training_progress()
                print(f"  -> Checkpoint guardado (episodio {episode})")
        
        # Guardar modelo final
        self.agent.save_model('dqn_model_final.keras')
        self.plot_training_progress()
        
        print("\n" + "="*70)
        print(f"ENTRENAMIENTO COMPLETADO - Mejor reward: {best_reward:.1f}")
        print("="*70)
    
    def plot_training_progress(self):
        """Generar gráfica de progreso"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Recompensas
        axes[0, 0].plot(self.episode_rewards, alpha=0.3)
        axes[0, 0].plot(self.smooth(self.episode_rewards, 50), linewidth=2)
        axes[0, 0].set_xlabel('Episodio')
        axes[0, 0].set_ylabel('Recompensa')
        axes[0, 0].set_title('Recompensa por Episodio')
        axes[0, 0].grid(True)
        
        # Longitud de episodios
        axes[0, 1].plot(self.episode_lengths, alpha=0.3)
        axes[0, 1].plot(self.smooth(self.episode_lengths, 50), linewidth=2)
        axes[0, 1].set_xlabel('Episodio')
        axes[0, 1].set_ylabel('Pasos')
        axes[0, 1].set_title('Duración de Episodios')
        axes[0, 1].grid(True)
        
        # Tasa de éxito
        axes[1, 0].plot(self.success_rate, linewidth=2)
        axes[1, 0].set_xlabel('Episodio')
        axes[1, 0].set_ylabel('Éxito (%)')
        axes[1, 0].set_title('Tasa de Éxito (Meta Alcanzada ≥500 pts)')
        axes[1, 0].grid(True)
        
        # Pérdida
        if self.agent.losses:
            axes[1, 1].plot(self.agent.losses, alpha=0.3)
            axes[1, 1].plot(self.smooth(self.agent.losses, 100), linewidth=2)
            axes[1, 1].set_xlabel('Paso de Entrenamiento')
            axes[1, 1].set_ylabel('Pérdida')
            axes[1, 1].set_title('Pérdida de Entrenamiento')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('dqn_training_progress.png', dpi=150)
        plt.close()
    
    @staticmethod
    def smooth(data, window=50):
        """Suavizar datos para visualización"""
        if len(data) < window:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(data[start:i+1]))
        return smoothed


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    controller = DQNRobotController()
    controller.run_training()
