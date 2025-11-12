from controller import Robot, DistanceSensor, Motor, GPS
import numpy as np
import pickle
import os

# Parámetros de Q-Learning
ALPHA = 0.1          # Tasa de aprendizaje
GAMMA = 0.9          # Factor de descuento
EPSILON = 0.3        # Exploración vs explotación
EPISODES = 1000      # Número de episodios de entrenamiento

# Parámetros del robot
MAX_SPEED = 6.28
TIME_STEP = 64

# Parámetros de detección con LIDAR
LIDAR_GOAL_THRESHOLD = 0.35  # Meta: todos los obstáculos a más de 35cm (área abierta)
LIDAR_MIN_CLEAR_RATIO = 0.85  # 85% de los puntos deben estar lejos

# Definición de acciones
ACTIONS = [
    'FORWARD',       # 0: Avanzar
    'TURN_LEFT',     # 1: Girar izquierda
    'TURN_RIGHT',    # 2: Girar derecha
    'BACK_LEFT',     # 3: Retroceder izquierda
    'BACK_RIGHT'     # 4: Retroceder derecha
]

# Sistema de recompensas
OBSTACLE_THRESHOLD_CLOSE = 500
OBSTACLE_THRESHOLD_NEAR = 100


class QLearningRobot:
    def __init__(self):
        # Inicializar robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Motores
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Sensores de distancia
        self.distance_sensors = []
        self.sensor_names = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
        for name in self.sensor_names:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.distance_sensors.append(sensor)
        
        # GPS para conocer posición
        self.gps = self.robot.getDevice('gps')
        if self.gps is None:
            print("Advertencia: GPS no encontrado")
        else:
            self.gps.enable(self.timestep)
        
        # LIDAR para detección de meta
        self.lidar = self.robot.getDevice('lidar')
        if self.lidar is None:
            print("ERROR: LIDAR no encontrado - el robot necesita un sensor LIDAR")
            exit(1)
        else:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
            print("✅ LIDAR habilitado correctamente")
        
        # Q-Table
        self.q_table = {}
        self.episode = 0
        self.steps = 0
        self.total_reward = 0
        
        # Cargar Q-table si existe
        self.load_qtable()
        
        print("Robot inicializado correctamente")
        print(f"Número de acciones: {len(ACTIONS)}")
    
    def get_sensor_state(self):
        """
        Discretiza los valores de los sensores en estados
        Clasifica cada sensor en: libre (0), cerca (1), muy_cerca (2)
        """
        state = []
        for sensor in self.distance_sensors:
            value = sensor.getValue()
            # Normalizar valores del sensor (0-4095, mayor valor = más cerca)
            if value < 100:
                state.append(0)  # Libre
            elif value < 500:
                state.append(1)  # Cerca
            else:
                state.append(2)  # Muy cerca
        
        # Usar solo los sensores frontales y laterales más importantes (ps0, ps2, ps5, ps7)
        # para reducir el espacio de estados
        relevant_sensors = [state[0], state[2], state[5], state[7]]
        
        # Debug: imprimir valores cada 200 pasos
        if self.steps % 200 == 0 and self.steps > 0:
            print(f"  [Debug] Estado completo: {state}")
            print(f"  [Debug] Estado relevante: {relevant_sensors}")
        
        return tuple(relevant_sensors)

    def detect_goal(self):

        # Obtener point cloud del LIDAR
        point_cloud = self.lidar.getPointCloud()
        
        if not point_cloud or len(point_cloud) == 0:
            return False
        
        # Calcular distancias a todos los puntos detectados
        distances = []
        for point in point_cloud:
            # Distancia en 2D (vista superior)
            dist = np.sqrt(point.x**2 + point.y**2)
            distances.append(dist)
        
        # Contar puntos que están LEJOS (> umbral)
        clear_points = sum(1 for d in distances if d > LIDAR_GOAL_THRESHOLD)
        total_points = len(distances)
        clear_ratio = clear_points / total_points if total_points > 0 else 0
        
        # Debug cada 20 pasos
        if self.steps % 20 == 0:
            min_dist = min(distances) if distances else 0
            avg_dist = np.mean(distances) if distances else 0
            print(f"  [LIDAR] Puntos: {total_points}, Libres (>{LIDAR_GOAL_THRESHOLD}m): {clear_points} ({clear_ratio*100:.1f}%)")
            print(f"  [LIDAR] Dist mín: {min_dist:.3f}m, Dist promedio: {avg_dist:.3f}m")
            print(f"  [META?] {'SÍ' if clear_ratio >= LIDAR_MIN_CLEAR_RATIO else 'NO'}")
        
        # META: si al menos 85% de puntos están lejos → área abierta
        return clear_ratio >= LIDAR_MIN_CLEAR_RATIO
    
    def calculate_reward(self, prev_state, current_state):
        """
        Calcula recompensa basada únicamente en sensores locales (más realista)
        """
        reward = 0

        # 1. Recompensa por alcanzar la meta (detectada por sensores)
        if self.detect_goal():
            reward += 100
            return reward, True

        # 2. Recompensa por moverse hacia áreas más abiertas
        # Comparar el estado actual con el anterior
        if prev_state is not None:
            prev_open_spaces = sum(1 for val in prev_state if val == 0)  # Estados libres
            current_open_spaces = sum(1 for val in current_state if val == 0)

            if current_open_spaces > prev_open_spaces:
                reward += 2  # Recompensa por encontrar más espacio abierto
            elif current_open_spaces < prev_open_spaces:
                reward -= 1  # Penalización por ir hacia áreas más cerradas

        # 3. Penalización por obstáculos
        current_sensor_values = [sensor.getValue() for sensor in self.distance_sensors]
        for value in current_sensor_values:
            if value > OBSTACLE_THRESHOLD_CLOSE:
                reward -= 5  # Obstáculo muy cerca
            elif value > OBSTACLE_THRESHOLD_NEAR:
                reward -= 1  # Obstáculo cerca

        # 4. Penalización por cada paso (incentiva eficiencia)
        reward -= 0.1

        return reward, False

    # Usando politica epsilon-greedy
    def choose_action(self, state):
        if np.random.random() < EPSILON:
            # Exploración: acción aleatoria
            return np.random.randint(0, len(ACTIONS))
        else:
            # Explotación: mejor acción según Q-table
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(ACTIONS))
            return np.argmax(self.q_table[state])
    
    def execute_action(self, action):
        speed = MAX_SPEED
        
        if action == 0:  # FORWARD
            self.left_motor.setVelocity(speed)
            self.right_motor.setVelocity(speed)
        elif action == 1:  # TURN_LEFT
            self.left_motor.setVelocity(-speed * 0.5)
            self.right_motor.setVelocity(speed * 0.5)
        elif action == 2:  # TURN_RIGHT
            self.left_motor.setVelocity(speed * 0.5)
            self.right_motor.setVelocity(-speed * 0.5)
        elif action == 3:  # BACK_LEFT
            self.left_motor.setVelocity(-speed * 0.5)
            self.right_motor.setVelocity(-speed * 0.3)
        elif action == 4:  # BACK_RIGHT
            self.left_motor.setVelocity(-speed * 0.3)
            self.right_motor.setVelocity(-speed * 0.5)

    # Actualiza el valor Q usando la ecuación de Q-Learning
    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(ACTIONS))
        
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def celebrate_goal(self):
        print("META ALCANZADA...")
        
        # Girar en el propio eje por 3 segundos (aprox 90 grados)
        spin_time = 3.0  # segundos
        spin_steps = int(spin_time * 1000 / self.timestep)  # convertir a pasos
        
        for _ in range(spin_steps):
            self.left_motor.setVelocity(MAX_SPEED * 0.5)
            self.right_motor.setVelocity(-MAX_SPEED * 0.5)
            self.robot.step(self.timestep)
        
        # Detenerse completamente
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        print("Robot detenido")
    
    def save_qtable(self):
        filename = 'qtable.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table guardada: {len(self.q_table)} estados")
    
    def load_qtable(self):
        filename = 'qtable.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table cargada: {len(self.q_table)} estados")
        else:
            print("No se encontró Q-table previa, iniciando desde cero")
    
    def run(self):
        print("Iniciando Q-Learning (Sistema Realista..")

        prev_state = None
        current_state = self.get_sensor_state()

        while self.robot.step(self.timestep) != -1:
            # Elegir y ejecutar acción
            action = self.choose_action(current_state)
            self.execute_action(action)

            # Esperar varios pasos para que el robot se mueva
            for _ in range(5):
                self.robot.step(self.timestep)

            self.steps += 1

            # Obtener nuevo estado
            next_state = self.get_sensor_state()

            # Calcular recompensa (solo sensores, sin posiciones absolutas)
            reward, goal_reached = self.calculate_reward(current_state, next_state)
            self.total_reward += reward

            # Actualizar Q-value
            self.update_q_value(current_state, action, reward, next_state)

            # Imprimir progreso cada 100 pasos
            if self.steps % 100 == 0:
                print(f"Episodio: {self.episode}, Pasos: {self.steps}, "
                      f"Recompensa total: {self.total_reward:.2f}, "
                      f"Estados aprendidos: {len(self.q_table)}")
                print(f"  Última acción: {ACTIONS[action]}")
                print(f"  Estado actual: {current_state}")
                print(f"  Meta detectada: {'SÍ' if self.detect_goal() else 'NO'}")

            # Si alcanzó la meta o pasó mucho tiempo, reiniciar episodio
            if goal_reached or self.steps > 500:
                self.episode += 1
                print(f"\n=== Fin del episodio {self.episode} ===")
                print(f"Recompensa total: {self.total_reward:.2f}")
                print(f"Estados en Q-table: {len(self.q_table)}")
                print(f"Meta alcanzada: {'SÍ' if goal_reached else 'NO'}")

                # Celebrar si se alcanzó la meta
                if goal_reached:
                    self.celebrate_goal()

                # Guardar Q-table cada 10 episodios
                if self.episode % 10 == 0:
                    self.save_qtable()

                # Reiniciar para nuevo episodio
                self.steps = 0
                self.total_reward = 0

                # Esperar un poco antes de continuar
                for _ in range(10):
                    self.robot.step(self.timestep)

            # Actualizar estado previo
            prev_state = current_state
            current_state = next_state

            # Si completó todos los episodios de entrenamiento
            if self.episode >= EPISODES:
                print("\n¡Entrenamiento completado!")
                self.save_qtable()
                break


def main():
    robot_controller = QLearningRobot()
    robot_controller.run()


if __name__ == "__main__":
    main()