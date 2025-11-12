# ====================================# Red Neuronal (MLP - Multilayer Perceptron) - MÁS SIMPLE PARA VELOCIDAD
INPUT_SIZE = 12            # 8 sensores + 2 LIDAR + 2 posición relativa a meta
HIDDEN_LAYERS = [32]       # REDUCIDO: Una sola capa de 32 neuronas para mayor velocidad
OUTPUT_SIZE = 8            # 8 acciones posibles (más exploratorias)====================================
# CONFIGURACIÓN DE DQN (Deep Q-Network)
# ============================================================================

# =======================
# HIPERPARÁMETROS DQN
# =======================

# Tasa de aprendizaje para la red neuronal
LEARNING_RATE = 0.001

# Factor de descuento (importancia de recompensas futuras)
GAMMA = 0.99

# Exploración vs Explotación - BALANCEADO
EPSILON_START = 0.5         # Exploración inicial (50% - balanceado)
EPSILON_END = 0.01          # Exploración mínima (1%)
EPSILON_DECAY = 0.995       # Decaimiento moderado

# Replay Buffer (Memoria de experiencias)
REPLAY_BUFFER_SIZE = 2000  # REDUCIDO: Menos memoria para mayor velocidad
BATCH_SIZE = 64            # AUMENTADO: Más eficiente en CPU
MIN_REPLAY_SIZE = 200      # REDUCIDO: Comenzar a aprender antes

# Target Network
TARGET_UPDATE_FREQUENCY = 10  # Cada cuántos episodios actualizar target network

# =======================
# PARÁMETROS DEL ROBOT
# =======================

MAX_SPEED = 6.28  # Velocidad máxima del E-puck (rad/s)
TIME_STEP = 64

# =======================
# ARQUITECTURA DE LA RED
# =======================

# Red Neuronal (MLP - Multilayer Perceptron) - MÁS SIMPLE PARA VELOCIDAD
INPUT_SIZE = 12            # 8 sensores + 2 LIDAR + 2 posición relativa a meta
HIDDEN_LAYERS = [32]       # REDUCIDO: Una sola capa de 32 neuronas para mayor velocidad
OUTPUT_SIZE = 8            # 8 acciones posibles (más exploratorias)

# =======================
# ENTRENAMIENTO
# =======================

EPISODES = 500             # AUMENTADO: Más episodios con configuración más rápida
MAX_STEPS_PER_EPISODE = 200 # REDUCIDO: Episodios más cortos para mayor velocidad

# ==========================
# SISTEMA DE RECOMPENSAS (ESTRATEGIA: BUSCAR ESPACIO ABIERTO)
# ==========================

# OBJETIVO: Escapar del laberinto = Encontrar área completamente abierta
REWARD_GOAL = 200.0               # ¡ÉXITO! Encontró espacio libre (escapó)
REWARD_OPEN_SPACE = 1.0           # Recompensa por mantener espacio libre
PENALTY_OBSTACLE_VERY_CLOSE = -10.0  # Penalización fuerte por obstáculo muy cerca
PENALTY_OBSTACLE_CLOSE = -5.0     # Penalización por obstáculo cerca
PENALTY_OBSTACLE_NEAR = -2.0      # Penalización por obstáculo moderadamente cerca
PENALTY_STEP = -0.1               # Penalización pequeña por cada paso (eficiencia)

# ==========================
# DETECCIÓN DE OBSTÁCULOS
# ==========================

# Valores de los sensores e-puck (0-4095)
SENSOR_THRESHOLD_VERY_CLOSE = 800  # Obstáculo muy cerca (colisión inminente)
SENSOR_THRESHOLD_CLOSE = 400       # Obstáculo cerca
SENSOR_THRESHOLD_NEAR = 150        # Obstáculo detectado

# ==========================
# POSICIONES DEL MUNDO
# ==========================

# Posición de inicio del robot (solo para referencia)
START_POS = [-1.2, -1.2, 0.01]  # [x, y, z]

# Posición de la meta (solo para referencia y cálculo de recompensas)
# NOTA: En un entorno real, el robot usaría detección visual/sensores
GOAL_POS = [1.2, 1.2]  # [x, y]

# Tolerancia para considerar que se alcanzó la meta (metros)
GOAL_TOLERANCE = 0.3

# ==========================
# DETECCIÓN DE META (ÁREA ABIERTA)
# ==========================

# Detección de zona de meta usando LIDAR (OBJETIVO = ESCAPAR)
# El robot busca áreas con MUCHO espacio libre (sin obstáculos cercanos)
LIDAR_GOAL_THRESHOLD = 1.0  # Meta: obstáculos a más de 1.0m (muy lejos)
LIDAR_MIN_CLEAR_RATIO = 0.80  # 80% de los puntos deben estar muy lejos

# ESTRATEGIA: El robot aprende a escapar del laberinto buscando espacio abierto
# Similar al Q-Learning previo, pero con DQN para mejor generalización

# ====================
# ACCIONES DEL ROBOT
# ====================

# Definición de acciones disponibles - MÁS EXPLORATORIAS
# [velocidad_motor_izquierdo, velocidad_motor_derecho]
# Valores multiplicados por MAX_SPEED (6.28 rad/s)
ACTIONS = {
    0: [0.8, 0.8],      # FORWARD - Avanzar recto (moderado)
    1: [-0.3, 0.8],     # TURN_LEFT - Giro suave izquierda
    2: [0.8, -0.3],     # TURN_RIGHT - Giro suave derecha
    3: [-0.6, 0.6],     # SHARP_LEFT - Giro más pronunciado izquierda
    4: [0.6, -0.6],     # SHARP_RIGHT - Giro más pronunciado derecha
    5: [0.0, 0.0],      # STOP - Detenerse (para pensar)
    6: [0.4, 0.8],      # SLIGHT_LEFT - Muy ligero giro izquierda
    7: [0.8, 0.4],      # SLIGHT_RIGHT - Muy ligero giro derecha
}

ACTION_NAMES = ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'SHARP_LEFT', 'SHARP_RIGHT', 'STOP', 'SLIGHT_LEFT', 'SLIGHT_RIGHT']

# =========================
# GUARDADO Y CARGA
# =========================

MODEL_SAVE_PATH = 'dqn_model.pth'
CHECKPOINT_FREQUENCY = 50  # Guardar cada N episodios
SAVE_BEST_MODEL = True     # Guardar el mejor modelo

# ====================
# LOGGING Y DEBUG
# ====================

LOG_FREQUENCY = 10         # Mostrar información cada N pasos
SAVE_TRAINING_LOG = True
LOG_FILENAME = 'dqn_training_log.txt'
PLOT_FREQUENCY = 10        # Actualizar gráficas cada N episodios

# ==================================
# DEVICE (CPU/GPU)
# ==================================

USE_GPU = False  # Cambiar a True si tienes GPU disponible
DEVICE = "cuda" if USE_GPU else "cpu"

# ============================================================================
# NOTAS Y RECOMENDACIONES PARA DQN
# ============================================================================

"""
1. DQN vs Q-Learning:
   - DQN usa redes neuronales → puede manejar espacios de estados continuos
   - Q-Learning usa tabla → solo para estados discretos
   - DQN aprende representaciones → generaliza mejor

2. Componentes clave de DQN:
   - Red neuronal Q (policy network)
   - Red objetivo (target network) → estabilidad
   - Replay buffer → rompe correlación temporal
   - Epsilon-greedy → exploración/explotación

3. Si el entrenamiento es inestable:
   - Reduce LEARNING_RATE (0.0001)
   - Aumenta TARGET_UPDATE_FREQUENCY
   - Aumenta MIN_REPLAY_SIZE
   - Reduce BATCH_SIZE

4. Si el robot no aprende:
   - Aumenta EPSILON_DECAY (más exploración)
   - Ajusta las recompensas (aumenta REWARD_GOAL)
   - Revisa que las recompensas no dominen
   - Aumenta HIDDEN_LAYERS

5. Para acelerar el entrenamiento:
   - Reduce MAX_STEPS_PER_EPISODE
   - Reduce REPLAY_BUFFER_SIZE
   - Aumenta BATCH_SIZE (si tienes RAM)
   - Usa GPU (USE_GPU = True)

6. Consejos para el mundo:
   - Empieza con laberinto simple
   - Incrementa complejidad gradualmente
   - Asegura que existe camino a la meta
   - Coloca meta en zona abierta (fácil detectar)
"""
