# ============================================================================
# PARÁMETROS DE Q-LEARNING
# ============================================================================

# Tasa de aprendizaje
# Controla cuánto se actualizan los valores Q en cada iteración
ALPHA = 0.1

# Factor de descuento
# Controla la importancia de recompensas futuras
GAMMA = 0.9

# Tasa de exploración
# Probabilidad de tomar una acción aleatoria vs la mejor acción
EPSILON = 0.3

# Epsilon decay (opcional)
# Reduce epsilon gradualmente durante el entrenamiento
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05

EPISODES = 1000

MAX_STEPS_PER_EPISODE = 1000

# ========================
# PARÁMETROS DEL ROBOT
# ========================

MAX_SPEED = 6.28
TIME_STEP = 64

# ==========================
# SISTEMA DE RECOMPENSAS
# ==========================

REWARD_GOAL = 100
REWARD_CLOSER = 1.0
PENALTY_FARTHER = -0.5
PENALTY_OBSTACLE_CLOSE = -5.0
PENALTY_OBSTACLE_NEAR = -1.0
PENALTY_STEP = -0.1

# =============================
# DISCRETIZACIÓN DE SENSORES
# =============================

# Los sensores e-puck devuelven valores de 0-4095
SENSOR_THRESHOLD_NEAR = 100
SENSOR_THRESHOLD_CLOSE = 500
# 0: Libre, 1: Cerca, 2: Muy cerca
NUM_SENSOR_LEVELS = 3
# Sensores a usar para el estado (índices 0-7)
RELEVANT_SENSORS = [0, 2, 5, 7]

# ==========================
# POSICIONES DEL MUNDO
# ==========================

# Posición de inicio (cuadro verde)
START_POS = [-0.37, -0.39]

# Posición objetivo (cuadro cyan)
GOAL_POS = [0.35, 0.36]

# Tolerancia para considerar que se alcanzó la meta (metros)
GOAL_TOLERANCE = 0.15

# =========================
# GUARDADO Y CARGA
# =========================

QTABLE_FILENAME = 'qtable.pkl'
SAVE_FREQUENCY = 10
LOAD_QTABLE = True

# ====================
# LOGGING Y DEBUG
# ====================

# Mostrar información cada N pasos
LOG_FREQUENCY = 100
VERBOSE_SENSORS = False
VERBOSE_REWARDS = False
SAVE_LOG = False
LOG_FILENAME = 'training_log.txt'

# ==========================
# ACCIONES DEL ROBOT
# ==========================

# Definición de acciones disponibles
ACTIONS = {
    'FORWARD': (1.0, 1.0),           # Avanzar recto
    'TURN_LEFT': (-0.5, 0.5),        # Girar izquierda
    'TURN_RIGHT': (0.5, -0.5),       # Girar derecha
    'BACK_LEFT': (-0.5, -0.3),       # Retroceder girando izquierda
    'BACK_RIGHT': (-0.3, -0.5),      # Retroceder girando derecha
}

# Nombres de acciones (deben coincidir con las llaves de ACTIONS)
ACTION_NAMES = list(ACTIONS.keys())

# ==================================
# PRESETS DE CONFIGURACIÓN
# ==================================

def get_exploration_config():
    return {
        'ALPHA': 0.2,
        'GAMMA': 0.95,
        'EPSILON': 0.5,
        'MAX_SPEED': 5.0
    }

def get_exploitation_config():
    return {
        'ALPHA': 0.05,
        'GAMMA': 0.95,
        'EPSILON': 0.1,
        'MAX_SPEED': 6.28
    }

def get_fast_training_config():
    return {
        'ALPHA': 0.3,
        'GAMMA': 0.85,
        'EPSILON': 0.4,
        'EPISODES': 100,
        'MAX_STEPS_PER_EPISODE': 500
    }

# ============================================================================
# NOTAS Y RECOMENDACIONES
# ============================================================================

"""
1. Si el robot colisiona mucho:
   - Aumenta PENALTY_OBSTACLE_CLOSE y PENALTY_OBSTACLE_NEAR
   - Reduce MAX_SPEED
   - Aumenta EPSILON para más exploración

2. Si el aprendizaje es muy lento:
   - Aumenta ALPHA
   - Reduce NUM_SENSOR_LEVELS para menos estados
   - Usa menos RELEVANT_SENSORS

3. Si el robot no encuentra la meta:
   - Aumenta REWARD_GOAL
   - Aumenta REWARD_CLOSER
   - Verifica que GOAL_TOLERANCE no sea muy pequeño
   - Aumenta EPSILON

4. Para afinar una política ya aprendida:
   - Reduce EPSILON a 0.05-0.1
   - Reduce ALPHA a 0.01-0.05
   - Mantén GAMMA alto (0.9-0.95)

5. Para análisis detallado:
   - Activa VERBOSE_SENSORS y VERBOSE_REWARDS
   - Activa SAVE_LOG
   - Reduce LOG_FREQUENCY a 50 o menos
"""
