import torch

# Configuración del entorno
ENV_NAME = 'CartPole-v1'
NUM_EPISODES = 1000  # Aumentado para más entrenamiento
MAX_STEPS = 500

# Hiperparámetros DQN
LEARNING_RATE = 0.001  # Ajustado para acelerar aprendizaje
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.996  # Decay más gradual para mejor exploración
BATCH_SIZE = 64  # Reducido para estabilidad
MEMORY_SIZE = 20000  # Aumentado para más experiencias
TARGET_UPDATE_FREQ = 5  # Más frecuente

# Arquitectura de la red neuronal
HIDDEN_LAYERS = [256, 128, 64]

# Parámetros de entrenamiento
SAVE_EVERY = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
PRINT_EVERY = 10
SAVE_MODEL_PATH = 'models/'
SAVE_PLOTS_PATH = 'plots/'
