import torch

# Configuración del entorno
ENV_NAME = 'CartPole-v1'
NUM_EPISODES = 500
MAX_STEPS = 500

# Hiperparámetros DQN
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10

# Arquitectura de la red neuronal
HIDDEN_LAYERS = [128, 128]

# Parámetros de entrenamiento
SAVE_EVERY = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
PRINT_EVERY = 10
SAVE_MODEL_PATH = 'models/'
SAVE_PLOTS_PATH = 'plots/'
