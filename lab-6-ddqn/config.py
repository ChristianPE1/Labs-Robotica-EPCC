import torch

# CONFIGURACIÓN DEL ENTORNO
ENV_NAME = 'CartPole-v1'
NUM_EPISODES = 1000  # Reducido, suficiente para CartPole
MAX_STEPS = 500

# HIPERPARÁMETROS COMUNES (Configuración estable probada)
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05  # Mantener algo de exploración siempre
EPSILON_DECAY = 0.997  # Decay más lento para explorar más tiempo
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100  # Actualizar target menos frecuente = más estabilidad

# ARQUITECTURA DE RED NEURONAL
HIDDEN_LAYERS = [128, 128]  # Red más grande para mejor representación

# PARÁMETROS DE ENTRENAMIENTO
SAVE_EVERY = 100
PRINT_EVERY = 50

# EARLY STOPPING (cuando alcance buen rendimiento)
EARLY_STOP_THRESHOLD = 475
EARLY_STOP_PATIENCE = 20

if not torch.cuda.is_available():
    raise RuntimeError("CUDA no disponible")

DEVICE = torch.device("cuda")
print(f"Usando GPU: {torch.cuda.get_device_name(0)}")

# RUTAS DE GUARDADO
SAVE_MODEL_PATH = 'models/'
SAVE_PLOTS_PATH = 'plots/'
SAVE_METRICS_PATH = 'metrics/'

# SEMILLA PARA REPRODUCIBILIDAD
RANDOM_SEED = 42
