# Laboratorio 5: Deep Q-Network con CUDA

Implementación de Deep Q-Network (DQN) usando PyTorch con aceleración CUDA para entrenamiento en Google Colab.

## Resumen

Este laboratorio demuestra aprendizaje por refuerzo usando el algoritmo DQN en el entorno CartPole-v1 de Gymnasium. La implementación soporta aceleración GPU a través de CUDA, haciéndolo ideal para entrenamiento en las instancias GPU gratuitas de Google Colab.

## Características

- **Implementación PyTorch**: Framework moderno de aprendizaje profundo con soporte CUDA
- **Replay de Experiencias**: Buffer de memoria eficiente para entrenamiento estable
- **Red Objetivo**: Red objetivo separada actualizada periódicamente para estabilidad
- **Exploración Epsilon-Greedy**: Estrategia balanceada de exploración-explotación
- **Puntos de Control**: Guardar y reanudar progreso de entrenamiento
- **Visualizaciones Profesionales**: Gráficas de alta calidad para análisis y reportes

## Estructura del Proyecto

```
dqn_gymnasium/
├── config.py              # Hiperparámetros y configuración
├── dqn_agent.py          # Implementación del agente DQN
├── train.py              # Script de entrenamiento
├── visualize.py          # Visualización y análisis
├── utils.py              # Funciones auxiliares
├── dqn_training.ipynb    # Notebook de Google Colab
├── requirements.txt      # Dependencias de Python
├── checkpoints/          # Modelos guardados (creados durante entrenamiento)
├── metrics/              # Métricas de entrenamiento (creadas durante entrenamiento)
└── plots/                # Visualizaciones generadas (creadas durante entrenamiento)
```

## Requisitos

- Python 3.8+
- PyTorch con soporte CUDA
- Gymnasium
- NumPy
- Matplotlib

## Instrucciones de Configuración

### Opción 1: Google Colab (Recomendado)

1. Abre el notebook de Colab: [dqn_training.ipynb](dqn_gymnasium/dqn_training.ipynb)
2. Asegúrate de que el runtime GPU esté habilitado:
   - Runtime → Change runtime type → Hardware accelerator → GPU
3. Ejecuta todas las celdas secuencialmente

El notebook maneja:
- Instalación de dependencias
- Clonado del repositorio
- Ejecución del entrenamiento
- Generación de visualizaciones
- Descarga de resultados

### Opción 2: Entorno Local

1. Clona el repositorio:
```bash
git clone https://github.com/ChristianPE1/Labs-Robotica-EPCC.git
cd Labs-Robotica-EPCC/lab-5-webots/dqn_gymnasium
```

2. Instala dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta el entrenamiento:
```bash
python train.py
```

4. Genera visualizaciones:
```bash
python visualize.py
```

## Configuración

Los hiperparámetros pueden modificarse en `config.py`:

| Parámetro | Descripción | Valor por Defecto |
|-----------|-------------|-------------------|
| `ENV_NAME` | Entorno de Gymnasium | CartPole-v1 |
| `NUM_EPISODES` | Episodios de entrenamiento | 500 |
| `LEARNING_RATE` | Tasa de aprendizaje del optimizador | 0.001 |
| `GAMMA` | Factor de descuento | 0.99 |
| `EPSILON_START` | Tasa inicial de exploración | 1.0 |
| `EPSILON_END` | Tasa mínima de exploración | 0.01 |
| `EPSILON_DECAY` | Tasa de decaimiento de exploración | 0.995 |
| `BATCH_SIZE` | Tamaño del lote de entrenamiento | 64 |
| `MEMORY_SIZE` | Capacidad del buffer de replay | 10000 |
| `HIDDEN_LAYERS` | Arquitectura de red neuronal | [128, 128] |
| `TARGET_UPDATE_FREQ` | Frecuencia de actualización de red objetivo | 100 |

## Proceso de Entrenamiento

El script de entrenamiento realiza los siguientes pasos:

1. **Configuración del Entorno**: Inicializar entorno de Gymnasium
2. **Inicialización del Agente**: Crear agente DQN con redes de política y objetivo
3. **Bucle de Entrenamiento**:
   - Interactuar con el entorno usando política epsilon-greedy
   - Almacenar transiciones en buffer de replay
   - Muestrear mini-lotes y actualizar red de política
   - Actualizar periódicamente red objetivo
4. **Puntos de Control**: Guardar modelos y métricas cada 50 episodios
5. **Guardado Final**: Guardar modelo completado y métricas completas

### Tiempo Esperado de Entrenamiento

- **Google Colab (GPU)**: ~10-15 minutos para 500 episodios
- **Local (CPU)**: ~30-45 minutos para 500 episodios

## Resultados y Visualización

El script de visualización genera cuatro gráficas clave:

1. **Curva de Recompensas**: Recompensas por episodio con promedio móvil de 100 episodios
2. **Longitud de Episodio**: Número de pasos por episodio a lo largo del tiempo
3. **Curva de Pérdida**: Progresión de pérdida de entrenamiento
4. **Tasa de Éxito**: Porcentaje de episodios que alcanzan longitud máxima (500 pasos)

Todas las gráficas se guardan en el directorio `plots/` a 300 DPI para calidad de publicación.

## Métricas de Rendimiento

Un entrenamiento exitoso típicamente alcanza:

- Recompensa promedio: > 450
- Longitud promedio de episodio: > 475 pasos
- Tasa de éxito: > 90% en los últimos 100 episodios
- Convergencia: Dentro de 300-400 episodios

## Detalles de Implementación

### Algoritmo DQN

La implementación sigue el algoritmo DQN estándar:

1. **Observación**: Recibir estado del entorno
2. **Selección de Acción**: Elegir acción usando política epsilon-greedy
3. **Almacenamiento de Transición**: Guardar (estado, acción, recompensa, siguiente_estado, terminado) en buffer de replay
4. **Muestreo de Lote**: Muestrear mini-lote aleatorio del buffer
5. **Cálculo de Valores Q**: Calcular valores Q actuales y objetivo
6. **Cálculo de Pérdida**: Calcular pérdida MSE entre valores Q actuales y objetivo
7. **Actualización de Red**: Retropropagar y optimizar red de política
8. **Actualización Objetivo**: Copiar periódicamente pesos de red de política a red objetivo

### Arquitectura de Red

- **Capa de Entrada**: Dimensión de estado (4 para CartPole)
- **Capas Ocultas**: Dos capas de 128 neuronas con activación ReLU
- **Capa de Salida**: Dimensión de acción (2 para CartPole)

## Licencia

Este proyecto es parte del curso de Laboratorio de Robótica en EPCC.
