# LABORATORIO 4 - DQN CON ROBOT E-PUCK

## Descripción del Proyecto

Este proyecto implementa un algoritmo de Deep Q-Network (DQN) para navegación robótica realista con el robot e-puck en Webots. Utiliza una red neuronal profunda para aproximar valores Q, permitiendo manejar espacios de estado continuos y mejorar el rendimiento en tareas complejas de navegación. El sistema emplea únicamente sensores locales, sin posiciones GPS absolutas, lo que lo hace replicable en entornos robóticos reales.

### Características Principales
- **DQN**: Red neuronal profunda con experiencia replay, target network y ε-decay para aprendizaje estable.
- **Sistema realista**: Reinicio manual, detección de meta basada en patrones de sensores, sin mapas pre-programados.
- **Visualización**: Gráficos de progreso de entrenamiento y análisis de políticas aprendidas.
- **Optimización**: Modelos guardados automáticamente para evaluación y despliegue.

### Algoritmo Implementado
- **DQN**: Aprendizaje con red neuronal, experiencia replay para estabilidad, y target network para reducir sobreestimación de valores Q.

### Parámetros Principales
- Learning Rate: 0.001
- Gamma: 0.99 (factor de descuento)
- Epsilon Inicial: 1.0 (exploración)
- Epsilon Decay: 0.995
- Epsilon Mínimo: 0.01
- Batch Size: 32
- Memory Size: 2000
- Episodios: 1000

## Estructura del Proyecto

```
lab-4-webots/
├── controllers/
│   └── epuck_dqn/
│       ├── epuck_dqn.py          # Controlador DQN principal
│       ├── best_dqn_model.keras  # Mejor modelo DQN guardado
│       ├── requirements.txt      # Dependencias Python
│       ├── config.py             # Configuración DQN
│       ├── visualize_dqn.py      # Script de visualización
│       └── dqn_training_progress.png # Progreso de entrenamiento
├── worlds/
│   ├── main-dqn.wbt              # Mundo principal para DQN
│   └── dqn-maze.wbt              # Laberinto para pruebas avanzadas
├── libraries/                    # Librerías auxiliares
├── plugins/                      # Plugins de Webots
├── protos/                       # Prototipos de objetos
├── README_DQN.md                 # Documentación específica DQN
├── GUIA_VENV.md                  # Guía de entorno virtual
└── README.md                     # Este archivo
```

## Requisitos
- Webots R2025a o superior
- Python 3.8+ con TensorFlow/Keras y NumPy
- Dependencias adicionales listadas en `controllers/epuck_dqn/requirements.txt`

## Instalación y Ejecución

### Configuración Inicial
1. Instalar dependencias: `pip install -r controllers/epuck_dqn/requirements.txt`
2. Verificar instalación de Webots y Python.

### Ejecución
1. Abrir Webots y cargar `worlds/main-dqn.wbt` o `worlds/dqn-maze.wbt`.
2. Configurar controlador del robot como `epuck_dqn`.
3. Colocar robot en posición inicial y presionar Play.
4. El modelo se entrena y guarda automáticamente; reposicionar manualmente el robot entre episodios si es necesario.
5. Usar `visualize_dqn.py` para analizar resultados y progreso.

## Parámetros Configurables
- Ajustar learning rate, gamma, epsilon decay en `epuck_dqn.py`.
- Modificar umbrales de sensores y recompensas en `config.py`.

## Resultados Esperados
- Convergencia del modelo DQN con reducción en pasos hasta la meta.
- Mejor rendimiento en entornos complejos como laberintos.
- Modelos guardados listos para evaluación y despliegue.
