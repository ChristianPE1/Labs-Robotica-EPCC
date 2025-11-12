# LABORATORIO 3 - Q-LEARNING REALISTA CON ROBOT E-PUCK

## Descripción del Proyecto

Este proyecto implementa un algoritmo de Q-Learning para entrenar a un robot e-puck a navegar hacia áreas abiertas evitando obstáculos, utilizando únicamente información de sensores locales. El sistema es completamente realista, ya que no depende de posiciones GPS absolutas ni mapas pre-programados, lo que lo hace replicable en entornos robóticos reales.

### Características Principales
- **Aprendizaje basado en sensores**: Utiliza 8 sensores de distancia discretizados en 3 niveles para definir estados.
- **Acciones disponibles**: 5 acciones posibles (avanzar, girar izquierda, girar derecha, retroceder izquierda, retroceder derecha).
- **Sistema de recompensas**: Basado en cambios locales en las lecturas de sensores, sin posiciones absolutas.
- **Detección de meta**: Patrón de sensores que indica áreas abiertas (al menos 6 de 8 sensores detectan espacio abierto).
- **Reinicio manual**: El robot se detiene al final de cada episodio y requiere reposicionamiento manual por el operador.

### Parámetros del Algoritmo
- **ALPHA**: 0.1 (tasa de aprendizaje)
- **GAMMA**: 0.9 (factor de descuento)
- **EPSILON**: 0.3 (exploración vs explotación)
- **EPISODES**: 1000 (número total de episodios)

## Estructura del Proyecto

```
lab-3-webots/
├── controllers/
│   └── epuck_qlearning/
│       ├── epuck_qlearning.py     # Controlador principal con Q-Learning
│       ├── qtable.pkl            # Tabla Q aprendida (generada automáticamente)
│       ├── config.py             # Configuración opcional
│       ├── visualize_learning.py # Script para visualizar el aprendizaje
│       └── qtable_analysis.png   # Análisis de la tabla Q
├── worlds/
│   └── main-world.wbt            # Mundo de simulación en Webots
├── libraries/                    # Librerías auxiliares
├── plugins/                      # Plugins de Webots
└── protos/                       # Prototipos de objetos
```

## Requisitos
- Webots R2025a o superior
- Python 3 con NumPy instalado

## Instalación y Ejecución
1. Verificar que Webots esté instalado y Python 3 con NumPy disponible.
2. Abrir Webots y cargar el mundo `worlds/main-world.wbt`.
3. Asegurarse de que el controlador del robot e-puck esté configurado como `epuck_qlearning`.
4. Colocar el robot en una posición inicial arbitraria.
5. Presionar Play para iniciar la simulación y el aprendizaje.
6. Al finalizar cada episodio, reposicionar manualmente el robot y repetir hasta completar los episodios.

## Parámetros Configurables
Los parámetros principales se pueden ajustar en `epuck_qlearning.py`:
- Tasa de aprendizaje (ALPHA)
- Factor de descuento (GAMMA)
- Nivel de exploración (EPSILON)
- Umbrales de distancia para detección de obstáculos y meta

## Resultados Esperados
- El robot aprende a navegar eficientemente hacia áreas abiertas.
- Mejora en recompensas totales y reducción en pasos hasta la meta con más episodios.
- Tabla Q con estados aprendidos basados en configuraciones de sensores.
