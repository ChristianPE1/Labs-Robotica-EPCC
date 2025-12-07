# Lab 6: Comparación DQN vs Double DQN

Este proyecto implementa y compara Deep Q-Network (DQN) y Double DQN en el entorno CartPole-v1 de Gymnasium.

## Descripción

Double DQN es una mejora sobre DQN estándar que reduce el problema de sobreestimación de valores Q mediante el desacoplamiento de la selección y evaluación de acciones.

### Diferencia Clave

**DQN estándar:**
```
target = r + γ * max_a' Q(s', a'; θ⁻)
```

**Double DQN:**
```
a* = argmax_a' Q(s', a'; θ)  # Red de política selecciona
target = r + γ * Q(s', a*; θ⁻)  # Red objetivo evalúa
```

## Estructura del Proyecto

```
lab-6-ddqn/
├── config.py              # Hiperparámetros compartidos
├── dqn_agent.py           # Agente unificado DQN/DDQN
├── utils.py               # Utilidades (guardar/cargar)
├── train.py               # Script de entrenamiento
├── visualize.py           # Gráficos comparativos
├── ddqn_comparison.ipynb  # Notebook con experimentos
├── main.tex               # Informe LaTeX
├── requirements.txt       # Dependencias
└── README.md              # Este archivo
```

## Instalación

```bash
# Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Entrenamiento desde Terminal

```bash
# Entrenar DQN estándar
python train.py --algorithm dqn

# Entrenar Double DQN
python train.py --algorithm ddqn
```

### Uso del Notebook

1. Abrir `ddqn_comparison.ipynb` en Jupyter o VS Code
2. Ejecutar todas las celdas secuencialmente
3. Los gráficos comparativos se generarán automáticamente

### Generar Gráficos Comparativos

```python
from visualize import generate_all_comparisons

generate_all_comparisons('dqn_metrics.pkl', 'ddqn_metrics.pkl')
```

## Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Episodios | 500 |
| Learning Rate | 0.001 |
| Gamma (descuento) | 0.99 |
| Batch Size | 64 |
| Memory Size | 10,000 |
| Epsilon inicial | 1.0 |
| Epsilon final | 0.01 |
| Epsilon decay | 0.995 |
| Hidden Layers | [128, 128] |
| Target Update Freq | 10 episodios |

## Métricas de Evaluación

- **Recompensa por episodio**: Suma de recompensas obtenidas
- **Tasa de éxito**: % de episodios con ≥450 pasos
- **Longitud del episodio**: Pasos antes de terminar
- **Pérdida (Loss)**: Error durante entrenamiento

## Resultados Esperados

Double DQN típicamente muestra:
- ✅ Menor sobreestimación de valores Q
- ✅ Entrenamiento más estable
- ✅ Convergencia más rápida
- ✅ Menor varianza en recompensas

## Gráficos Generados

- `comparison_rewards.png` - Recompensas DQN vs DDQN
- `comparison_lengths.png` - Longitud de episodios
- `comparison_losses.png` - Pérdidas durante entrenamiento
- `comparison_success_rates.png` - Tasas de éxito

## Compilar Informe LaTeX

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Referencias

- [Deep Q-Network (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Double DQN (van Hasselt et al., 2016)](https://arxiv.org/abs/1509.06461)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## Autor

Christian Peñaranda - Universidad Nacional de San Agustín

## Licencia

Este proyecto es parte del curso de Robótica - UNSA 2024.
