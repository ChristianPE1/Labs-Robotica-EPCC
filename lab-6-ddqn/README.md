# Lab 6: Comparación DQN vs Double DQN

Este proyecto implementa y compara Deep Q-Network (DQN) y Double DQN en el entorno CartPole-v1 de Gymnasium usando **PyTorch con CUDA (GPU obligatoria)**.

## ⚠️ Requisitos IMPORTANTES

- **GPU con CUDA es OBLIGATORIA** - El proyecto está optimizado para aprovechar aceleración GPU
- PyTorch con CUDA instalado
- CUDA Toolkit 11.8 o superior

## Instalación

```bash
# Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar PyTorch con CUDA (ejemplo para CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Instalar otras dependencias
pip install gymnasium numpy matplotlib
```

## Verificación de CUDA

**PASO IMPORTANTE:** Antes de entrenar, verifica que CUDA esté correctamente configurado:

```bash
python verify_cuda.py
```

Deberías ver:
```
✓ PyTorch version: 2.x.x+cuXXX
✓ CUDA disponible: True
✓ Dispositivo GPU: Tesla T4 (o tu GPU)
✓ Memoria GPU total: X.XX GB
✅ TODAS LAS VERIFICACIONES PASARON
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

## Hiperparámetros Optimizados para GPU

| Parámetro | Valor | Nota |
|-----------|-------|------|
| Episodios | **2000** | ↑ Aprovecha velocidad GPU |
| Learning Rate | **0.0005** | ↓ Más estable |
| Gamma (descuento) | 0.99 | - |
| Batch Size | **128** | ↑ Mejor uso GPU |
| Memory Size | **50,000** | ↑ Más experiencia |
| Epsilon inicial | 1.0 | - |
| Epsilon final | 0.01 | - |
| Epsilon decay | **0.9995** | ↓ Decay más lento |
| Hidden Layers | **[256, 256, 128]** | ↑ Red más profunda |
| Target Update Freq | **50 episodios** | ↑ Más estabilidad |

## Diferencias clave DQN vs Double DQN

**DQN estándar:**
```python
target = r + γ * max_a' Q(s', a'; θ⁻)
```

**Double DQN:**
```python
a* = argmax_a' Q(s', a'; θ)  # Red de política selecciona
target = r + γ * Q(s', a*; θ⁻)  # Red objetivo evalúa
```

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
