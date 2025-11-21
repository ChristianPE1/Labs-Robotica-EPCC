# Funciones de utilidad para entrenamiento y evaluación

import os
import json
import numpy as np
import torch


def save_checkpoint(model, optimizer, episode, metrics, filepath):
    # Guardar checkpoint del modelo con métricas de entrenamiento
    # model: Modelo PyTorch a guardar
    # optimizer: Estado del optimizador
    # episode: Número de episodio actual
    # metrics: Diccionario de métricas de entrenamiento
    # filepath: Ruta para guardar el checkpoint
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    # Cargar checkpoint del modelo:
    # Returna tupla de (episodio, métricas)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode'], checkpoint['metrics']


def save_metrics(metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convertir arrays numpy a listas para serialización JSON
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, list):
            serializable_metrics[key] = value
        else:
            serializable_metrics[key] = value

    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)


def load_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_statistics(data, window=50):
    # Calcular promedio móvil e intervalo de confianza
    if len(data) < window:
        return np.array(data), np.zeros(len(data))

    moving_avg = []
    moving_std = []

    for i in range(len(data)):
        start = max(0, i - window + 1)
        window_data = data[start:i+1]
        moving_avg.append(np.mean(window_data))
        moving_std.append(np.std(window_data))

    return np.array(moving_avg), np.array(moving_std)


def get_device_info():
    # Obtener información del dispositivo de cómputo disponible
    # retorna diccionario con información del dispositivo
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            'cuda_available': True,
            'device_name': device_name,
            'device_memory': device_memory
        }
    else:
        return {
            'cuda_available': False,
            'device_name': 'CPU',
            'device_memory': 0.0
        }