import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    # Buffer de replay de experiencias para almacenar y muestrear transiciones

    def __init__(self, capacity):
        # Inicializar buffer de replay con capacidad fija
        # capacity: Número máximo de transiciones a almacenar
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Almacenar una transición en el buffer
        # state: Estado actual de observación
        # action: Acción tomada
        # reward: Recompensa recibida
        # next_state: Observación del siguiente estado
        # done: Flag de terminación del episodio
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Muestrear un lote aleatorio de transiciones
        # retorna tupla de (estados, acciones, recompensas, siguientes_estados, terminados) en lotes
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )

    def __len__(self):
        # Retornar el tamaño actual del buffer
        return len(self.buffer)


class DQNNetwork(nn.Module):
    # Arquitectura de red neuronal Deep Q-Network

    def __init__(self, state_dim, action_dim, hidden_layers):
        # Inicializar red DQN
        # state_dim: Dimensión del espacio de estados
        # action_dim: Dimensión del espacio de acciones
        # hidden_layers: Lista de tamaños de capas ocultas
        super(DQNNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: Tensor de estado de entrada
        # Returna valores Q para cada acción
        return self.network(x)


class DQNAgent:
    # Agente Deep Q-Network con replay de experiencias y red objetivo

    def __init__(self, state_dim, action_dim, config, device):
        # Inicializar agente DQN
        # state_dim: Dimensión del espacio de estados
        # action_dim: Dimensión del espacio de acciones
        # config: Objeto de configuración con hiperparámetros
        # device: Dispositivo PyTorch (CPU o CUDA)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Hiperparámetros
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.batch_size = config.BATCH_SIZE
        self.target_update_freq = config.TARGET_UPDATE_FREQ

        # Redes
        self.policy_net = DQNNetwork(
            state_dim, action_dim, config.HIDDEN_LAYERS
        ).to(device)
        self.target_net = DQNNetwork(
            state_dim, action_dim, config.HIDDEN_LAYERS
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizador y función de pérdida
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.MSELoss()

        # Buffer de replay
        self.memory = ReplayBuffer(config.MEMORY_SIZE)

        # Métricas de entrenamiento
        self.steps = 0
        self.losses = []
    
    def select_action(self, state, training=True, epsilon=None):
        # Seleccionar acción usando política epsilon-greedy
        # state: Observación del estado actual
        # training: Si está en modo de entrenamiento (habilita exploración)
        # epsilon: Valor de epsilon opcional para forzar exploración
        # Returns: Índice de acción seleccionada
        eps = epsilon if epsilon is not None else self.epsilon
        if training and random.random() < eps:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self):
        # Realizar un paso de entrenamiento usando replay de experiencias
        # Valor de pérdida si se realizó entrenamiento, None en caso contrario
        if len(self.memory) < self.batch_size:
            return None

        # Muestrear lote
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convertir a tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Calcular valores Q actuales
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calcular valores Q objetivo
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calcular pérdida
        loss = self.criterion(current_q_values, target_q_values)

        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Actualizar epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Actualizar red objetivo
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value
    
    def store_transition(self, state, action, reward, next_state, done):
        # Almacenar una transición en el buffer de replay
        # done: Flag de terminación del episodio
        self.memory.push(state, action, reward, next_state, done)

    def get_state_dict(self):
        # Obtener estado del agente para guardar
        return {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }

    def load_state_dict(self, state_dict):
        # Cargar estado del agente desde checkpoint
        self.policy_net.load_state_dict(state_dict['policy_net'])
        self.target_net.load_state_dict(state_dict['target_net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epsilon = state_dict['epsilon']
        self.steps = state_dict['steps']
