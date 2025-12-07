import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    # Buffer de experiencias para almacenar y muestrear transiciones.
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Almacenar una transición en el buffer.
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # Muestrear un lote aleatorio de transiciones.
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
        return len(self.buffer)


class DQNNetwork(nn.Module):
    # Red neuronal para aproximación de función Q.
    
    def __init__(self, state_dim, action_dim, hidden_layers):
        # Inicializar red DQN.
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    # Agente que soporta tanto DQN como Double DQN.
    
    def __init__(self, state_dim, action_dim, config, device, use_double_dqn=False):
        # Inicializar agente DQN o Double DQN.

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.use_double_dqn = use_double_dqn
        
        # Hiperparámetros
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.batch_size = config.BATCH_SIZE
        self.target_update_freq = config.TARGET_UPDATE_FREQ
        
        # TAU para soft update (clave para evitar olvido catastrófico)
        self.tau = 0.005
        
        # Redes neuronales
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
        self.criterion = nn.SmoothL1Loss()  # Huber Loss - más robusto que MSE
        
        # Buffer de replay
        self.memory = ReplayBuffer(config.MEMORY_SIZE)
        
        # Contadores y métricas
        self.steps = 0
        self.losses = []
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        # Muestrear batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convertir a tensores y mover a GPU
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calcular Q-values actuales
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calcular Q-values objetivo (aquí está la diferencia clave)
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: seleccionar acción con policy_net, evaluar con target_net
                next_actions = self.policy_net(next_states).argmax(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # DQN estándar: seleccionar y evaluar con target_net
                next_q_values = self.target_net(next_states).max(1)[0]
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calcular pérdida y optimizar
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Actualizar epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Incrementar pasos
        self.steps += 1
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self):
        # Soft update: θ_target = τ*θ_policy + (1-τ)*θ_target
        # Esto evita cambios bruscos y previene el olvido catastrófico
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
    
    def store_transition(self, state, action, reward, next_state, done):
        # Almacenar una transición en el buffer de replay.
        self.memory.push(state, action, reward, next_state, done)
    
    def get_algorithm_name(self):
        # Retornar nombre del algoritmo.
        return "Double DQN" if self.use_double_dqn else "DQN"
    
    def get_state_dict(self):
        # Obtener estado del agente para guardar.
        return {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'use_double_dqn': self.use_double_dqn
        }
    
    def load_state_dict(self, state_dict):
        # Cargar estado del agente desde checkpoint.
        self.policy_net.load_state_dict(state_dict['policy_net'])
        self.target_net.load_state_dict(state_dict['target_net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epsilon = state_dict['epsilon']
        self.steps = state_dict['steps']
