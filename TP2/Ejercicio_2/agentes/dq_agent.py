from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle
import random

class QAgent(Agent):
    """
    Agente de Q-Learning.
    Completar la discretización del estado y la función de acción.
    """
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.95,
                    epsilon=0.001, epsilon_decay=0.995, min_epsilon=0.01, load_q_table_path="src/flappy_birds_q_table_final.pkl"):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.game = game
        self.game_height = self.game.height
        self.game_width = self.game.width
        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        # Parámetros de discretización
        self.num_bins = {
            'relative_bird_safespace_y': 10,
            'next_pipe_dist_to_player' : 4,
            'vertical_velocity' : 10
        }

    def discretize_state(self, state):
        """
        Discretiza el estado continuo en un estado discreto (tupla).
        """
        player_center_y = state['player_y']
        pipe_safespace_center_y = state['next_pipe_top_y'] + ((state['next_pipe_bottom_y']-state['next_pipe_top_y'])//2)
        relative_bird_safespace_y = pipe_safespace_center_y - player_center_y
        next_pipe_dist_to_player = state['next_pipe_dist_to_player']
        vertical_velocity = state['player_vel']

        # Discretización en bins
        rel_y_bin = int(np.digitize(relative_bird_safespace_y, np.linspace(-150, 150, self.num_bins['relative_bird_safespace_y'])))
        dist_bin = int(np.digitize(next_pipe_dist_to_player, np.linspace(0, 300, self.num_bins['next_pipe_dist_to_player'])))
        vel_bin = int(np.digitize(vertical_velocity, np.linspace(-16, 10, self.num_bins['vertical_velocity'])))

        return (rel_y_bin, dist_bin, vel_bin)

    def act(self, state):
        """
        Elige una acción usando epsilon-greedy sobre la Q-table.
        COMPLETAR: Implementar la política epsilon-greedy.
        """
        # Sugerencia:
        # - Discretizar el estado
        # - Con probabilidad epsilon elegir acción aleatoria
        # - Si no, elegir acción con mayor Q-value

        discrete_state = self.discretize_state(state)
        if random.random() < self.epsilon:
            #self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            return random.choice(self.actions)
        else:
            q_values = self.q_table[discrete_state]
            #self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            return self.actions[np.argmax(q_values)]

        #raise NotImplementedError("Completar la función de selección de acción (act)")

    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de Q-learning.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
