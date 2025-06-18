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
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, load_q_table_path="flappy_birds_q_table.pkl"):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.game = game


        self.game_height = self.game.height # PLE pasa el objeto game directamente
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
            'relative_bird_nn_safespace_y': 10, 
            'next_pipe_dist_to_player' : 5,
            'next_next_pipe_dist_to_player' : 5
        }

    def discretize_state(self, state):
        """
        Discretiza el estado continuo en un estado discreto (tupla).
        COMPLETAR: Implementar la discretización adecuada para el entorno.
        """

        """ 
        player_y": self.player.pos_y,
            "player_vel": self.player.vel,
            
            "next_pipe_dist_to_player": next_pipe.x - self.player.pos_x,
            "next_pipe_top_y": next_pipe.gap_start,
            "next_pipe_bottom_y": next_pipe.gap_start+self.pipe_gap, 
            
            "next_next_pipe_dist_to_player": next_next_pipe.x - self.player.pos_x,
            "next_next_pipe_top_y": next_next_pipe.gap_start,
            "next_next_pipe_bottom_y": next_next_pipe.gap_start+self.pipe_gap 

        """

        # Centro del pajarito
        player_center_y = state['player_y']
        # Centro del espacio entre tuberías
        pipe_safespace_center_y = state['next_pipe_top_y'] + ((state['next_pipe_bottom_y']-state['next_pipe_top_y'])//2)

        # Posición relativa del pajarito al centro del espacio entre tuberías
        relative_bird_safespace_y = pipe_safespace_center_y - player_center_y
        scaled_relative_bird_safespace_y = (relative_bird_safespace_y + self.game_height / 2) / self.game_height
        scaled_relative_bird_safespace_y_bin = int(np.clip(scaled_relative_bird_safespace_y * self.num_bins['relative_bird_safespace_y'], 0, self.num_bins['relative_bird_safespace_y'] - 1))

        # Centro del espacio entre tuberías
        nn_pipe_safespace_center_y = state['next_next_pipe_top_y'] + ((state['next_next_pipe_bottom_y']-state['next_next_pipe_top_y'])//2)

        # Posición relativa del pajarito al centro del espacio entre tuberías
        relative_bird_nn_safespace_y = nn_pipe_safespace_center_y - player_center_y
        scaled_relative_bird_nn_safespace_y = (relative_bird_nn_safespace_y + self.game_height / 2) / self.game_height
        scaled_relative_bird_nn_safespace_y_bin = int(np.clip(scaled_relative_bird_nn_safespace_y * self.num_bins['relative_bird_nn_safespace_y'], 0, self.num_bins['relative_bird_nn_safespace_y'] - 1))

        # 2. Signo de la velocidad del jugador
        if state['player_vel'] < 0:
            player_velocity_sign_bin = 2 # Moviéndose arriba
        elif state['player_vel'] > 0:
            player_velocity_sign_bin = 0 # Moviéndose abajo
        else:
            player_velocity_sign_bin = 1 # Quieto o casi quieto

        n_pipe_dist_to_player_normalized = state['next_pipe_dist_to_player'] / self.game_width
        n_pipe_dist_to_player = int(np.clip(n_pipe_dist_to_player_normalized * self.num_bins['next_pipe_dist_to_player'], 0, self.num_bins['next_pipe_dist_to_player'] - 1))  

        nn_pipe_dist_to_player_normalized = state['next_next_pipe_dist_to_player'] / self.game_width
        nn_pipe_dist_to_player = int(np.clip(nn_pipe_dist_to_player_normalized * self.num_bins['next_next_pipe_dist_to_player'], 0, self.num_bins['next_next_pipe_dist_to_player'] - 1))   
        # Ejemplo:
        # return (player_y_bin, player_vel_bin, ...)
        return (
            scaled_relative_bird_safespace_y_bin,
            scaled_relative_bird_nn_safespace_y_bin,
            player_velocity_sign_bin,
            n_pipe_dist_to_player,
            nn_pipe_dist_to_player
        )

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
            return random.choice(self.actions)
        else:
            q_values = self.q_table[discrete_state]
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
