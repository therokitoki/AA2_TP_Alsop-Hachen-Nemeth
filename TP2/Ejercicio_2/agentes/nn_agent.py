from agentes.base import Agent
import numpy as np
import tensorflow as tf
import random

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='D:\\Git\\AA2_TP_Alsop-Hachen-Nemeth\\TP2\\Ejercicio_2\\flappy_q_nn_model.h5'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.game = game
        self.game_height = self.game.height # PLE pasa el objeto game directamente
        self.game_width = self.game.width

        # Parámetros de discretización
        self.num_bins = {
            'relative_bird_safespace_y': 10,
            #'relative_bird_nn_safespace_y': 10,
            'next_pipe_dist_to_player' : 5,
            #'next_next_pipe_dist_to_player' : 5
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

        # # Centro del espacio entre tuberías
        # nn_pipe_safespace_center_y = state['next_next_pipe_top_y'] + ((state['next_next_pipe_bottom_y']-state['next_next_pipe_top_y'])//2)

        # # Posición relativa del pajarito al centro del espacio entre tuberías
        # relative_bird_nn_safespace_y = nn_pipe_safespace_center_y - player_center_y
        # scaled_relative_bird_nn_safespace_y = (relative_bird_nn_safespace_y + self.game_height / 2) / self.game_height
        # scaled_relative_bird_nn_safespace_y_bin = int(np.clip(scaled_relative_bird_nn_safespace_y * self.num_bins['relative_bird_nn_safespace_y'], 0, self.num_bins['relative_bird_nn_safespace_y'] - 1))

        # 2. Signo de la velocidad del jugador
        if state['player_vel'] < 0:
            player_velocity_sign_bin = 2 # Moviéndose arriba
        elif state['player_vel'] > 0:
            player_velocity_sign_bin = 0 # Moviéndose abajo
        else:
            player_velocity_sign_bin = 1 # Quieto o casi quieto

        n_pipe_dist_to_player_normalized = state['next_pipe_dist_to_player'] / self.game_width
        n_pipe_dist_to_player = int(np.clip(n_pipe_dist_to_player_normalized * self.num_bins['next_pipe_dist_to_player'], 0, self.num_bins['next_pipe_dist_to_player'] - 1))

        if state['next_pipe_dist_to_player'] < 60:
            pajaro_en_el_nido = 1
        else:
            pajaro_en_el_nido = 0

        # nn_pipe_dist_to_player_normalized = state['next_next_pipe_dist_to_player'] / self.game_width
        # nn_pipe_dist_to_player = int(np.clip(nn_pipe_dist_to_player_normalized * self.num_bins['next_next_pipe_dist_to_player'], 0, self.num_bins['next_next_pipe_dist_to_player'] - 1))
        # Ejemplo:
        # return (player_y_bin, player_vel_bin, ...)
        return (
            scaled_relative_bird_safespace_y_bin,
            #scaled_relative_bird_nn_safespace_y_bin,
            player_velocity_sign_bin,
            n_pipe_dist_to_player,
            #pajaro_en_el_nido,
            #nn_pipe_dist_to_player
        )


    def act(self, state):
        """
        COMPLETAR: Implementar la función de acción.
        Debe transformar el estado al formato de entrada de la red y devolver la acción con mayor Q.
        """
        discrete_state = self.discretize_state(state)

        discrete_state=np.array(discrete_state)
        q_values = self.model.predict(np.expand_dims(discrete_state, axis=0))

        return self.actions[np.argmax(q_values)]

