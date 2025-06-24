from agentes.base import Agent
import numpy as np
import tensorflow as tf
import random

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='src/flappy_q_nn_model.h5'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.game = game
        self.game_height = self.game.height # PLE pasa el objeto game directamente
        self.game_width = self.game.width

        self.num_bins = {
            'relative_bird_safespace_y': 30,
            'next_pipe_dist_to_player' : 15,
            'vertical_velocity' : 15
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
        COMPLETAR: Implementar la función de acción.
        Debe transformar el estado al formato de entrada de la red y devolver la acción con mayor Q.
        """
        discrete_state = self.discretize_state(state)
        discrete_state=np.array(discrete_state)
        q_values = self.model.predict(np.expand_dims(discrete_state, axis=0), verbose=0)
        return self.actions[np.argmax(q_values)]

