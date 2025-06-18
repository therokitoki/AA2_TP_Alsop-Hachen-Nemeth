from agentes.base import Agent
import pygame
import numpy as np

class ManualAgent(Agent):
    """Agente que toma acciones manualmente: salta al presionar la barra espaciadora."""
    def __init__(self, actions, game=None):
        super().__init__(actions, game)
        self.jump_action = actions[0]
        self.noop_action = actions[1]
        self._space_was_pressed = False
        self.game = game


        self.game_height = self.game.height # PLE pasa el objeto game directamente
        self.game_width = self.game.width

        self.num_bins = {
            'relative_bird_safespace_y': 10,   
            'next_pipe_dist_to_player' : 5
        }

    def act(self, state):
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        jump = False
        player_center_y = state['player_y']
        pipe_safespace_center_y = state['next_pipe_top_y'] + ((state['next_pipe_bottom_y']-state['next_pipe_top_y'])//2)
        relative_bird_safespace_y = pipe_safespace_center_y - player_center_y
        scaled_relative_bird_safespace_y = (relative_bird_safespace_y + self.game_height / 2) / self.game_height
        scaled_relative_bird_safespace_y_bin = int(np.clip(scaled_relative_bird_safespace_y * self.num_bins['relative_bird_safespace_y'], 0, self.num_bins['relative_bird_safespace_y'] - 1))
        
        n_pipe_dist_to_player_normalized = state['next_pipe_dist_to_player'] / self.game_width
        n_pipe_dist_to_player = int(np.clip(n_pipe_dist_to_player_normalized * self.num_bins['next_pipe_dist_to_player'], 0, self.num_bins['next_pipe_dist_to_player'] - 1))    
        
        print(n_pipe_dist_to_player)
    
        if not self._space_was_pressed and keys[pygame.K_SPACE]:
            jump = True
        self._space_was_pressed = keys[pygame.K_SPACE]
        if jump:
            return self.jump_action
        return self.noop_action
