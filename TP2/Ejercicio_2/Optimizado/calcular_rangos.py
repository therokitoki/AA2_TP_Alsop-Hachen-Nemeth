import numpy as np

# --- Inicializar acumuladores ---
state_keys = ['relative_bird_safespace_y', 'next_pipe_dist_to_player', 'vertical_velocity']
min_vals = {key: float('inf') for key in state_keys}
max_vals = {key: float('-inf') for key in state_keys}

def calcular_rangos(state_dict):
    """
    Actualiza los valores mínimos y máximos de las variables derivadas del estado.
    """
    global min_vals, max_vals

    player_center_y = state_dict['player_y']
    pipe_safespace_center_y = state_dict['next_pipe_top_y'] + ((state_dict['next_pipe_bottom_y'] - state_dict['next_pipe_top_y']) // 2)
    relative_bird_safespace_y = pipe_safespace_center_y - player_center_y
    next_pipe_dist_to_player = state_dict['next_pipe_dist_to_player']
    vertical_velocity = state_dict['player_vel']

    valores = {
        'relative_bird_safespace_y': relative_bird_safespace_y,
        'next_pipe_dist_to_player': next_pipe_dist_to_player,
        'vertical_velocity': vertical_velocity
    }

    for key in state_keys:
        min_vals[key] = min(min_vals[key], valores[key])
        max_vals[key] = max(max_vals[key], valores[key])

def guardar_rangos_en_txt(output_path='src/state_ranges.txt'):
    with open(output_path, 'w') as f:
        for key in state_keys:
            f.write(f"{key}: min = {min_vals[key]:.4f}, max = {max_vals[key]:.4f}\n")
    print(f"\n Rangos guardados en '{output_path}'")
