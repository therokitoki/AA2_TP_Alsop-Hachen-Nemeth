# Trabajo Practico N°2 AAII

Alumnos: Agustin Alsop, Rocio Hachen, Ulises Nemeth

## Ejercicio 1

### Descripción:
Entrenar agentes que puedan jugar Flappy Bird usando diferentes enfoques con PyGame Learning Environment.

### Objetivo:
El objetivo de este ejercicio es entrenar agentes para resolver videojuegos sencillos usando Q-Learning y la librería PLE. En primer lugar, usar Q-learning para entrenar a un agente para jugar Flappy Bird. Luego, entrenar a otro agente usando Deep Q-learning y la Q-table del agente provisto.

### Entrega:
Entrega: La entrega debe incluir el código fuente de la solución, completando los archivos y scripts provistos en el template del proyecto.
1- Agente Q-Learning (Corresponde al Ejercicio A del README.md):
    a- Implementación del Agente.
    b- Entrenamiento del Agente.
    c- Prueba del Agente Entrenado
Agente Basado en Red Neuronal (Corresponde al Ejercicio B del README.md):
Entrenamiento de la Red Neuronal.
Implementación del Agente Neuronal.
Prueba del Agente Neuronal
Deben poder utilizarse los distintos tipos de agentes usando el parámetro --agent al ejecutar test_agent.py.
Dentro del repositorio se debe incluir un archivo conclusiones.md (usando Markdown) con:
Descripción de la ingeniería de características sobre el estado del juego (discretización).
Análisis y comparación de los resultados obtenidos para los diferentes agentes.

### Variables utilizadas:

relative_bird_safespace_y => La distancia entre el centro de las tuberias que hay que esquivar y el jugador

next_pipe_dist_to_player => La distancia entre el jugador y el proximo obstaculo

vertical_velocity => La velocidad vertical del jugador

next_pipe_top_y_position =>

next_pipe_bottom_y_position =>

### Discretizacion:

`rel_y_bin`
- Dato utilizado: relative_bird_safespace_y
- Valores contemplados: 0 a 512
- Cantidad de bins: 20

`dist_bin`
- Dato utilizado: next_pipe_dist_to_player
- Valores contemplados: 0 a 288
- Cantidad de bins: 10

`vel_bin`
- Dato utilizado: vertical_velocity
- Valores contemplados: -16 a 10
- Cantidad de bins: 10

`top_y_bin`
- Dato utilizado: next_pipe_top_y_position
- Valores contemplados: 0 a 512
- Cantidad de bins: 20

`bottom_y_bin`
- Dato utilizado: next_pipe_bottom_y_position
- Valores contemplados: 0 a 512
- Cantidad de bins: 20

### Justificación

Dado que el objetivo del juego es evitar colisiones con los obstáculos (tuberías, suelo y techo), observamos que el agente debe priorizar cuatro aspectos fundamentales:
- Minimizar la distancia vertical entre el jugador y el centro del espacio seguro entre las tuberías (`relative_bird_safespace_y`).
- Alcanzar esa altura antes de que la distancia horizontal a la próxima tubería sea cero (`next_pipe_dist_to_player`).
- Estabilizar la velocidad vertical (`vertical_velocity`) alrededor de cero una vez alcanzada la altura deseada, para mantener el control.
- Conocer la posición vertical exacta del borde superior (`next_pipe_top_y_position`) y del borde inferior (`next_pipe_bottom_y_position`).

Mientras se cumplan estas cuatro condiciones, el jugador debería poder evitar colisiones y, por lo tanto, ganar puntos.

En un intento de mejorar el rendimiento del agente, incorporamos una nueva variable: la distancia vertical al espacio seguro de la tubería que sigue a la actual (`next_relative_bird_safespace_y`). Sin embargo, esto provocó una disminución en el desempeño del agente. Creemos que esto se debe a dos motivos principales:

1. La incorporación de esta variable incrementaba aun más el tamaño del espacio de estados, lo que podía dificultar el aprendizaje del agente.
2. La información que aporta esta variable no es útil en el momento en que se la introduce, ya que el obstáculo actual aún no fue superado. Esto parece inducir al agente a tomar decisiones prematuras o erráticas.

En base a estas observaciones, decidimos mantener el modelo lo más simple posible, limitándonos a variables asociadas exclusivamente a la tubería próxima y al movimiento inmediato del jugador. Esta versión demostró ser mucho más consistente y efectiva. De hecho, en la sección de "Optimización "de este documento se muestra cómo, usando solo tres variables y un espacio de estados mucho más reducido, se lograron resultados sobresalientes. 

No obstante, esa versión más simple requería una codificación one-hot para poder entrenar correctamente la red neuronal, lo cual estaba fuera del alcance planteado inicialmente. Por ello, en esta sección decidimos adoptar la siguiente mejor versión: un modelo con cinco variables discretizadas, que si bien amplía el espacio de estados, permite un gran desempeño tanto en el agente Q como en el agente con red neuronal.

La discretización actual considera las siguientes variables y particiones:

- `rel_y_bin`: 20 bins
- `dist_bin`: 10 bins
- `vel_bin`: 10 bins
- `top_y_bin`: 20 bins
- `bottom_y_bin`: 20 bins

Lo que da un total de:
`20 × 10 × 10 × 20 × 20 = 800,000` estados posibles.

### Optimización (Extra)

Como se mencionó anteriormente, una vez finalizado el trabajo práctico, se decidió continuar explorando variantes del modelo, modificando tanto las variables como la cantidad de bins de discretización. Además, se implementó una codificación one-hot para los estados, lo que permitió entrenar la red neuronal de forma más precisa.

La ultima version de esta optimizacion incluyó las siguientes variables discretizadas:

`rel_y_bin`
- Dato utilizado: relative_bird_safespace_y
- Valores contemplados: -150 a 150
- Cantidad de bins: 30

`dist_bin`
- Dato utilizado: next_pipe_dist_to_player
- Valores contemplados: 0 a 300
- Cantidad de bins: 15

`vel_bin`
- Dato utilizado: vertical_velocity
- Valores contemplados: -16 a 10
- Cantidad de bins: 15

Así se cuenta con `30 × 15 × 15 = 6,750` estados posibles. Con esta discretización, el agente fue capaz de obtener recompensas acumuladas superiores a 10,000 puntos, lo cual representa un desempeño excepcional.
Todos los archivos modificados y utilizados para alcanzar estos resultados se encuentran en la carpeta "Optimizado".