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

### Datos utilizados:

relative_bird_safespace_y => La distancia entre el centro de las tuberias que hay que esquivar y el jugador

next_pipe_dist_to_player => La distancia entre el jugador y el proximo obstaculo

vertical_velocity => La velocidad vertical del jugador

### Discretizacion:

relative_bird_safespace_y

    Valores contemplados: 0 a Alto del juego

    Cantidad de bins: 20

next_pipe_dist_to_player

    Valores contemplados: 0 a ancho del juego

    Cantidad de bins: 10

vertical_velocity

    Valores contemplados: -16 a 10

    Cantidad de bins: 10

next_pipe_top_y_position

    Valores contemplados: 0 a Alto del juego

    Cantidad de bins: 20

next_pipe_bottom_y_position

    Valores contemplados: 0 a Alto del juego

    Cantidad de bins: 20

### Justificación

Dado que el objetivo del juego es evitar colisiones con los obstáculos (tuberías, suelo y techo), observamos que el agente debe priorizar cinco aspectos fundamentales:
- Minimizar la distancia vertical entre el jugador y el centro del espacio seguro entre las tuberías (`relative_bird_safespace_y`).
- Alcanzar esa altura antes de que la distancia horizontal al próximo obstáculo sea cero (`next_pipe_dist_to_player`).
- Estabilizar la velocidad vertical (`vertical_velocity`) alrededor de cero una vez alcanzada la altura deseada, para mantener el control.
- Ademas de conocer el centro de la zona seguro necesita saber con margen puede alejarse medio `next_pipe_top_y_position` y `next_pipe_bottom_y_position`

Mientras se cumplan estas cuatro condiciones, el jugador debería poder evitar colisiones y, por lo tanto, ganar puntos.

En un intento de mejorar el rendimiento del agente, incorporamos una nueva variable: la distancia vertical al espacio seguro de la tubería que viene luego de la actual (`next_relative_bird_safespace_y`). Sin embargo, esto provocó una disminución en la performance. Creemos que esto se debe a dos motivos principales:

1. Por un lado, la incorporación de esta variable incrementó significativamente el tamaño del espacio de estados, dificultando el aprendizaje estable del agente.
2. Por otro lado, la información que aporta esta variable no es útil en el momento en que se la introduce, ya que el obstáculo actual aún no fue superado. Esto parece inducir al agente a tomar decisiones prematuras o erráticas, al considerar elementos que aún no influyen directamente en la jugada actual.

En base a estas observaciones, decidimos mantener el modelo lo más simple posible, limitándonos a variables asociadas exclusivamente al obstáculo presente y al movimiento inmediato del jugador. Esto resultó en un comportamiento más consistente y efectivo del agente.

La discretización de nuestro agente tiene 3 variables con 30, 15 y 15 bins cada una respectivamente, pudiendo haber entonces una totalidad de `20 × 10 × 10 x 20 x 20 = 800000` estados en total.


### Optimizado (Extra)

Una vez finalizado el trabajo practico se decidio seguir jugando cambiando y agregando ya sea estados discretos como numeros de bins para cada uno de esos estados, ademas se implemento un onehot a los mismos antes de entrenar la red neuronal tratando de alcanzar el máximo puntaje posible.

La ultima version de esta optimizacion tuvo los siquientes parametros

relative_bird_safespace_y

    Valores contemplados: -150 a 150

    Cantidad de bins: 30

next_pipe_dist_to_player

    Valores contemplados: 0 a 300

    Cantidad de bins: 15

vertical_velocity

    Valores contemplados: -16 a 10

    Cantidad de bins: 15

Bajo los mismos el pajaro llego a tener valores de reward de mas de 10000 puntos.

En la carpeta "Optimizado" se encuentra los archivos modificados utilizados para lograr este score