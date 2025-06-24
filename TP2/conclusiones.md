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

    Valores contemplados: -150 a 150 

    Cantidad de bins: 30

next_pipe_dist_to_player

    Valores contemplados: 0 a 300

    Cantidad de bins: 15

vertical_velocity

    Valores contemplados: -16 a 10

    Cantidad de bins: 15

### Justificación
Dado el objetivo del juego (no colisionar con ningun obstaculo: tuberias, piso y parte superior de la pantalla), observamos que es necesario minimizar la distancia en el eje vertical entre el centro de las tuberias que hay que esquivar y el jugador (relative_bird_safespace_y), pero a esto hay que hacerlo antes de hacer contacto con las tuberias, es decir, antes de que la distancia entre el jugador y el proximo obstaculo sea 0 (next_pipe_dist_to_player) y manteniendo una velocidad vertical lo mas cerca posible de 0 (vertical_velocity) una vez que se logro conseguir la altura deseada. Mientras se cumplan estas tres consignas el jugador va a conseguir puntos.

Intentamos agregar mas informacion que intuimos que iba a mejorar la performance del agente, como por ejemplo la distancia en el eje vertical del proximo obstaculo, antes de esquivar el actual (next_relative_bird_safespace_y) pero esto generaba peores resultados.

Creemos que esto se debe a que al agregar esta variable se vuelve mucho más grande el espacio de estados posibles, haciendo más difícil que el agente pueda aprender de forma consistente. Además, esta información no era realmente útil en el momento en que se la tenía en cuenta, porque todavía no se había superado el primer obstáculo, y lo único que lograba era confundir al agente. También notamos que se empezaba a comportar de forma más errática, probablemente porque tenía en cuenta cosas que todavía no influían directamente en la jugada actual. Por eso, decidimos mantener el modelo más simple y usar solo variables relacionadas al obstáculo actual y al movimiento inmediato del jugador.
