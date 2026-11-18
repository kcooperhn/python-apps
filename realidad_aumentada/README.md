# Laboratorio de Realidad Aumentada con marcadores ArUco

Esta app usa la cámara de la computadora para encontrar **marcadores ArUco** (cuadros blanco y negro, parecidos a un código QR) y dibuja sobre ellos objetos virtuales que se mueven junto con el papel. Eso es la **realidad aumentada (AR)**: mezclar el mundo real con contenido digital.

## ¿Qué puedes hacer?

| Tecla | Modo | Qué hace |
|---|---|---|
| `1` | **Objetos 3D** | ID 0: un cubo que gira. ID 1: una pirámide. ID 2: una casa. Cualquier otro ID muestra los ejes X, Y, Z. Inclina el papel y verás el objeto desde otro ángulo. |
| `2` | **Sistema solar** | ID 0 es el Sol y los ID 1 a 8 son los planetas. Cada marcador muestra una tarjeta con datos curiosos. Acomoda los marcadores en la mesa y arma tu propio sistema solar. |
| `3` | **Regla AR** | Con 2 o más marcadores, mide la distancia **real** entre ellos en centímetros. |
| `4` | **Perilla de control** | Gira el marcador como si fuera la perilla de un estéreo. ID 0 controla el "volumen" e ID 1 controla el **brillo del video**. |

Otras teclas: `H` muestra u oculta la ayuda, `S` guarda una captura en `capturas/` y `Q` (o `Esc`) cierra la app.

## Instalación y uso

```bash
# 1. Instalar las librerías (en tu entorno virtual)
pip install -r requirements.txt

# 2. Generar los marcadores para imprimir (se crea la carpeta marcadores/)
python generar_marcadores.py

# 3. Ejecutar la app
python app.py
```

- Imprime `marcadores/hoja_1.png` y `hoja_2.png` **al 100 %** (sin "ajustar a página"). También puedes mostrar `marcador_<id>.png` en la pantalla del celular.
- Mide con una regla el lado del cuadro **negro**. Si no mide 5 cm, indícale a la app la medida real: `python app.py --tam 4.6`
- Si tienes varias cámaras: `python app.py --camara 1`

> **Importante:** instala **solo** `opencv-contrib-python`. Si también tienes `opencv-python`, pueden chocar entre sí. Para arreglarlo:
> `pip uninstall -y opencv-python opencv-contrib-python` y luego `pip install opencv-contrib-python`.

## ¿Cómo funciona? (la idea en 4 pasos)

1. **Detectar**: OpenCV busca cuadrados negros en la imagen y lee el código de bits que tienen dentro. Así obtiene el **ID** del marcador y la posición de sus **4 esquinas** en pixeles.
2. **Calcular la pose**: como sabemos cuánto mide el marcador en la vida real (5 cm), la función `solvePnP` calcula dónde está en el espacio 3D (`tvec`, en metros) y hacia dónde apunta (`rvec`, su rotación).
3. **Proyectar**: definimos el objeto virtual en 3D (por ejemplo las 8 esquinas de un cubo) y con `projectPoints` calculamos en qué pixeles de la imagen deben aparecer.
4. **Dibujar**: unimos esos pixeles con líneas y polígonos. Como esto se repite unas 30 veces por segundo, el objeto parece pegado al papel.

## Partes importantes del código (`app.py`)

### `Camara`: cómo "ve" la cámara
```python
focal = 0.9 * ancho
self.matriz = np.array([[focal, 0, ancho / 2],
                        [0, focal, alto / 2],
                        [0, 0, 1]])
```
La **matriz de cámara** contiene la distancia focal y el centro de la imagen. Sin estos datos no se puede convertir de 3D a 2D. Aquí usamos un valor aproximado. Si ejecutas `calibrar_camara.py`, se guarda la matriz real en `calibracion.npz` y la app la usa automáticamente (las medidas salen más exactas).

### `DetectorAruco.detectar`: encontrar marcadores y su pose
```python
esquinas, ids, _ = self.detector.detectMarkers(gris)
ok, rvec, tvec = cv2.solvePnP(self.puntos_modelo, esq, camara.matriz,
                              camara.distorsion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
```
- `detectMarkers` devuelve las esquinas y el ID de cada marcador encontrado.
- `puntos_modelo` son las esquinas del marcador **en el mundo real** (en metros, con el centro en 0,0).
- `solvePnP` compara las esquinas reales con las de la imagen y deduce la posición (`tvec`) y la rotación (`rvec`).
- `_suavizar` promedia la pose con la del frame anterior para que los objetos no tiemblen.

### `proyectar`: de 3D a pixeles
```python
pts, _ = cv2.projectPoints(puntos_3d, marcador.rvec, marcador.tvec,
                           camara.matriz, camara.distorsion)
```
Recibe puntos en coordenadas del marcador (X a la derecha, Y hacia arriba y Z saliendo del papel) y devuelve dónde se ven en la imagen. Todos los modos la usan.

### Modo 1, `dibujar_solido`: objetos 3D con caras de color
- `modelo_cubo`, `modelo_piramide` y `modelo_casa` definen **vértices** (puntos 3D) y **caras** (qué vértices forman cada cara y de qué color es).
- Se usa el **algoritmo del pintor**: primero se dibujan las caras más lejanas y al final las más cercanas, para que las de enfrente tapen a las de atrás.
- La **normal** de cada cara (hacia dónde mira) se usa para oscurecerla o aclararla y que el objeto se vea con volumen.
- `rotacion_z` hace que el cubo gire con el tiempo (animación).

### Modo 2, `pegar_imagen`: imágenes sobre el marcador
```python
matriz = cv2.getPerspectiveTransform(origen, destino)
deformada = cv2.warpPerspective(imagen, matriz, (ancho, alto))
```
Una **homografía** deforma una imagen plana para que encaje en 4 puntos cualquiera, igual que cuando ves una hoja de papel inclinada. `crear_tarjeta` dibuja cada planeta. Si guardas una imagen como `imagenes/5.png`, el marcador ID 5 mostrará tu imagen.

### Modo 3, `ModoRegla`: medir distancias
```python
distancia = np.linalg.norm(a.tvec - b.tvec) * 100
```
`tvec` es la posición 3D de cada marcador en metros. La distancia entre dos puntos 3D es la longitud del vector que los une (teorema de Pitágoras en 3D). Se multiplica por 100 para pasar a centímetros.

### Modo 4, `ModoPerilla`: el marcador como control
```python
angulo = math.degrees(math.atan2(sup_der[1] - sup_izq[1], sup_der[0] - sup_izq[0]))
objetivo = (np.clip(angulo, -135, 135) + 135) / 270 * 100
```
Se mide la inclinación del borde superior del marcador. Ese ángulo (de -135° a 135°) se convierte en un valor de 0 a 100 %. Con ID 1, `convertScaleAbs` multiplica cada pixel por un factor y así cambia el brillo de todo el video.

### `main`: el ciclo principal
Leer frame → detectar marcadores → dibujar el modo activo → dibujar la interfaz (`dibujar_hud`) → mostrar → leer el teclado. Esto se repite hasta que presionas `Q`.

## Otros archivos

| Archivo | Para qué sirve |
|---|---|
| `generar_marcadores.py` | Crea los marcadores individuales, 2 hojas para imprimir (IDs 0-11) y un tablero de ajedrez para calibrar. |
| `calibrar_camara.py` | (Opcional) Calibra la cámara con el tablero de ajedrez: muéstralo desde unos 15 ángulos distintos (tecla `ESPACIO`) y presiona `C`. |
| `imagenes/` | (Opcional) Pon aquí `<id>.png` para usar tus propias imágenes en el modo 2. |

## Consejos si no detecta los marcadores

- Que haya buena luz y sin reflejos (el papel mate funciona mejor que una pantalla brillante).
- Deja el **margen blanco** alrededor del cuadro negro y no tapes ninguna esquina con los dedos.
- Si el marcador se ve muy pequeño en la pantalla, acércalo a la cámara.

## Retos para los alumnos

1. Agrega un nuevo objeto 3D para el ID 3 (por ejemplo una mesa o un árbol) creando una función `modelo_...`.
2. Haz que la pirámide también gire, pero en sentido contrario al cubo.
3. Cambia los datos de `PLANETAS` por datos de otro tema: animales, países o partes de la célula.
4. En el modo Regla, muestra una alerta en rojo si dos marcadores están a menos de 10 cm.
5. Calibra la cámara y compara las medidas de la Regla AR antes y después.
