# Apps de Visión por Computadora con Python

Dos aplicaciones demostrativas hechas con **Python** y **OpenCV** para conocer de forma práctica dos tecnologías de visión por computadora: la **realidad aumentada con marcadores ArUco** y la **detección y reconocimiento facial con Haar Cascade**. Las dos usan la cámara de la computadora y funcionan en tiempo real.

## Las apps

### 1. Laboratorio de Realidad Aumentada (`realidad_aumentada/`)
La cámara encuentra marcadores ArUco impresos en papel, calcula su posición en 3D y dibuja contenido virtual encima de ellos.

| Modo | Qué hace |
|---|---|
| Objetos 3D | Un cubo que gira, una pirámide y una casa, todos con caras de color, sobre el marcador. |
| Sistema solar | Cada marcador muestra la tarjeta de un planeta con datos curiosos. |
| Regla AR | Mide en centímetros la distancia real entre marcadores. |
| Perilla de control | Al girar el marcador se controla el volumen o el brillo del video. |

Más detalles en [realidad_aumentada/README.md](realidad_aumentada/README.md).

### 2. Laboratorio Haar Cascade (`reconocimiento_facial/`)
Detecta rostros, ojos y sonrisas, y usa esa información para tareas útiles y divertidas.

| Modo | Qué hace |
|---|---|
| Detector | Marca rostros, ojos y sonrisas, y cuenta cuántas personas hay. |
| Privacidad | Pixela o desenfoca las caras automáticamente. |
| Filtros | Lentes, gorro de fiesta, bigote u orejas de gato que siguen tu cara. |
| Selfie sonriente | Toma la foto sola cuando sonríes. |
| Reconocimiento | Aprende tu cara y te reconoce por tu nombre (con LBPH). |

Más detalles en [reconocimiento_facial/README.md](reconocimiento_facial/README.md).

## Requisitos

- Python 3.9 o más reciente
- Una webcam
- Librerías: `opencv-contrib-python` y `numpy`

## Instalación rápida

```bash
# Crear y activar un entorno virtual (solo la primera vez)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Instalar las librerías (las dos apps usan las mismas)
pip install -r realidad_aumentada/requirements.txt
```

> Instala **solo** `opencv-contrib-python`. Si también tienes `opencv-python`, los dos paquetes pueden chocar entre sí. Para arreglarlo:
> `pip uninstall -y opencv-python opencv-contrib-python` y luego `pip install opencv-contrib-python`.

## Cómo ejecutarlas

```bash
# Realidad aumentada
cd realidad_aumentada
python generar_marcadores.py    # crea los marcadores para imprimir (solo la primera vez)
python app.py

# Reconocimiento facial
cd reconocimiento_facial
python app.py
```

En las dos apps, la ventana se maneja con el teclado: los números cambian de modo, `H` muestra la ayuda, `S` guarda una captura y `Q` cierra la app.

## Estructura del proyecto

```
python-apps/
├── README.md                     # este archivo
├── realidad_aumentada/
│   ├── app.py                    # app principal de realidad aumentada
│   ├── generar_marcadores.py     # crea los marcadores y las hojas para imprimir
│   ├── calibrar_camara.py        # calibración opcional de la cámara
│   ├── requirements.txt
│   └── README.md
└── reconocimiento_facial/
    ├── app.py                    # app principal de Haar Cascade + LBPH
    ├── haarcascade_frontalface_alt.xml
    ├── requirements.txt
    └── README.md
```

## Conceptos que se aprenden

- **Visión por computadora**: cómo una computadora "entiende" lo que ve una cámara.
- **Pose 3D y proyección**: pasar de puntos en el espacio a pixeles en la pantalla (`solvePnP`, `projectPoints`).
- **Homografía**: deformar una imagen plana para que encaje en perspectiva.
- **Clasificadores en cascada**: detectar objetos rápido usando patrones de luz y sombra.
- **Detectar y reconocer no son lo mismo**: encontrar *una* cara no es lo mismo que saber *de quién* es.
- **Privacidad**: el uso responsable de las imágenes de las personas.
