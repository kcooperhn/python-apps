"""
Genera los marcadores ArUco listos para imprimir.

Crea la carpeta marcadores/ con:
  - marcador_<id>.png      un marcador por archivo (para mostrar en el celular)
  - hoja_1.png, hoja_2.png  hojas tamano carta/A4 con 6 marcadores de 5 cm cada una
  - tablero_ajedrez.png     tablero para calibrar la camara (calibrar_camara.py)

Al imprimir, mide con una regla el lado del cuadro NEGRO del marcador.
Si no mide 5 cm, ejecuta la app con:  python app.py --tam <medida_en_cm>
"""

import os

import cv2
import numpy as np

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marcadores")
DICCIONARIO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DPI = 300                      # puntos por pulgada de la hoja
CM = DPI / 2.54                # pixeles por centimetro
FUENTE = cv2.FONT_HERSHEY_SIMPLEX

# Que muestra cada ID en los modos 1, 2 y 4 de la app
USOS = {
    0: "Cubo | Sol | Volumen",
    1: "Piramide | Mercurio | Brillo",
    2: "Casa | Venus",
    3: "Ejes | Tierra",
    4: "Ejes | Marte",
    5: "Ejes | Jupiter",
    6: "Ejes | Saturno",
    7: "Ejes | Urano",
    8: "Ejes | Neptuno",
    9: "Ejes | Personalizable",
    10: "Ejes | Personalizable",
    11: "Ejes | Personalizable",
}


def generar_individuales():
    for id_ in USOS:
        marcador = cv2.aruco.generateImageMarker(DICCIONARIO, id_, 400)
        # Margen blanco: el detector necesita contraste alrededor del borde negro.
        marcador = cv2.copyMakeBorder(marcador, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(os.path.join(CARPETA, f"marcador_{id_}.png"), marcador)


def generar_hoja(ids, nombre, lado_cm=5.0):
    ancho, alto = int(21.0 * CM), int(27.9 * CM)  # cabe en carta y en A4
    hoja = np.full((alto, ancho), 255, np.uint8)
    lado = int(lado_cm * CM)

    titulo = f"Marcadores ArUco 6x6 - lado del cuadro negro = {lado_cm:g} cm"
    cv2.putText(hoja, titulo, (int(1.5 * CM), int(1.5 * CM)), FUENTE, 1.6, 0, 3, cv2.LINE_AA)
    cv2.putText(hoja, "Imprime al 100% (sin 'ajustar a pagina') y verifica con una regla",
                (int(1.5 * CM), int(2.3 * CM)), FUENTE, 1.1, 0, 2, cv2.LINE_AA)

    for k, id_ in enumerate(ids):
        fila, col = divmod(k, 2)
        x = int((2.5 + col * 9.0) * CM)
        y = int((3.5 + fila * 8.0) * CM)
        hoja[y:y + lado, x:x + lado] = cv2.aruco.generateImageMarker(DICCIONARIO, id_, lado)
        cv2.putText(hoja, f"ID {id_}", (x, y + lado + int(0.9 * CM)), FUENTE, 1.8, 0, 4, cv2.LINE_AA)
        cv2.putText(hoja, USOS[id_], (x, y + lado + int(1.6 * CM)), FUENTE, 1.0, 80, 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(CARPETA, nombre), hoja)


def generar_tablero(columnas=10, filas=7, cuadro_cm=2.2):
    """Tablero de ajedrez de 10x7 cuadros = 9x6 esquinas internas (hoja horizontal)."""
    c = int(cuadro_cm * CM)
    margen = int(1.2 * CM)
    tablero = np.full((filas * c + 2 * margen, columnas * c + 2 * margen), 255, np.uint8)
    for f in range(filas):
        for k in range(columnas):
            if (f + k) % 2 == 0:
                y, x = margen + f * c, margen + k * c
                tablero[y:y + c, x:x + c] = 0
    cv2.putText(tablero, f"Tablero 9x6 esquinas - cuadro = {cuadro_cm:g} cm",
                (margen, margen - 25), FUENTE, 1.2, 0, 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(CARPETA, "tablero_ajedrez.png"), tablero)


if __name__ == "__main__":
    os.makedirs(CARPETA, exist_ok=True)
    generar_individuales()
    generar_hoja([0, 1, 2, 3, 4, 5], "hoja_1.png")
    generar_hoja([6, 7, 8, 9, 10, 11], "hoja_2.png")
    generar_tablero()
    print(f"Listo. Archivos en: {CARPETA}")
    print("Imprime hoja_1.png y hoja_2.png al 100% de su tamano.")
