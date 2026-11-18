"""
Calibracion de camara (opcional, mejora la exactitud de la Regla AR).

1. Imprime marcadores/tablero_ajedrez.png (lo crea generar_marcadores.py).
2. Ejecuta: python calibrar_camara.py --cuadro 2.2   (lado de un cuadro en cm)
3. Mueve el tablero frente a la camara. Cuando se vean las lineas de colores
   presiona ESPACIO para guardar esa vista. Toma unas 15 desde angulos distintos.
4. Presiona C para calcular. Se crea calibracion.npz y app.py lo usara solo.
"""

import argparse
import os
import sys

import cv2
import numpy as np

CARPETA = os.path.dirname(os.path.abspath(__file__))
PATRON = (9, 6)  # esquinas internas del tablero (columnas, filas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camara", type=int, default=0)
    parser.add_argument("--cuadro", type=float, default=2.2, help="lado de un cuadro en cm")
    parser.add_argument("--ancho", type=int, default=1280)
    parser.add_argument("--alto", type=int, default=720)
    args = parser.parse_args()

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camara, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.alto)

    # Coordenadas reales de las esquinas del tablero (en metros, sobre el plano Z=0)
    modelo = np.zeros((PATRON[0] * PATRON[1], 3), np.float32)
    modelo[:, :2] = np.mgrid[0:PATRON[0], 0:PATRON[1]].T.reshape(-1, 2) * args.cuadro / 100

    puntos_mundo, puntos_imagen = [], []
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    tam = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer la camara")
            return 1
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tam = gris.shape[::-1]
        encontrado, esquinas = cv2.findChessboardCorners(gris, PATRON, cv2.CALIB_CB_FAST_CHECK)
        vista = frame.copy()
        if encontrado:
            cv2.drawChessboardCorners(vista, PATRON, esquinas, encontrado)
        cv2.putText(vista, f"Vistas: {len(puntos_mundo)}  [ESPACIO] guardar  [C] calibrar  [Q] salir",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Calibracion", vista)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(" ") and encontrado:
            esquinas = cv2.cornerSubPix(gris, esquinas, (11, 11), (-1, -1), criterio)
            puntos_mundo.append(modelo)
            puntos_imagen.append(esquinas)
            print(f"Vista {len(puntos_mundo)} guardada")
        elif tecla == ord("c"):
            if len(puntos_mundo) < 8:
                print("Necesitas al menos 8 vistas")
                continue
            error, matriz, distorsion, _, _ = cv2.calibrateCamera(
                puntos_mundo, puntos_imagen, tam, None, None)
            np.savez(os.path.join(CARPETA, "calibracion.npz"), matriz=matriz,
                     distorsion=distorsion, ancho=tam[0], alto=tam[1])
            print(f"Calibracion guardada. Error de reproyeccion: {error:.3f} px (menor a 1 es bueno)")
            print("Matriz de camara:\n", np.round(matriz, 1))
            break
        elif tecla in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
