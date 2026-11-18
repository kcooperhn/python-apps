"""
Laboratorio de Realidad Aumentada con marcadores ArUco
======================================================

La camara busca marcadores ArUco (cuadros blanco y negro con un codigo).
Con las 4 esquinas de cada marcador calculamos su POSE: donde esta y hacia
donde apunta en el espacio 3D. Con esa pose dibujamos objetos virtuales
"pegados" al marcador.

Modos (teclas 1-4):
  1. Objetos 3D     -> cubo giratorio, piramide y casa sobre el marcador
  2. Sistema solar  -> cada marcador muestra la tarjeta de un planeta
  3. Regla AR       -> mide la distancia entre marcadores en centimetros
  4. Perilla        -> gira el marcador como si fuera una perilla de control

Antes de empezar ejecuta:  python generar_marcadores.py
"""

import argparse
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

CARPETA = os.path.dirname(os.path.abspath(__file__))
DICCIONARIO = cv2.aruco.DICT_6X6_250
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
# Nota: las fuentes de OpenCV no tienen acentos ni el simbolo de grados,
# por eso los textos en pantalla se escriben sin acentos.


# ---------------------------------------------------------------------------
# Utilidades de dibujo
# ---------------------------------------------------------------------------
def escribir(img, mensaje, origen, escala=0.6, color=(255, 255, 255),
             fondo=(0, 0, 0), grosor=1):
    """Escribe texto con un rectangulo de fondo para que se lea sobre el video."""
    (ancho, alto), base = cv2.getTextSize(mensaje, FUENTE, escala, grosor)
    x, y = int(origen[0]), int(origen[1])
    if fondo is not None:
        cv2.rectangle(img, (x - 4, y - alto - 6), (x + ancho + 4, y + base + 2), fondo, -1)
    cv2.putText(img, mensaje, (x, y), FUENTE, escala, color, grosor, cv2.LINE_AA)


def escribir_centrado(img, mensaje, centro_x, y, escala=0.6, color=(255, 255, 255), grosor=1):
    (ancho, _), _ = cv2.getTextSize(mensaje, FUENTE, escala, grosor)
    cv2.putText(img, mensaje, (int(centro_x - ancho / 2), int(y)), FUENTE, escala,
                color, grosor, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Camara: parametros intrinsecos
# ---------------------------------------------------------------------------
class Camara:
    """Guarda la 'matriz de camara': distancia focal y centro de la imagen.

    Sin calibrar usamos valores aproximados (funcionan bien para la demo).
    Si existe calibracion.npz (creado con calibrar_camara.py) se usa ese,
    y las distancias medidas seran mucho mas exactas.
    """

    def __init__(self, ancho, alto):
        archivo = os.path.join(CARPETA, "calibracion.npz")
        if os.path.exists(archivo):
            datos = np.load(archivo)
            self.matriz = datos["matriz"].astype(np.float64)
            self.distorsion = datos["distorsion"].astype(np.float64)
            # Si la calibracion se hizo con otra resolucion, se escala.
            self.matriz[0] *= ancho / float(datos["ancho"])
            self.matriz[1] *= alto / float(datos["alto"])
            self.calibrada = True
        else:
            focal = 0.9 * ancho  # aproximacion para una webcam tipica (~60 grados)
            self.matriz = np.array([[focal, 0, ancho / 2],
                                    [0, focal, alto / 2],
                                    [0, 0, 1]], dtype=np.float64)
            self.distorsion = np.zeros(5)
            self.calibrada = False


# ---------------------------------------------------------------------------
# Deteccion de marcadores y calculo de pose
# ---------------------------------------------------------------------------
@dataclass
class Marcador:
    id: int
    esquinas: np.ndarray  # 4x2 en pixeles: sup-izq, sup-der, inf-der, inf-izq
    rvec: np.ndarray      # rotacion (formato Rodrigues)
    tvec: np.ndarray      # posicion en metros respecto a la camara

    @property
    def centro(self):
        return self.esquinas.mean(axis=0)

    @property
    def distancia_cm(self):
        return float(np.linalg.norm(self.tvec)) * 100


class DetectorAruco:
    def __init__(self, tam_marcador_m):
        self.tam = tam_marcador_m
        diccionario = cv2.aruco.getPredefinedDictionary(DICCIONARIO)
        parametros = cv2.aruco.DetectorParameters()
        parametros.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # esquinas precisas
        self.detector = cv2.aruco.ArucoDetector(diccionario, parametros)

        # Esquinas del marcador en el mundo real (metros), con el origen en su centro.
        # Eje X a la derecha, Y hacia arriba y Z saliendo del marcador hacia la camara.
        m = tam_marcador_m / 2
        self.puntos_modelo = np.array([[-m, m, 0], [m, m, 0], [m, -m, 0], [-m, -m, 0]],
                                      dtype=np.float32)
        self.historial = {}  # id -> (rvec, tvec) del frame anterior

    def detectar(self, frame, camara):
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        esquinas, ids, _ = self.detector.detectMarkers(gris)

        marcadores = []
        if ids is not None:
            for esq, id_ in zip(esquinas, ids.flatten()):
                esq = esq.reshape(4, 2)
                # solvePnP: "si este cuadrado mide X cm y lo veo asi, donde esta?"
                ok, rvec, tvec = cv2.solvePnP(self.puntos_modelo, esq, camara.matriz,
                                              camara.distorsion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    rvec, tvec = self._suavizar(int(id_), rvec, tvec)
                    marcadores.append(Marcador(int(id_), esq, rvec, tvec))

        # Olvidamos los marcadores que ya no se ven para que no "salten" al reaparecer.
        visibles = {m.id for m in marcadores}
        for id_ in list(self.historial):
            if id_ not in visibles:
                del self.historial[id_]
        return marcadores

    def _suavizar(self, id_, rvec, tvec, peso_nuevo=0.6):
        """Promedia con el frame anterior para reducir el temblor de los objetos."""
        previo = self.historial.get(id_)
        if previo is not None and np.linalg.norm(rvec - previo[0]) < 0.5:
            rvec = peso_nuevo * rvec + (1 - peso_nuevo) * previo[0]
            tvec = peso_nuevo * tvec + (1 - peso_nuevo) * previo[1]
        self.historial[id_] = (rvec, tvec)
        return rvec, tvec


def proyectar(puntos_3d, marcador, camara):
    """Convierte puntos 3D (en coordenadas del marcador) a pixeles de la imagen."""
    pts, _ = cv2.projectPoints(np.asarray(puntos_3d, dtype=np.float32), marcador.rvec,
                               marcador.tvec, camara.matriz, camara.distorsion)
    return pts.reshape(-1, 2)


def contorno_marcador(frame, marcador, color=(0, 255, 0)):
    cv2.polylines(frame, [marcador.esquinas.astype(np.int32)], True, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# MODO 1: objetos 3D solidos
# ---------------------------------------------------------------------------
def rotacion_z(angulo):
    c, s = math.cos(angulo), math.sin(angulo)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def dibujar_solido(img, vertices, caras, marcador, camara):
    """Dibuja un objeto 3D relleno.

    vertices: puntos 3D del objeto.  caras: lista de (indices, color).
    Usa el 'algoritmo del pintor': primero las caras lejanas, luego las cercanas.
    """
    R, _ = cv2.Rodrigues(marcador.rvec)
    en_camara = vertices @ R.T + marcador.tvec.reshape(1, 3)  # posicion vista desde la camara
    pixeles = proyectar(vertices, marcador, camara).astype(np.int32)

    orden = sorted(caras, key=lambda cara: -en_camara[cara[0]][:, 2].mean())
    for indices, color in orden:
        p = en_camara[indices]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        normal /= np.linalg.norm(normal) + 1e-9
        luz = 0.35 + 0.65 * abs(normal[2])  # caras que miran a la camara se ven mas claras
        sombreado = tuple(int(c * luz) for c in color)
        cv2.fillConvexPoly(img, pixeles[indices], sombreado, cv2.LINE_AA)
        cv2.polylines(img, [pixeles[indices]], True, (20, 20, 20), 1, cv2.LINE_AA)


def modelo_cubo(lado):
    a = lado / 2
    v = np.array([[-a, -a, 0], [a, -a, 0], [a, a, 0], [-a, a, 0],
                  [-a, -a, lado], [a, -a, lado], [a, a, lado], [-a, a, lado]], dtype=np.float32)
    verde, azul, amarillo = (60, 200, 60), (220, 140, 40), (40, 210, 230)
    caras = [([0, 1, 2, 3], verde), ([4, 5, 6, 7], amarillo),
             ([0, 1, 5, 4], azul), ([1, 2, 6, 5], verde),
             ([2, 3, 7, 6], azul), ([3, 0, 4, 7], verde)]
    return v, caras


def modelo_piramide(lado):
    a = lado / 2
    v = np.array([[-a, -a, 0], [a, -a, 0], [a, a, 0], [-a, a, 0], [0, 0, lado * 1.3]],
                 dtype=np.float32)
    rojo, naranja = (50, 50, 220), (30, 140, 250)
    caras = [([0, 1, 2, 3], rojo), ([0, 1, 4], naranja), ([1, 2, 4], rojo),
             ([2, 3, 4], naranja), ([3, 0, 4], rojo)]
    return v, caras


def modelo_casa(lado):
    a = lado / 2
    alto = lado * 0.8
    v = np.array([[-a, -a, 0], [a, -a, 0], [a, a, 0], [-a, a, 0],
                  [-a, -a, alto], [a, -a, alto], [a, a, alto], [-a, a, alto],
                  [0, 0, alto + lado * 0.6]], dtype=np.float32)
    pared, techo = (200, 230, 245), (40, 40, 180)
    caras = [([0, 1, 5, 4], pared), ([1, 2, 6, 5], pared), ([2, 3, 7, 6], pared),
             ([3, 0, 4, 7], pared), ([4, 5, 8], techo), ([5, 6, 8], techo),
             ([6, 7, 8], techo), ([7, 4, 8], techo)]
    return v, caras


class ModoObjetos3D:
    nombre = "Objetos 3D"
    ayuda = ["ID 0: cubo que gira",
             "ID 1: piramide",
             "ID 2: casa",
             "Otro ID: ejes X (rojo), Y (verde), Z (azul)",
             "Inclina el marcador y mira el objeto",
             "desde distintos angulos."]

    def dibujar(self, frame, marcadores, camara, t, tam):
        for m in marcadores:
            contorno_marcador(frame, m)
            if m.id in (0, 1, 2):
                constructor = {0: modelo_cubo, 1: modelo_piramide, 2: modelo_casa}[m.id]
                vertices, caras = constructor(tam)
                if m.id == 0:  # el cubo gira 45 grados por segundo
                    vertices = vertices @ rotacion_z(math.radians(45 * t)).T
                dibujar_solido(frame, vertices, caras, m, camara)
            else:
                cv2.drawFrameAxes(frame, camara.matriz, camara.distorsion, m.rvec, m.tvec, tam)
            x, y = m.esquinas[:, 0].min(), m.esquinas[:, 1].max() + 22
            escribir(frame, f"ID {m.id} | {m.distancia_cm:.0f} cm", (x, y), 0.5)


# ---------------------------------------------------------------------------
# MODO 2: sistema solar (imagenes pegadas sobre el marcador)
# ---------------------------------------------------------------------------
PLANETAS = {  # id: (nombre, color BGR, datos curiosos)
    0: ("Sol", (0, 190, 255), ["Es una estrella", "Diametro: 1,392,700 km", "Superficie: 5,500 C"]),
    1: ("Mercurio", (150, 150, 160), ["Diametro: 4,879 km", "El mas cercano al Sol", "No tiene lunas"]),
    2: ("Venus", (110, 190, 230), ["Diametro: 12,104 km", "El mas caliente: 465 C", "Gira al reves"]),
    3: ("Tierra", (200, 120, 40), ["Diametro: 12,742 km", "Tiene 1 luna", "71% es agua"]),
    4: ("Marte", (40, 70, 200), ["Diametro: 6,779 km", "2 lunas: Fobos y Deimos", "Monte Olimpo: 22 km"]),
    5: ("Jupiter", (110, 160, 210), ["Diametro: 139,820 km", "El planeta mas grande", "Gran Mancha Roja"]),
    6: ("Saturno", (130, 200, 225), ["Diametro: 116,460 km", "Anillos de hielo y roca", "Flotaria en agua"]),
    7: ("Urano", (215, 210, 140), ["Diametro: 50,724 km", "Gira de lado (98 grados)", "Planeta de hielo"]),
    8: ("Neptuno", (190, 90, 40), ["Diametro: 49,244 km", "Vientos de 2,100 km/h", "El mas lejano"]),
}
TAM_TARJETA = 400


def crear_tarjeta(id_):
    """Crea la imagen que se pega sobre el marcador.

    Si existe imagenes/<id>.png (o .jpg) se usa esa imagen: asi los alumnos
    pueden poner sus propias fotos sobre los marcadores.
    """
    for ext in (".png", ".jpg", ".jpeg"):
        ruta = os.path.join(CARPETA, "imagenes", f"{id_}{ext}")
        imagen = cv2.imread(ruta) if os.path.exists(ruta) else None
        if imagen is not None:
            return cv2.resize(imagen, (TAM_TARJETA, TAM_TARJETA))

    t = TAM_TARJETA
    tarjeta = np.full((t, t, 3), (45, 20, 10), np.uint8)  # fondo azul noche
    rng = np.random.default_rng(id_)
    for _ in range(90):  # estrellas
        cv2.circle(tarjeta, (int(rng.integers(0, t)), int(rng.integers(0, t))), 1, (255, 255, 255), -1)

    if id_ in PLANETAS:
        nombre, color, datos = PLANETAS[id_]
        cx, cy, r = t // 2, 150, 85
        for k in range(r, 0, -2):  # circulos cada vez mas claros = efecto de esfera
            f = 0.55 + 0.6 * (1 - k / r)
            desplazamiento = int((r - k) * 0.35)
            tono = tuple(min(255, int(c * f)) for c in color)
            cv2.circle(tarjeta, (cx - desplazamiento, cy - desplazamiento), k, tono, -1, cv2.LINE_AA)
        if nombre == "Saturno":
            cv2.ellipse(tarjeta, (cx, cy), (140, 30), -15, 0, 360, (170, 215, 240), 5, cv2.LINE_AA)
    else:
        nombre = f"Marcador {id_}"
        datos = ["Pon una imagen en", f"imagenes/{id_}.png", "para personalizarlo"]

    escribir_centrado(tarjeta, nombre, t / 2, 285, 1.2, (255, 255, 255), 2)
    for i, linea in enumerate(datos):
        escribir_centrado(tarjeta, linea, t / 2, 322 + i * 28, 0.6, (200, 230, 255), 1)
    cv2.rectangle(tarjeta, (4, 4), (t - 5, t - 5), (0, 200, 255), 4)
    return tarjeta


def pegar_imagen(frame, imagen, destino):
    """Deforma 'imagen' para que encaje en los 4 puntos 'destino' (homografia)."""
    h, w = imagen.shape[:2]
    origen = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matriz = cv2.getPerspectiveTransform(origen, np.float32(destino))
    alto, ancho = frame.shape[:2]
    deformada = cv2.warpPerspective(imagen, matriz, (ancho, alto))
    mascara = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matriz, (ancho, alto))
    frame[mascara > 0] = deformada[mascara > 0]


class ModoSistemaSolar:
    nombre = "Sistema solar"
    ayuda = ["ID 0: Sol, ID 1-8: planetas",
             "Pon varios marcadores en la mesa",
             "y arma tu propio sistema solar.",
             "Tus imagenes: carpeta imagenes/<id>.png"]

    def __init__(self):
        self.tarjetas = {}  # cache: cada tarjeta se crea una sola vez

    def dibujar(self, frame, marcadores, camara, t, tam):
        # Primero los marcadores lejanos para que los cercanos queden encima.
        for m in sorted(marcadores, key=lambda m: -m.distancia_cm):
            if m.id not in self.tarjetas:
                self.tarjetas[m.id] = crear_tarjeta(m.id)
            a = tam * 0.85  # la tarjeta es un poco mas grande que el marcador
            esquinas_3d = [[-a, a, 0], [a, a, 0], [a, -a, 0], [-a, -a, 0]]
            pegar_imagen(frame, self.tarjetas[m.id], proyectar(esquinas_3d, m, camara))


# ---------------------------------------------------------------------------
# MODO 3: regla AR (medir distancias)
# ---------------------------------------------------------------------------
class ModoRegla:
    nombre = "Regla AR"
    ayuda = ["Muestra 2 o mas marcadores.",
             "Se mide la distancia real entre",
             "sus centros, en centimetros.",
             "Para mas exactitud: calibrar_camara.py",
             "y usa --tam con la medida real."]

    def dibujar(self, frame, marcadores, camara, t, tam):
        for m in marcadores:
            contorno_marcador(frame, m, (255, 200, 0))
            cv2.circle(frame, tuple(m.centro.astype(int)), 6, (0, 255, 255), -1)
            x, y = m.esquinas[:, 0].min(), m.esquinas[:, 1].max() + 22
            escribir(frame, f"ID {m.id}: a {m.distancia_cm:.0f} cm de la camara", (x, y), 0.45)

        if len(marcadores) < 2:
            h, w = frame.shape[:2]
            escribir(frame, "Muestra 2 o mas marcadores para medir", (w // 2 - 220, h // 2), 0.7,
                     (0, 255, 255))
            return

        for k, (a, b) in enumerate(itertools.combinations(marcadores[:4], 2)):
            # Distancia 3D real: resta de las posiciones (tvec) de ambos marcadores.
            distancia = float(np.linalg.norm(a.tvec - b.tvec)) * 100
            p1, p2 = a.centro, b.centro
            cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 255), 3, cv2.LINE_AA)
            # La etiqueta se separa de la linea (a un lado u otro) para que no se encimen.
            direccion = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-9)
            perpendicular = np.array([direccion[1], -direccion[0]])
            lado = 35 if k % 2 == 0 else -45
            medio = (p1 + p2) / 2 + perpendicular * lado - [45, -8]
            escribir(frame, f"{distancia:.1f} cm", medio, 0.8, (0, 0, 0), (0, 255, 255), 2)


# ---------------------------------------------------------------------------
# MODO 4: perilla de control (el marcador como interfaz)
# ---------------------------------------------------------------------------
class ModoPerilla:
    nombre = "Perilla de control"
    ayuda = ["Gira el marcador como una perilla.",
             "ID 0: controla el volumen",
             "ID 1: controla el brillo del video",
             "Derecha = sube, izquierda = baja"]
    ETIQUETAS = {0: "Volumen", 1: "Brillo"}

    def __init__(self):
        self.valores = {}  # id -> valor 0..100 (suavizado)

    def dibujar(self, frame, marcadores, camara, t, tam):
        for m in marcadores:
            # Angulo del borde superior del marcador en la imagen (0 = derecho).
            sup_izq, sup_der = m.esquinas[0], m.esquinas[1]
            angulo = math.degrees(math.atan2(sup_der[1] - sup_izq[1], sup_der[0] - sup_izq[0]))
            objetivo = (np.clip(angulo, -135, 135) + 135) / 270 * 100
            previo = self.valores.get(m.id, objetivo)
            self.valores[m.id] = 0.7 * previo + 0.3 * objetivo

        if any(m.id == 1 for m in marcadores):  # el brillo afecta a toda la imagen
            factor = 0.2 + self.valores[1] / 100 * 1.6
            cv2.convertScaleAbs(frame, frame, alpha=factor)

        for i, m in enumerate(marcadores[:3]):
            valor = self.valores[m.id]
            etiqueta = self.ETIQUETAS.get(m.id, f"Control {m.id}")
            centro = tuple(m.centro.astype(int))
            radio = int(np.linalg.norm(m.esquinas[0] - m.esquinas[2]) * 0.75)
            color = (0, int(255 * (1 - valor / 100)), int(255 * valor / 100))  # verde -> rojo
            fin = 135 + 270 * valor / 100
            cv2.ellipse(frame, centro, (radio, radio), 0, 135, 405, (70, 70, 70), 10, cv2.LINE_AA)
            cv2.ellipse(frame, centro, (radio, radio), 0, 135, fin, color, 10, cv2.LINE_AA)
            aguja = (int(centro[0] + radio * math.cos(math.radians(fin))),
                     int(centro[1] + radio * math.sin(math.radians(fin))))
            cv2.line(frame, centro, aguja, (255, 255, 255), 3, cv2.LINE_AA)
            escribir(frame, f"{etiqueta}: {valor:.0f}%", (centro[0] - 60, centro[1] - radio - 15),
                     0.7, (255, 255, 255), (0, 0, 0), 2)

            # Barra horizontal abajo a la izquierda, como un indicador de nivel
            h = frame.shape[0]
            y = h - 55 - i * 35
            x0, x1 = 140, 440
            cv2.rectangle(frame, (x0, y - 18), (x1, y), (60, 60, 60), -1)
            cv2.rectangle(frame, (x0, y - 18), (int(x0 + (x1 - x0) * valor / 100), y), color, -1)
            escribir(frame, etiqueta, (12, y - 2), 0.55)


# ---------------------------------------------------------------------------
# Interfaz y programa principal
# ---------------------------------------------------------------------------
def dibujar_hud(frame, numero, modo, fps, n_marcadores, ayuda, camara):
    h, w = frame.shape[:2]
    frame[:36] = (frame[:36] * 0.35).astype(np.uint8)       # barra superior oscura
    frame[h - 30:] = (frame[h - 30:] * 0.35).astype(np.uint8)  # barra inferior oscura
    escribir(frame, f"Modo {numero}: {modo.nombre}", (12, 25), 0.7, (0, 255, 255), None, 2)
    calib = "calibrada" if camara.calibrada else "sin calibrar"
    escribir(frame, f"Marcadores: {n_marcadores}  FPS: {fps:.0f}  Camara {calib}",
             (w - 430, 25), 0.55, (255, 255, 255), None)
    escribir(frame, "[1-4] Modo   [H] Ayuda   [S] Captura   [Q] Salir", (12, h - 10), 0.5,
             (220, 220, 220), None)
    if ayuda:
        alto_panel = 20 + 24 * len(modo.ayuda)
        sub = frame[50:50 + alto_panel, 10:400]
        sub[:] = (sub * 0.3).astype(np.uint8)
        for i, linea in enumerate(modo.ayuda):
            escribir(frame, linea, (20, 75 + i * 24), 0.55, (255, 255, 255), None)


def abrir_camara(indice, ancho, alto):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(indice, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(indice)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # mas FPS en HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    return cap


def main():
    parser = argparse.ArgumentParser(description="Laboratorio de Realidad Aumentada con ArUco")
    parser.add_argument("--camara", type=int, default=0, help="indice de la camara (0, 1, ...)")
    parser.add_argument("--tam", type=float, default=5.0,
                        help="lado del marcador impreso en cm (mide el cuadro negro)")
    parser.add_argument("--ancho", type=int, default=1280)
    parser.add_argument("--alto", type=int, default=720)
    args = parser.parse_args()

    cap = abrir_camara(args.camara, args.ancho, args.alto)
    ok, frame = cap.read()
    if not ok:
        print("No se pudo abrir la camara. Prueba con --camara 1")
        return 1

    camara = Camara(frame.shape[1], frame.shape[0])
    detector = DetectorAruco(args.tam / 100)
    modos = [ModoObjetos3D(), ModoSistemaSolar(), ModoRegla(), ModoPerilla()]
    actual, ayuda, fps = 0, True, 0.0
    inicio = anterior = time.time()
    os.makedirs(os.path.join(CARPETA, "capturas"), exist_ok=True)

    ventana = "Laboratorio de Realidad Aumentada"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    print(f"Resolucion {frame.shape[1]}x{frame.shape[0]} | marcador de {args.tam} cm")
    print("Teclas: 1-4 cambia de modo, H ayuda, S captura, Q salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Se perdio la senal de la camara")
            break

        marcadores = detector.detectar(frame, camara)
        modos[actual].dibujar(frame, marcadores, camara, time.time() - inicio, detector.tam)

        ahora = time.time()
        fps = 0.9 * fps + 0.1 / max(ahora - anterior, 1e-6)
        anterior = ahora
        dibujar_hud(frame, actual + 1, modos[actual], fps, len(marcadores), ayuda, camara)
        cv2.imshow(ventana, frame)

        tecla = cv2.waitKey(1) & 0xFF
        letra = chr(tecla).lower() if tecla != 255 else ""
        if letra == "q" or tecla == 27:
            break
        if letra and letra in "1234":
            actual = int(letra) - 1
        elif letra == "h":
            ayuda = not ayuda
        elif letra == "s":
            ruta = os.path.join(CARPETA, "capturas", time.strftime("ar_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(ruta, frame)
            print("Captura guardada:", ruta)
        if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:  # cerraron con la X
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
