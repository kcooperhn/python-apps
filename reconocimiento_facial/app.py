"""
Laboratorio de Vision por Computadora: Haar Cascade
===================================================

Un clasificador Haar Cascade busca patrones de luz y sombra (por ejemplo:
"la zona de los ojos es mas oscura que las mejillas") para encontrar rostros,
ojos y sonrisas en la imagen, de forma muy rapida.

Modos (teclas 1-5):
  1. Detector          -> rostros, ojos y sonrisas + contador de personas
  2. Privacidad        -> pixela o desenfoca los rostros automaticamente
  3. Filtros           -> lentes, bigote, gorro de fiesta y gato
  4. Selfie sonriente  -> toma la foto sola cuando sonries
  5. Reconocimiento    -> aprende tu cara y te reconoce por tu nombre (LBPH)
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

CARPETA = os.path.dirname(os.path.abspath(__file__))
CARPETA_ROSTROS = os.path.join(CARPETA, "rostros")
CARPETA_CAPTURAS = os.path.join(CARPETA, "capturas")
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
# Nota: las fuentes de OpenCV no tienen acentos, por eso los textos van sin acentos.


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


def ampliar(rostro, factor, forma):
    """Agranda un rectangulo (x, y, w, h) desde su centro sin salirse de la imagen."""
    x, y, w, h = rostro
    dx, dy = int(w * factor), int(h * factor)
    alto, ancho = forma[:2]
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(ancho, x + w + dx), min(alto, y + h + dy)
    return x0, y0, x1 - x0, y1 - y0


# ---------------------------------------------------------------------------
# Detector Haar: rostros, ojos y sonrisas
# ---------------------------------------------------------------------------
def cargar_cascada(nombre):
    """Busca el archivo XML junto a este script o en los que trae OpenCV."""
    for carpeta in (CARPETA, cv2.data.haarcascades):
        ruta = os.path.join(carpeta, nombre)
        if os.path.exists(ruta):
            cascada = cv2.CascadeClassifier(ruta)
            if not cascada.empty():
                return cascada
    raise FileNotFoundError(f"No se encontro el clasificador {nombre}")


class DetectorHaar:
    def __init__(self):
        self.cascada_rostro = cargar_cascada("haarcascade_frontalface_alt.xml")
        self.cascada_ojos = cargar_cascada("haarcascade_eye_tree_eyeglasses.xml")
        self.cascada_sonrisa = cargar_cascada("haarcascade_smile.xml")

    @staticmethod
    def preparar(frame):
        """Haar trabaja en escala de grises; ecualizar mejora el contraste."""
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gris)

    def rostros(self, gris):
        minimo = max(60, gris.shape[0] // 8)  # ignora "caras" demasiado pequenas
        caras = self.cascada_rostro.detectMultiScale(
            gris,
            scaleFactor=1.1,   # cuanto se reduce la imagen en cada pasada (10%)
            minNeighbors=5,    # cuantas detecciones vecinas se necesitan para confirmar
            minSize=(minimo, minimo),
        )
        return [tuple(int(v) for v in c) for c in caras]

    def ojos(self, gris, rostro):
        x, y, w, h = rostro
        zona = gris[y:y + int(h * 0.55), x:x + w]  # los ojos estan en la mitad superior
        ojos = self.cascada_ojos.detectMultiScale(zona, 1.1, 8, minSize=(w // 8, w // 8))
        return [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in ojos][:2]

    def sonrisas(self, gris, rostro):
        x, y, w, h = rostro
        y0 = y + h // 2  # la boca esta en la mitad inferior
        zona = gris[y0:y + h, x:x + w]
        sonrisas = self.cascada_sonrisa.detectMultiScale(zona, 1.7, 22, minSize=(w // 4, h // 10))
        return [(x + sx, y0 + sy, sw, sh) for sx, sy, sw, sh in sonrisas][:1]


# ---------------------------------------------------------------------------
# MODO 1: detector
# ---------------------------------------------------------------------------
class ModoDetector:
    nombre = "Detector de rostros"
    ayuda = ["Verde: rostro   Azul: ojos",
             "Amarillo: sonrisa",
             "Arriba a la derecha ves la imagen",
             "en grises que analiza el algoritmo."]

    def procesar(self, frame, limpio, gris, rostros, detector):
        for i, (x, y, w, h) in enumerate(rostros, start=1):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            escribir(frame, f"Rostro {i}", (x, y - 10), 0.6, (0, 0, 0), (0, 255, 0))
            for ex, ey, ew, eh in detector.ojos(gris, (x, y, w, h)):
                cv2.circle(frame, (ex + ew // 2, ey + eh // 2), ew // 2, (255, 120, 0), 2)
            for sx, sy, sw, sh in detector.sonrisas(gris, (x, y, w, h)):
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
                escribir(frame, "Sonriendo!", (x, y + h + 25), 0.6, (0, 0, 0), (0, 255, 255))

        # Miniatura: lo que "ve" el algoritmo (grises ecualizados)
        alto, ancho = frame.shape[:2]
        mini = cv2.resize(gris, (ancho // 4, alto // 4))
        mini = cv2.cvtColor(mini, cv2.COLOR_GRAY2BGR)
        frame[45:45 + mini.shape[0], ancho - mini.shape[1] - 10:ancho - 10] = mini
        escribir(frame, "Lo que ve el algoritmo", (ancho - mini.shape[1] - 6, 45 + mini.shape[0] + 20),
                 0.45)

        h = frame.shape[0]
        escribir(frame, f"Personas: {len(rostros)}", (12, h - 45), 0.9, (0, 0, 0), (0, 255, 0), 2)


# ---------------------------------------------------------------------------
# MODO 2: privacidad
# ---------------------------------------------------------------------------
class ModoPrivacidad:
    nombre = "Privacidad (anonimizar)"
    ayuda = ["Los rostros se ocultan solos,",
             "como en noticias o Street View.",
             "[B] cambia pixelado / desenfoque"]

    def __init__(self):
        self.pixelado = True

    def tecla(self, letra):
        if letra == "b":
            self.pixelado = not self.pixelado

    def procesar(self, frame, limpio, gris, rostros, detector):
        for rostro in rostros:
            x, y, w, h = ampliar(rostro, 0.12, frame.shape)
            zona = frame[y:y + h, x:x + w]
            if self.pixelado:
                # Reducir a 12x12 y volver a agrandar sin suavizar = cuadros grandes
                chica = cv2.resize(zona, (12, 12), interpolation=cv2.INTER_LINEAR)
                frame[y:y + h, x:x + w] = cv2.resize(chica, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                k = (w // 3) | 1  # el tamano del filtro debe ser impar
                frame[y:y + h, x:x + w] = cv2.GaussianBlur(zona, (k, k), 0)
        tipo = "pixelado" if self.pixelado else "desenfoque"
        escribir(frame, f"Rostros ocultos: {len(rostros)} ({tipo})", (12, frame.shape[0] - 45),
                 0.7, (0, 0, 0), (0, 200, 255), 2)


# ---------------------------------------------------------------------------
# MODO 3: filtros divertidos (dibujados con figuras de OpenCV)
# ---------------------------------------------------------------------------
def dibujar_lentes(img, x, y, w, h):
    ojo_izq = (x + int(w * 0.31), y + int(h * 0.40))
    ojo_der = (x + int(w * 0.69), y + int(h * 0.40))
    radio = int(w * 0.15)
    for centro in (ojo_izq, ojo_der):
        cv2.circle(img, centro, radio, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, centro, radio, (0, 215, 255), 3, cv2.LINE_AA)
        brillo = (centro[0] - radio // 3, centro[1] - radio // 3)
        cv2.circle(img, brillo, max(2, radio // 5), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.line(img, (ojo_izq[0] + radio, ojo_izq[1]), (ojo_der[0] - radio, ojo_der[1]), (0, 215, 255), 3)
    cv2.line(img, (ojo_izq[0] - radio, ojo_izq[1]), (x, ojo_izq[1] - radio // 3), (0, 215, 255), 3)
    cv2.line(img, (ojo_der[0] + radio, ojo_der[1]), (x + w, ojo_der[1] - radio // 3), (0, 215, 255), 3)


def dibujar_bigote(img, x, y, w, h):
    cx, cy = x + w // 2, y + int(h * 0.72)
    eje = (int(w * 0.14), int(w * 0.05))
    cv2.ellipse(img, (cx - eje[0], cy), eje, 15, 0, 360, (30, 30, 40), -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx + eje[0], cy), eje, -15, 0, 360, (30, 30, 40), -1, cv2.LINE_AA)


def dibujar_gorro(img, x, y, w, h):
    punta = (x + w // 2, y - int(h * 0.75))
    base = np.array([punta, (x + int(w * 0.22), y + int(h * 0.05)),
                     (x + int(w * 0.78), y + int(h * 0.05))], np.int32)
    cv2.fillPoly(img, [base], (180, 60, 230), cv2.LINE_AA)
    for k in range(1, 4):  # rayas del gorro
        t = k / 4
        izq = (int(punta[0] + (base[1][0] - punta[0]) * t), int(punta[1] + (base[1][1] - punta[1]) * t))
        der = (int(punta[0] + (base[2][0] - punta[0]) * t), int(punta[1] + (base[2][1] - punta[1]) * t))
        cv2.line(img, izq, der, (0, 230, 255), max(3, w // 30), cv2.LINE_AA)
    cv2.circle(img, punta, max(6, w // 12), (255, 255, 255), -1, cv2.LINE_AA)


def dibujar_gato(img, x, y, w, h):
    rosa, oscuro = (180, 130, 255), (60, 60, 60)
    for lado in (-1, 1):
        cx = x + w // 2 + lado * int(w * 0.30)
        oreja = np.array([(cx - int(w * 0.15), y + int(h * 0.05)),
                          (cx + lado * int(w * 0.05), y - int(h * 0.35)),
                          (cx + int(w * 0.15), y + int(h * 0.05))], np.int32)
        cv2.fillPoly(img, [oreja], oscuro, cv2.LINE_AA)
        cv2.fillPoly(img, [((oreja - oreja.mean(axis=0)) * 0.55 + oreja.mean(axis=0)).astype(np.int32)],
                     rosa, cv2.LINE_AA)
    nariz = (x + w // 2, y + int(h * 0.62))
    cv2.fillPoly(img, [np.array([(nariz[0] - w // 18, nariz[1] - h // 30),
                                 (nariz[0] + w // 18, nariz[1] - h // 30),
                                 (nariz[0], nariz[1] + h // 25)], np.int32)], rosa, cv2.LINE_AA)
    for lado in (-1, 1):
        for k in (-1, 0, 1):
            inicio = (nariz[0] + lado * int(w * 0.10), nariz[1] + k * h // 30)
            fin = (nariz[0] + lado * int(w * 0.45), nariz[1] + k * h // 12)
            cv2.line(img, inicio, fin, (40, 40, 40), 2, cv2.LINE_AA)


class ModoFiltros:
    nombre = "Filtros divertidos"
    ayuda = ["Como los filtros de redes sociales:",
             "se dibujan usando la posicion del rostro.",
             "[F] cambia de filtro"]
    FILTROS = [("Fiesta", [dibujar_gorro, dibujar_lentes]),
               ("Detective", [dibujar_lentes, dibujar_bigote]),
               ("Gato", [dibujar_gato])]

    def __init__(self):
        self.actual = 0

    def tecla(self, letra):
        if letra == "f":
            self.actual = (self.actual + 1) % len(self.FILTROS)

    def procesar(self, frame, limpio, gris, rostros, detector):
        nombre, funciones = self.FILTROS[self.actual]
        for x, y, w, h in rostros:
            for dibujar in funciones:
                dibujar(frame, x, y, w, h)
        escribir(frame, f"Filtro: {nombre}  [F] siguiente", (12, frame.shape[0] - 45), 0.7,
                 (0, 0, 0), (180, 60, 230), 2)


# ---------------------------------------------------------------------------
# MODO 4: selfie automatica al sonreir
# ---------------------------------------------------------------------------
class ModoSelfie:
    nombre = "Selfie sonriente"
    ayuda = ["Mira a la camara y sonrie.",
             "Si mantienes la sonrisa se llena la",
             "barra y la foto se toma sola.",
             "Las fotos quedan en capturas/"]
    FRAMES_NECESARIOS = 12
    ESPERA = 3.0  # segundos entre fotos

    def __init__(self):
        self.contador = 0
        self.ultima_foto = 0.0
        self.miniatura = None

    def procesar(self, frame, limpio, gris, rostros, detector):
        alto, ancho = frame.shape[:2]
        sonriendo = False
        if rostros:
            principal = max(rostros, key=lambda r: r[2] * r[3])  # el rostro mas grande
            x, y, w, h = principal
            sonriendo = bool(detector.sonrisas(gris, principal))
            color = (0, 255, 255) if sonriendo else (200, 200, 200)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        self.contador = self.contador + 1 if sonriendo else max(0, self.contador - 2)
        ahora = time.time()
        if self.contador >= self.FRAMES_NECESARIOS and ahora - self.ultima_foto > self.ESPERA:
            ruta = os.path.join(CARPETA_CAPTURAS, time.strftime("selfie_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(ruta, limpio)
            print("Selfie guardada:", ruta)
            self.miniatura = cv2.resize(limpio, (ancho // 5, alto // 5))
            self.ultima_foto, self.contador = ahora, 0

        # Barra de progreso de la sonrisa
        progreso = min(1.0, self.contador / self.FRAMES_NECESARIOS)
        cv2.rectangle(frame, (12, alto - 75), (312, alto - 50), (60, 60, 60), -1)
        cv2.rectangle(frame, (12, alto - 75), (12 + int(300 * progreso), alto - 50), (0, 255, 255), -1)
        texto = "Sonrie!" if rostros else "No veo ningun rostro"
        escribir(frame, texto, (325, alto - 55), 0.7, (0, 255, 255), None, 2)

        transcurrido = ahora - self.ultima_foto
        if transcurrido < 0.25:  # destello blanco de "camara"
            frame[:] = cv2.addWeighted(frame, 0.3, np.full_like(frame, 255), 0.7, 0)
        if transcurrido < 2.0:
            escribir(frame, "Foto guardada!", (ancho // 2 - 130, alto // 2), 1.2, (0, 0, 0), (0, 255, 0), 3)
        if self.miniatura is not None:
            mh, mw = self.miniatura.shape[:2]
            frame[alto - 90 - mh:alto - 90, ancho - mw - 10:ancho - 10] = self.miniatura
            escribir(frame, "Ultima foto", (ancho - mw - 6, alto - 95 - mh), 0.45)


# ---------------------------------------------------------------------------
# MODO 5: reconocimiento facial con LBPH
# ---------------------------------------------------------------------------
TAM_ROSTRO = (150, 150)


def recortar_rostro(gris, rostro):
    """Recorte del rostro en grises con tamano fijo: asi entrena y compara LBPH."""
    x, y, w, h = rostro
    return cv2.resize(gris[y:y + h, x:x + w], TAM_ROSTRO)


class ModoReconocimiento:
    nombre = "Reconocimiento facial (LBPH)"
    ayuda = ["[R] registrar una persona nueva",
             "[+] / [-] cambia el umbral",
             "Distancia baja = se parece mucho.",
             "Si supera el umbral: Desconocido."]
    MUESTRAS = 30

    def __init__(self):
        self.modelo = None
        self.nombres = {}       # etiqueta numerica -> nombre
        self.umbral = 70        # distancia maxima para aceptar que es la misma persona
        self.escribiendo = False
        self.texto = ""
        self.registrando = None  # nombre de la persona que se esta registrando
        self.guardadas = 0
        self.cuadro = 0
        self.mensaje, self.mensaje_hasta = "", 0.0
        self.entrenar()

    # --- entrenamiento -----------------------------------------------------
    def entrenar(self):
        """Lee rostros/<nombre>/*.png y entrena el reconocedor LBPH."""
        imagenes, etiquetas, self.nombres = [], [], {}
        if os.path.isdir(CARPETA_ROSTROS):
            for etiqueta, persona in enumerate(sorted(os.listdir(CARPETA_ROSTROS))):
                carpeta = os.path.join(CARPETA_ROSTROS, persona)
                if not os.path.isdir(carpeta):
                    continue
                for archivo in os.listdir(carpeta):
                    img = cv2.imread(os.path.join(carpeta, archivo), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        imagenes.append(cv2.resize(img, TAM_ROSTRO))
                        etiquetas.append(etiqueta)
                        self.nombres[etiqueta] = persona
        if not imagenes:
            self.modelo = None
            return
        self.modelo = cv2.face.LBPHFaceRecognizer_create()
        self.modelo.train(imagenes, np.array(etiquetas))
        print(f"Modelo entrenado: {len(imagenes)} fotos de {len(self.nombres)} persona(s)")

    # --- teclado -----------------------------------------------------------
    def tecla(self, letra):
        if letra == "r":
            self.escribiendo, self.texto = True, ""
        elif letra in "+=":  # '=' por si el '+' necesita Shift en el teclado
            self.umbral = min(150, self.umbral + 5)
        elif letra == "-":
            self.umbral = max(20, self.umbral - 5)

    def tecla_texto(self, codigo):
        """Mientras se escribe el nombre, todas las teclas llegan aqui."""
        if codigo == 13:  # Enter
            nombre = self.texto.strip().replace(" ", "_")
            if nombre:
                self.registrando, self.guardadas, self.cuadro = nombre, 0, 0
                os.makedirs(os.path.join(CARPETA_ROSTROS, nombre), exist_ok=True)
            self.escribiendo = False
        elif codigo == 27:  # Esc
            self.escribiendo = False
        elif codigo == 8:  # Borrar
            self.texto = self.texto[:-1]
        elif 32 <= codigo < 127 and len(self.texto) < 20:
            caracter = chr(codigo)
            if caracter.isalnum() or caracter in " _-":
                self.texto += caracter

    # --- cada frame ----------------------------------------------------------
    def procesar(self, frame, limpio, gris, rostros, detector):
        alto, ancho = frame.shape[:2]
        if self.registrando:
            self._registrar(frame, gris, rostros)
        else:
            for rostro in rostros:
                x, y, w, h = rostro
                if self.modelo is None:
                    texto, color = "Presiona R para registrarte", (0, 200, 255)
                else:
                    etiqueta, distancia = self.modelo.predict(recortar_rostro(gris, rostro))
                    if distancia <= self.umbral:
                        texto, color = f"{self.nombres[etiqueta]} ({distancia:.0f})", (0, 255, 0)
                    else:
                        texto, color = f"Desconocido ({distancia:.0f})", (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                escribir(frame, texto, (x, y - 10), 0.7, (0, 0, 0), color, 2)

        personas = ", ".join(self.nombres.values()) or "nadie"
        escribir(frame, f"Umbral: {self.umbral}   Registrados: {personas}", (12, alto - 45), 0.6,
                 (0, 0, 0), (0, 255, 255))

        if self.escribiendo:
            cv2.rectangle(frame, (ancho // 2 - 300, alto // 2 - 60), (ancho // 2 + 300, alto // 2 + 50),
                          (40, 40, 40), -1)
            cv2.rectangle(frame, (ancho // 2 - 300, alto // 2 - 60), (ancho // 2 + 300, alto // 2 + 50),
                          (0, 255, 255), 2)
            escribir(frame, "Escribe tu nombre y presiona Enter", (ancho // 2 - 280, alto // 2 - 25),
                     0.65, (255, 255, 255), None)
            cursor = "_" if int(time.time() * 2) % 2 else " "
            escribir(frame, self.texto + cursor, (ancho // 2 - 280, alto // 2 + 20), 0.9,
                     (0, 255, 255), None, 2)
        if time.time() < self.mensaje_hasta:
            escribir(frame, self.mensaje, (12, alto - 85), 0.8, (0, 0, 0), (0, 255, 0), 2)

    def _registrar(self, frame, gris, rostros):
        alto = frame.shape[0]
        if len(rostros) != 1:
            escribir(frame, "Debe haber UNA sola persona frente a la camara", (12, alto - 85), 0.6,
                     (0, 0, 0), (0, 0, 255), 2)
            return
        x, y, w, h = rostros[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        self.cuadro += 1
        if self.cuadro % 3 == 0:  # una foto cada 3 frames para tener variedad
            ruta = os.path.join(CARPETA_ROSTROS, self.registrando, f"{int(time.time() * 1000)}.png")
            cv2.imwrite(ruta, recortar_rostro(gris, rostros[0]))
            self.guardadas += 1
        progreso = self.guardadas / self.MUESTRAS
        cv2.rectangle(frame, (x, y + h + 10), (x + w, y + h + 30), (60, 60, 60), -1)
        cv2.rectangle(frame, (x, y + h + 10), (x + int(w * progreso), y + h + 30), (0, 255, 255), -1)
        escribir(frame, f"Registrando a {self.registrando}: {self.guardadas}/{self.MUESTRAS}",
                 (12, alto - 120), 0.7, (0, 0, 0), (0, 255, 255), 2)
        escribir(frame, "Mueve un poco la cabeza y cambia de gesto", (12, alto - 85), 0.6)
        if self.guardadas >= self.MUESTRAS:
            nombre, self.registrando = self.registrando, None
            self.entrenar()
            self.mensaje, self.mensaje_hasta = f"Listo! Ya te conozco, {nombre}", time.time() + 3


# ---------------------------------------------------------------------------
# Interfaz y programa principal
# ---------------------------------------------------------------------------
def dibujar_hud(frame, numero, modo, fps, ayuda):
    h, w = frame.shape[:2]
    frame[:36] = (frame[:36] * 0.35).astype(np.uint8)
    frame[h - 30:] = (frame[h - 30:] * 0.35).astype(np.uint8)
    escribir(frame, f"Modo {numero}: {modo.nombre}", (12, 25), 0.7, (0, 255, 255), None, 2)
    escribir(frame, f"FPS: {fps:.0f}", (w - 100, 25), 0.55, (255, 255, 255), None)
    escribir(frame, "[1-5] Modo  [H] Ayuda  [S] Foto  [R] Registrar  [Q] Salir",
             (12, h - 10), 0.5, (220, 220, 220), None)
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    return cap


def main():
    parser = argparse.ArgumentParser(description="Laboratorio de deteccion facial con Haar Cascade")
    parser.add_argument("--camara", type=int, default=0, help="indice de la camara (0, 1, ...)")
    parser.add_argument("--ancho", type=int, default=640)
    parser.add_argument("--alto", type=int, default=480)
    args = parser.parse_args()

    detector = DetectorHaar()
    cap = abrir_camara(args.camara, args.ancho, args.alto)
    if not cap.isOpened():
        print("No se pudo abrir la camara. Prueba con --camara 1")
        return 1

    os.makedirs(CARPETA_CAPTURAS, exist_ok=True)
    modos = [ModoDetector(), ModoPrivacidad(), ModoFiltros(), ModoSelfie(), ModoReconocimiento()]
    reconocimiento = modos[4]
    actual, ayuda, fps = 0, True, 0.0
    anterior = time.time()
    ventana = "Laboratorio Haar Cascade"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    print("Teclas: 1-5 modo, H ayuda, S foto, R registrar rostro, Q salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Se perdio la senal de la camara")
            break

        frame = cv2.flip(frame, 1)  # efecto espejo, como la camara frontal del celular
        limpio = frame.copy()       # copia sin dibujos (para las selfies)
        gris = detector.preparar(frame)
        rostros = detector.rostros(gris)
        modos[actual].procesar(frame, limpio, gris, rostros, detector)

        ahora = time.time()
        fps = 0.9 * fps + 0.1 / max(ahora - anterior, 1e-6)
        anterior = ahora
        dibujar_hud(frame, actual + 1, modos[actual], fps, ayuda)
        cv2.imshow(ventana, frame)

        codigo = cv2.waitKey(1) & 0xFF
        if codigo != 255 and reconocimiento.escribiendo:
            reconocimiento.tecla_texto(codigo)  # escribiendo un nombre: no son atajos
        elif codigo != 255:
            letra = chr(codigo).lower()
            if letra == "q" or codigo == 27:
                break
            if letra in "12345":
                actual = int(letra) - 1
            elif letra == "h":
                ayuda = not ayuda
            elif letra == "s":
                ruta = os.path.join(CARPETA_CAPTURAS, time.strftime("foto_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(ruta, frame)
                print("Foto guardada:", ruta)
            elif letra == "r":
                actual = 4
                reconocimiento.tecla("r")
            elif hasattr(modos[actual], "tecla"):
                modos[actual].tecla(letra)
        if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:  # cerraron con la X
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
