# Laboratorio de Visión por Computadora: Haar Cascade

Esta app usa la cámara para **encontrar rostros, ojos y sonrisas** con clasificadores **Haar Cascade** y, con eso, hace cosas útiles y divertidas: contar personas, ocultar caras para proteger la privacidad, poner filtros como en las redes sociales, tomar una selfie cuando sonríes y hasta **reconocer quién eres** por tu nombre.

## ¿Qué puedes hacer?

| Tecla | Modo | Qué hace |
|---|---|---|
| `1` | **Detector** | Encierra rostros (verde), ojos (azul) y sonrisas (amarillo), y cuenta cuántas personas hay. Arriba a la derecha ves la imagen en grises que analiza el algoritmo. |
| `2` | **Privacidad** | Pixela automáticamente todas las caras, como en las noticias o en Google Street View. `B` cambia entre pixelado y desenfoque. |
| `3` | **Filtros** | Dibuja un gorro de fiesta, lentes, bigote u orejas de gato que siguen tu cara. `F` cambia de filtro. |
| `4` | **Selfie sonriente** | Sonríe y mantén la sonrisa: se llena una barra y la foto se toma sola. Se guarda en `capturas/`. |
| `5` | **Reconocimiento** | Presiona `R`, escribe tu nombre y presiona `Enter`. La app toma 30 fotos de tu cara, aprende y después te reconoce. `+` y `-` cambian el umbral. |

Otras teclas: `H` muestra u oculta la ayuda, `S` guarda una foto y `Q` (o `Esc`) cierra la app.

## Instalación y uso

```bash
pip install -r requirements.txt
python app.py
```

- Si tienes varias cámaras: `python app.py --camara 1`
- Se necesita `opencv-contrib-python` porque el reconocedor LBPH (`cv2.face`) viene en el paquete "contrib". Instala **solo** ese paquete: si también tienes `opencv-python`, pueden chocar. Para arreglarlo: `pip uninstall -y opencv-python opencv-contrib-python` y luego `pip install opencv-contrib-python`.

## Detectar no es lo mismo que reconocer

- **Detectar** = "aquí hay *una* cara". Esto lo hace Haar Cascade (modos 1 a 4).
- **Reconocer** = "esta cara es de *Ana*". Para eso primero se detecta la cara con Haar y luego se compara con caras guardadas usando **LBPH** (modo 5).

## ¿Cómo funciona Haar Cascade?

1. **Características Haar**: son rectángulos blancos y negros que miden diferencias de luz. Por ejemplo, la zona de los ojos suele ser más oscura que la de las mejillas, y el puente de la nariz es más claro que los ojos.
2. **Ventana deslizante**: una ventana recorre la imagen a muchos tamaños y en cada posición pregunta "¿esto parece una cara?".
3. **Cascada**: la pregunta se hace por etapas. Las primeras son muy simples y descartan rápido casi todo lo que no es cara. Solo las zonas que pasan todas las etapas se aceptan como rostro. Por eso es tan rápido.
4. El "conocimiento" ya viene entrenado en archivos `.xml` que trae OpenCV (`haarcascade_frontalface_alt.xml`, `haarcascade_eye_tree_eyeglasses.xml` y `haarcascade_smile.xml`).

**Limitaciones**: funciona mejor con la cara de frente y con buena luz. Si giras mucho la cabeza o hay sombras fuertes, puede fallar. Esto también es parte de lo que se aprende.

## Partes importantes del código (`app.py`)

### `cargar_cascada`: cargar el clasificador
```python
for carpeta in (CARPETA, cv2.data.haarcascades):
    cascada = cv2.CascadeClassifier(ruta)
```
Primero busca el `.xml` junto al programa y, si no lo encuentra, usa los que vienen instalados con OpenCV (`cv2.data.haarcascades`).

### `DetectorHaar.preparar`: preparar la imagen
```python
gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
return cv2.equalizeHist(gris)
```
Haar solo usa la luz, no el color, así que se pasa a **escala de grises**. `equalizeHist` reparte mejor el contraste para que funcione aunque la iluminación no sea perfecta.

### `DetectorHaar.rostros`: el detector
```python
caras = self.cascada_rostro.detectMultiScale(
    gris, scaleFactor=1.1, minNeighbors=5, minSize=(minimo, minimo))
```
- `scaleFactor=1.1`: en cada pasada la imagen se reduce un 10 % para encontrar caras grandes y pequeñas.
- `minNeighbors=5`: cuántas detecciones cercanas se necesitan para confirmar una cara. Más alto = menos errores, pero puede perder caras.
- `minSize`: ignora objetos demasiado pequeños.
- Devuelve una lista de rectángulos `(x, y, ancho, alto)`.

### `ojos` y `sonrisas`: buscar dentro de la cara
Los ojos se buscan solo en la **mitad de arriba** del rostro y la sonrisa solo en la **mitad de abajo**. Así el programa es más rápido y se equivoca menos (por ejemplo, no confunde una fosa nasal con un ojo).

### Modo 2, `ModoPrivacidad`: pixelar
```python
chica = cv2.resize(zona, (12, 12))
frame[y:y+h, x:x+w] = cv2.resize(chica, (w, h), interpolation=cv2.INTER_NEAREST)
```
Se reduce la cara a 12x12 pixeles y se vuelve a agrandar sin suavizar: quedan cuadros grandes y no se reconoce a la persona.

### Modo 3, `dibujar_lentes`, `dibujar_gorro`...: filtros
Cada filtro se dibuja usando **proporciones del rectángulo de la cara**. Por ejemplo, los ojos están más o menos al 40 % de la altura y al 31 % y 69 % del ancho. Por eso los lentes siguen tu cara aunque te muevas o te acerques.

### Modo 4, `ModoSelfie`: disparo automático
Cuenta cuántos frames seguidos detecta una sonrisa. Cuando llega a 12 (aproximadamente medio segundo), guarda la foto **sin dibujos** (`limpio`) y muestra un destello blanco. Entre una foto y otra espera 3 segundos.

### Modo 5, `ModoReconocimiento`: reconocimiento con LBPH
```python
self.modelo = cv2.face.LBPHFaceRecognizer_create()
self.modelo.train(imagenes, np.array(etiquetas))
etiqueta, distancia = self.modelo.predict(recortar_rostro(gris, rostro))
```
- **Registro**: se guardan 30 recortes de tu cara en grises (150x150) en `rostros/<tu_nombre>/`.
- **Entrenamiento** (`entrenar`): LBPH (*Local Binary Patterns Histograms*) compara cada pixel con sus 8 vecinos y forma un código de 0 y 1 que describe la **textura** de la piel, los ojos y la boca. Luego hace histogramas de esos códigos por zonas de la cara.
- **Predicción**: compara el histograma de la cara nueva con los guardados y devuelve la persona más parecida y una **distancia**. Distancia baja = se parece mucho. Si la distancia es mayor que el **umbral**, se muestra "Desconocido".
- Prueba a subir y bajar el umbral con `+` y `-` y observa cuándo se confunde.

### `main`: el ciclo principal
Leer frame → voltear como espejo → pasar a grises → detectar rostros → el modo activo dibuja → interfaz → leer teclado. Mientras escribes tu nombre, las teclas van al cuadro de texto y no cambian de modo.

## Privacidad

Las fotos del modo 5 se guardan **solo en esta computadora**, en la carpeta `rostros/`. Para que la app "olvide" a todos, borra esa carpeta. Las selfies quedan en `capturas/`. Pide permiso antes de registrar la cara de otra persona.

## Retos para los alumnos

1. Cambia `minNeighbors` a 2 y luego a 10. ¿Qué pasa con los falsos positivos?
2. Crea un filtro nuevo (corona, nariz de payaso, antifaz) con `cv2.circle`, `cv2.ellipse` o `cv2.fillPoly`.
3. Usa `haarcascade_frontalcatface.xml` para detectar caras de gato en una foto.
4. Haz una "alarma de somnolencia": si no se detectan ojos durante 2 segundos, muestra una alerta en rojo.
5. Registra a 3 compañeros y mide cuántas veces los reconoce bien con distintos umbrales.
