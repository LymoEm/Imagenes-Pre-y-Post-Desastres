import os
import cv2
import numpy as np
import joblib
import xml.etree.ElementTree as ET
from PIL import Image
import warnings

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")


from PIL import Image

def leer_imagen_segura(ruta):

    try:
        img_pil = Image.open(ruta).convert("RGB")

        img = np.array(img_pil)

        # convertir a formato opencv
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    except:
        return None


# preprocesamiento
def preprocesar(imagen):

    # redimensionar
    imagen = cv2.resize(imagen, (256, 256))

    # escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # reducir ruido
    gris = cv2.GaussianBlur(gris, (5,5), 0)

    # mejorar contraste
    gris = cv2.equalizeHist(gris)

    return gris


# descriptor global (bordes)
def descriptor_global(gris):

    bordes = cv2.Canny(gris, 100, 200)

    hist = cv2.calcHist([bordes], [0], None, [256], [0,256])
    hist = hist.flatten()

    # normalizar
    hist = hist / (np.sum(hist) + 1e-6)

    return hist


# verificar si es imagen
def es_imagen_valida(nombre):
    return nombre.lower().endswith(('.jpg','.jpeg'))


# leer xml (annotations)
def leer_xml(ruta_xml):

    try:
        tree = ET.parse(ruta_xml)
        root = tree.getroot()

        objetos = root.findall("object")

        # cantidad de objetos (daños)
        return len(objetos)

    except:
        return 0


# cargar dataset completo
def cargar_dataset(ruta_base):

    datos = []
    etiquetas = []

    clases = ["damage", "no_damage"]

    for clase in clases:

        ruta_img = os.path.join(ruta_base, clase, "images")
        ruta_xml = os.path.join(ruta_base, clase, "annotations")

        print("leyendo carpeta:", ruta_img)

        for archivo in os.listdir(ruta_img):

            if not es_imagen_valida(archivo):
                continue

            ruta_imagen = os.path.join(ruta_img, archivo)

            try:
                # leer imagen segura
                imagen = leer_imagen_segura(ruta_imagen)

                if imagen is None:
                    print("imagen corrupta:", archivo)
                    continue

                # evitar imagen muy pequeña
                if imagen.shape[0] < 50 or imagen.shape[1] < 50:
                    continue

                gris = preprocesar(imagen)

                desc = descriptor_global(gris)

                # leer annotation
                nombre_xml = archivo.replace(".jpg", ".xml")
                ruta_anot = os.path.join(ruta_xml, nombre_xml)

                num_objetos = leer_xml(ruta_anot)

                # agregar feature extra
                desc = np.append(desc, num_objetos)

                datos.append(desc)

                # etiqueta
                if clase == "damage":
                    etiquetas.append(1)
                else:
                    etiquetas.append(0)

            except Exception as e:
                print("error procesando:", archivo)
                continue

    print("total imagenes validas:", len(datos))

    return np.array(datos), np.array(etiquetas)


# entrenamiento
def entrenar(x, y):

    if len(x) == 0:
        print("no hay datos validos")
        return None

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    modelo = SVC(kernel="linear", probability=True)

    modelo.fit(x_train, y_train)

    pred = modelo.predict(x_test)

    print("\nreporte:\n")
    print(classification_report(y_test, pred))

    # guardar modelo
    joblib.dump(modelo, "modelo_svm.pkl")

    print("modelo guardado correctamente")

    return modelo


# main
if __name__ == "__main__":

    print("cargando datos...")
    x, y = cargar_dataset("data")

    print("entrenando modelo...")
    entrenar(x, y)

    print("proceso finalizado")