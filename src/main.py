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


# leer imagen segura (solo PIL)
def leer_imagen_segura(ruta):

    try:
        img_pil = Image.open(ruta)

        # reducir tamaño para evitar cuelgues
        img_pil.thumbnail((512, 512))

        img = np.array(img_pil.convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    except:
        return None


# preprocesamiento
def preprocesar(imagen):

    imagen = cv2.resize(imagen, (256, 256))
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5,5), 0)
    gris = cv2.equalizeHist(gris)

    return gris


# descriptor global
def descriptor_global(gris):

    bordes = cv2.Canny(gris, 100, 200)

    hist = cv2.calcHist([bordes], [0], None, [256], [0,256])
    hist = hist.flatten()
    hist = hist / (np.sum(hist) + 1e-6)

    return hist


# validar imagen
def es_imagen_valida(nombre):
    return nombre.lower().endswith(('.jpg','.jpeg','.png'))


# leer annotations xml
def leer_xml(ruta_xml):

    try:
        tree = ET.parse(ruta_xml)
        root = tree.getroot()
        objetos = root.findall("object")
        return len(objetos)
    except:
        return 0


# cargar dataset (OPTIMIZADO)
def cargar_dataset(ruta_base, limite=300):

    datos = []
    etiquetas = []

    clases = ["damage", "no_damage"]

    for clase in clases:

        ruta_img = os.path.join(ruta_base, clase, "images")
        ruta_xml = os.path.join(ruta_base, clase, "annotations")

        print("\nleyendo carpeta:", ruta_img)

        archivos = os.listdir(ruta_img)

        for i, archivo in enumerate(archivos[:limite]):

            # progreso
            if i % 50 == 0:
                print(f"procesadas {i} imagenes...")

            if not es_imagen_valida(archivo):
                continue

            ruta_imagen = os.path.join(ruta_img, archivo)

            try:
                # evitar imágenes muy pesadas
                tamano = os.path.getsize(ruta_imagen)
                if tamano > 5 * 1024 * 1024:
                    continue

                imagen = leer_imagen_segura(ruta_imagen)

                if imagen is None:
                    continue

                if imagen.shape[0] < 50 or imagen.shape[1] < 50:
                    continue

                gris = preprocesar(imagen)
                desc = descriptor_global(gris)

                # leer xml
                nombre_xml = archivo.replace(".jpg", ".xml").replace(".png", ".xml")
                ruta_anot = os.path.join(ruta_xml, nombre_xml)

                num_obj = leer_xml(ruta_anot)

                # agregar feature extra
                desc = np.append(desc, num_obj)

                datos.append(desc)

                etiquetas.append(1 if clase == "damage" else 0)

            except:
                continue

    print("\ntotal imagenes validas:", len(datos))

    return np.array(datos), np.array(etiquetas)


# entrenar modelo
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

    joblib.dump(modelo, "modelo_svm.pkl")

    print("modelo guardado correctamente")

    return modelo


# main
if __name__ == "__main__":

    print("cargando datos...")

    # puedes subir este numero despues
    x, y = cargar_dataset("data", limite=300)

    print("\nentrenando modelo...")
    entrenar(x, y)

    print("\nproceso finalizado")