import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# preprocesamiento

def preprocesar(imagen):

    # redimensionar
    imagen = cv2.resize(imagen, (256, 256))

    # gris
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # ruido
    gris = cv2.GaussianBlur(gris, (5,5), 0)

    # contraste
    gris = cv2.equalizeHist(gris)

    return gris


def descriptor_global(gris):

    # bordes
    bordes = cv2.Canny(gris, 100, 200)

    # histograma
    hist = cv2.calcHist([bordes], [0], None, [256], [0,256])
    hist = hist.flatten()

    # normalizar
    hist = hist / np.sum(hist)

    return hist


#cargar dataset
def cargar_dataset(ruta_base):

    datos = []
    etiquetas = []

    clases = ["damage", "no_damage"]

    for clase in clases:

        ruta_clase = os.path.join(ruta_base, clase)

        for archivo in os.listdir(ruta_clase):

            ruta_imagen = os.path.join(ruta_clase, archivo)

            imagen = cv2.imread(ruta_imagen)

            if imagen is None:
                continue

            gris = preprocesar(imagen)

            caracteristicas = descriptor_global(gris)

            datos.append(caracteristicas)

            # etiqueta
            if clase == "damage":
                etiquetas.append(1)
            else:
                etiquetas.append(0)

    return np.array(datos), np.array(etiquetas)


def entrenar(x, y):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    modelo = SVC(kernel="linear")

    modelo.fit(x_train, y_train)

    pred = modelo.predict(x_test)

    print("\nreporte:\n")
    print(classification_report(y_test, pred))

    return modelo


# main
if __name__ == "__main__":

    ruta_dataset = "data"

    print("cargando datos...")
    x, y = cargar_dataset(ruta_dataset)

    print("entrenando modelo...")
    modelo = entrenar(x, y)

    print("fin")