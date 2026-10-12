import numpy as np


def cargar_datos():
    return [float(i) for i in range(3_000_000)]


def normalizar(datos):
    media = sum(datos) / len(datos)
    return [d - media for d in datos]


def entrenar(datos):
    arr = np.array(datos)
    for _ in range(20):
        arr = arr * 0.99 + 0.01
    return arr


def main():
    datos = cargar_datos()
    datos = normalizar(datos)
    entrenar(datos)


main()