import math
import random

random.seed(3)


# --- 1. Un punto de silla ---
# Hacia los lados (x) es un valle, pero en la otra dirección (y) es una cima.
# (0, 0) es el punto de silla. Los fondos de verdad están en y = 1 e y = -1.
def perdida(x, y):
    return x ** 2 + y ** 4 / 4 - y ** 2 / 2

def gradiente(x, y):
    return 2 * x, y ** 3 - y


def bajar(ruido, pasos=300, tasa=0.1):
    x, y = 1.0, 0.0                       # salimos justo encima de la "cresta"
    for _ in range(pasos):
        gx, gy = gradiente(x, y)
        gx += random.gauss(0, ruido)      # ruido = usar un lote pequeño de datos, no todos
        gy += random.gauss(0, ruido)
        x, y = x - tasa * gx, y - tasa * gy
    return x, y


print("Punto de silla (el fondo de verdad tiene pérdida -0,25):")
for nombre, ruido in [("Sin ruido (todos los datos)", 0.0), ("Con ruido (lotes pequeños)", 0.01)]:
    x, y = bajar(ruido)
    gx, gy = gradiente(x, y)
    print(f"  {nombre:28} acaba en x = {x:5.2f}, y = {y:5.2f} | pendiente = {math.hypot(gx, gy):.3f}"
          f" | pérdida = {perdida(x, y):6.3f}")


# --- 2. Calentamiento + coseno: la tasa que cambia durante el entrenamiento ---
def tasa_en(paso, total=1000, calentamiento=100, tasa_max=0.001):
    if paso < calentamiento:
        return tasa_max * paso / calentamiento                     # sube poco a poco
    avance = (paso - calentamiento) / (total - calentamiento)       # de 0 a 1
    return tasa_max * 0.5 * (1 + math.cos(math.pi * avance))        # baja suave hasta 0


print("\nTasa de aprendizaje con calentamiento + coseno (máximo 0,001):")
for paso in [0, 50, 100, 300, 550, 800, 1000]:
    t = tasa_en(paso)
    print(f"  paso {paso:4}: {t:.5f}  {'█' * round(t / 0.001 * 40)}")