import numpy as np

rng = np.random.default_rng(0)

# --- 1. De un número a un tensor de 3 dimensiones ---
nota = np.array(7)                                   # 0 dimensiones: un número
alumno = np.array([7, 5])                            # 1 dimensión: [mates, física]
clase = rng.integers(3, 11, size=(4, 2))             # 2 dimensiones: 4 alumnos x 2 asignaturas
colegio = rng.integers(3, 11, size=(3, 4, 2))        # 3 dimensiones: 3 clases x 4 alumnos x 2 asignaturas

for nombre, t in [("nota", nota), ("alumno", alumno), ("clase", clase), ("colegio", colegio)]:
    print(f"{nombre:8} dimensiones = {t.ndim}   forma = {str(t.shape):10} números = {t.size}")
print("\nEl colegio entero (3 clases):\n", colegio)

# --- 2. Reducir un eje: hacer la media "a lo largo" de una dimensión ---
print("\nMedia sobre el eje 0 (juntar las 3 clases)               -> forma", colegio.mean(axis=0).shape)
print("Media sobre el eje 1 (juntar los alumnos de cada clase) -> forma", colegio.mean(axis=1).shape)
print(np.round(colegio.mean(axis=1), 2), "  <- media de mates y de física en cada clase")
print("Media sobre el eje 2 (juntar las asignaturas)            -> forma", colegio.mean(axis=2).shape)
print(np.round(colegio.mean(axis=2), 2), "  <- nota media de cada alumno")

# --- 3. Cambiar de forma sin cambiar los números ---
fila = np.arange(12)
print("\n12 números:", fila)
print("reshape(3, 4):\n", fila.reshape(3, 4))
print("reshape(2, -1)  (el -1 se calcula solo):", fila.reshape(2, -1).shape)
try:
    fila.reshape(5, 3)
except ValueError as e:
    print("reshape(5, 3): ERROR ->", e)

# --- 4. Añadir o quitar dimensiones de tamaño 1 ---
v = np.array([1, 2, 3])
print("\nv:", v.shape, "| v[None, :]:", v[None, :].shape, "| v[:, None]:", v[:, None].shape)
print("np.squeeze de la forma (1, 3, 1):", np.squeeze(np.zeros((1, 3, 1))).shape)

# --- 5. Las formas típicas en deep learning ---
fotos = np.zeros((32, 3, 224, 224))     # 32 fotos, 3 colores (RGB), 224 x 224 píxeles
frases = np.zeros((16, 128, 768))       # 16 frases, 128 tokens, 768 números por token
def con_puntos(n):
    return f"{n:,}".replace(",", ".")

print(f"\nLote de fotos  {fotos.shape}: {con_puntos(fotos.size)} números")
print(f"Lote de frases {frases.shape}: {con_puntos(frases.size)} números")
print("Embedding de cada frase (media de sus tokens):", frases.mean(axis=1).shape)