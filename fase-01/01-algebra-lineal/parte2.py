import numpy as np

# --- 1. Independencia y rango ---
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = 2 * v1 + v2
print("Rango de {v1, v2, v3}:", np.linalg.matrix_rank(np.stack([v1, v2, v3])))

# Dataset de pisos: [m2, precio en euros, precio en dólares]
pisos = np.array([
    [ 60, 180_000, 194_400],
    [ 85, 240_000, 259_200],
    [120, 310_000, 334_800],
    [ 45, 150_000, 162_000],
])
print("Pisos: columnas =", pisos.shape[1], "| rango =", np.linalg.matrix_rank(pisos))


# --- 2. Proyección ---
def proyectar(a, b):
    return (a @ b) / (b @ b) * b


a = np.array([3.0, 4.0])
b = np.array([1.0, 0.0])
p = proyectar(a, b)
residuo = a - p
print("\nProyección de a sobre b:", p)
print("Residuo:", residuo, "| residuo · b =", residuo @ b)

x = np.array([1.0, 2.0, 3.0])
print("Proyección de [1,2,3] sobre [1,1,1]:", proyectar(x, np.ones(3)))


# --- 3. Gram-Schmidt ---
def gram_schmidt(vectores):
    base = []
    for v in vectores:
        w = v.astype(float)
        for u in base:
            w = w - (w @ u) * u          # quitar la sombra sobre cada eje anterior
        if np.linalg.norm(w) > 1e-10:    # si queda algo, es una dirección nueva
            base.append(w / np.linalg.norm(w))
    return np.array(base)


Q = gram_schmidt([np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 1])])
print("\nBase ortonormal:\n", Q.round(3))
print("¿Perpendiculares y de longitud 1?", np.allclose(Q @ Q.T, np.eye(3)))

Q2 = gram_schmidt([v1, v2, v3])
print("Con {v1, v2, v3}: Gram-Schmidt devuelve", len(Q2), "vectores")