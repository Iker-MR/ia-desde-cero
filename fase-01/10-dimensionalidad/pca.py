import numpy as np

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(0)

# 200 alumnos: nota de Matemáticas y de Física. Quien es bueno en una, suele serlo en la otra.
mates = rng.normal(6, 1.8, 200)
fisica = mates + rng.normal(0, 0.6, 200)
X = np.column_stack([mates, fisica])               # 200 filas x 2 columnas
print("Primeros 3 alumnos [mates, física]:\n", X[:3])

# --- PCA paso a paso ---
# 1. Centrar: restar la media de cada columna
media = X.mean(axis=0)
Xc = X - media

# 2. Covarianza: cómo varía cada columna y cómo se mueven juntas
cov = Xc.T @ Xc / (len(X) - 1)
print("\nMatriz de covarianza:\n", cov)

# 3. Autovectores y autovalores (lección 1.03): las direcciones naturales de la nube
valores, vectores = np.linalg.eigh(cov)
orden = np.argsort(valores)[::-1]                  # de mayor a menor
valores, vectores = valores[orden], vectores[:, orden]
print("\nDirección 1 (la de más dispersión):", vectores[:, 0], "| autovalor:", round(valores[0], 3))
print("Dirección 2:                       ", vectores[:, 1], "| autovalor:", round(valores[1], 3))

# 4. ¿Qué parte de la información guarda cada dirección?
explicada = valores / valores.sum()
print(f"\nVarianza explicada: dirección 1 = {explicada[0]:.1%} | dirección 2 = {explicada[1]:.1%}")

# 5. Proyectar: cada alumno pasa a ser UN número (su coordenada en la dirección 1)
resumen = Xc @ vectores[:, 0]
print("\nPrimeros 3 alumnos resumidos en 1 número:", resumen[:3])

# 6. Reconstruir desde ese único número y medir lo que se ha perdido
reconstruido = np.outer(resumen, vectores[:, 0]) + media
print("Reconstruidos [mates, física]:\n", reconstruido[:3])
error = np.mean((X - reconstruido) ** 2)
print(f"\nError medio de reconstrucción: {error:.3f}  (varianza total de los datos: {valores.sum():.3f})")