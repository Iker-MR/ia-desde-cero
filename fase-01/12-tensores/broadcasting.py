import numpy as np

np.set_printoptions(precision=2, suppress=True)

# --- 1. Broadcasting: operar con formas distintas ---
notas = np.array([[6, 7, 5],
                  [8, 4, 9],
                  [5, 6, 7],
                  [7, 8, 6]], dtype=float)           # (4 alumnos, 3 asignaturas)
extra = np.array([1, 0, 0.5])                        # (3,): un punto extra por asignatura
print("notas (4, 3) + extra (3,) ->", (notas + extra).shape)
print(notas + extra)

try:
    notas + np.array([1, 2, 3, 4])                    # (4,): uno por alumno... ¿encaja?
except ValueError as e:
    print("\nnotas (4, 3) + (4,) -> ERROR:", e)

por_alumno = np.array([1, 2, 3, 4])[:, None]          # (4, 1): ahora sí, una columna
print("\nnotas (4, 3) + (4, 1) ->", (notas + por_alumno).shape)
print(notas + por_alumno)

# --- 2. El fallo silencioso más famoso ---
real = np.array([3.0, 5.0, 7.0])                      # (3,)
prediccion = np.array([[2.9], [5.2], [6.8]])          # (3, 1): así salen de muchas redes
resta = prediccion - real
print("\nprediccion (3, 1) - real (3,) ->", resta.shape, " ¡sin ningún error!")
print("Error cuadrático medio CORRECTO:", np.mean((prediccion[:, 0] - real) ** 2).round(4))
print("Error cuadrático medio con el FALLO:", np.mean(resta ** 2).round(4))

# --- 3. einsum: decir con letras qué ejes se multiplican y cuáles se suman ---
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
print("\nproducto escalar  'i,i->'      :", np.einsum("i,i->", a, b), "| con @:", a @ b)

A = np.arange(6, dtype=float).reshape(2, 3)            # (2, 3)
B = np.arange(12, dtype=float).reshape(3, 4)           # (3, 4)
print("matriz x matriz   'ik,kj->ij'  :", np.einsum("ik,kj->ij", A, B).shape,
      "| igual que A @ B:", np.allclose(np.einsum("ik,kj->ij", A, B), A @ B))

lote_A = np.random.default_rng(0).normal(size=(8, 2, 3))   # 8 pares de matrices a la vez
lote_B = np.random.default_rng(1).normal(size=(8, 3, 4))
print("lote de matrices  'bik,bkj->bij':", np.einsum("bik,bkj->bij", lote_A, lote_B).shape)
print("suma de cada fila 'ij->i'      :", np.einsum("ij->i", notas), "| con sum(axis=1):", notas.sum(axis=1))
