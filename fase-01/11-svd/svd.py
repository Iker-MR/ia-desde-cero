import numpy as np

np.set_printoptions(precision=2, suppress=True)

peliculas = ["Acción 1", "Acción 2", "Acción 3", "Romántica 1", "Romántica 2"]
usuarios = ["Ana", "Luis", "Marta", "Pablo", "Sara", "Iker"]

# Notas del 1 al 5. Por detrás hay solo 2 "gustos": acción y romance.
R = np.array([
    [5, 5, 4, 1, 1],    # Ana:   mucha acción
    [4, 5, 5, 1, 2],    # Luis:  mucha acción
    [1, 1, 2, 5, 5],    # Marta: muy romántica
    [2, 1, 1, 4, 5],    # Pablo: muy romántico
    [4, 4, 4, 4, 4],    # Sara:  le gusta todo
    [3, 3, 2, 3, 3],    # Iker:  un poco de todo, sin entusiasmo
], dtype=float)
print("Notas (filas = usuarios, columnas = películas):\n", R)

# --- SVD: R = U · S · Vt ---
U, S, Vt = np.linalg.svd(R, full_matrices=False)
print("\nValores singulares (la importancia de cada 'capa'):", S)

# --- Reconstruir con las k capas más importantes ---
for k in [1, 2]:
    Rk = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    error = np.abs(R - Rk).mean()
    print(f"\nCon {k} capa(s): error medio = {error:.2f} puntos")
    print(Rk)

# --- ¿Qué ha descubierto? Los 2 gustos ocultos ---
print("\nCapa 2 vista desde las películas (signo + o −):")
for p, v in zip(peliculas, Vt[1]):
    print(f"  {p:12} {v:+.2f}")
print("\nCapa 2 vista desde los usuarios:")
for u, v in zip(usuarios, U[:, 1]):
    print(f"  {u:6} {v:+.2f}")