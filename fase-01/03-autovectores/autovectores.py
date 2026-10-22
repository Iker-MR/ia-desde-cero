import numpy as np

np.set_printoptions(precision=3, suppress=True)

# --- 1. Corte y reflexión sobre el cuadrado unidad ---
cuadrado = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T     # cada columna, una esquina
corte = np.array([[1, 1], [0, 1]])
espejo = np.array([[-1, 0], [0, 1]])
print("Esquinas tras el corte:\n", (corte @ cuadrado).T)
print("det(corte) =", round(np.linalg.det(corte), 3), "| det(espejo) =", round(np.linalg.det(espejo), 3))

# --- 2. Autovalores y autovectores ---
A = np.array([[2, 1], [1, 2]], dtype=float)
valores, vectores = np.linalg.eigh(A)    # eigh: para matrices simétricas
print("\nAutovalores de A:", valores)
for i in range(2):
    v = vectores[:, i]
    print(f"  v = {v}  ->  A @ v = {A @ v}  =  {valores[i]:.0f} × v")

x = np.array([1.0, 0.0])
print("Un vector cualquiera: [1, 0] -> A @ [1, 0] =", A @ x, "(cambia de dirección)")

# --- 3. ¿Tiene autovectores una rotación de 90°? ---
R = np.array([[0, -1], [1, 0]], dtype=float)
print("\nAutovalores de la rotación de 90°:", np.linalg.eig(R)[0])

# --- 4. Multiplicar muchas veces: explotar o desvanecerse ---
for nombre, M in [("A", A), ("0.3 × A", 0.3 * A)]:
    v = np.array([1.0, 0.0])
    print(f"\nAplicar {nombre} (autovalores {np.linalg.eigvalsh(M)}) una y otra vez a [1, 0]:")
    for paso in range(1, 11):
        v = M @ v
        if paso in (1, 2, 5, 10):
            print(f"  paso {paso:2}: longitud = {np.linalg.norm(v):12.4f} | dirección = {v / np.linalg.norm(v)}")

# --- 5. Descomposición: A = V @ D @ V⁻¹ ---
V, D = vectores, np.diag(valores)
print("\nV @ D @ V⁻¹ =\n", V @ D @ np.linalg.inv(V))