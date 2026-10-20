import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# --- 1. Elemento a elemento frente a matricial ---
print("A * B =\n", A * B)
print("A @ B =\n", A @ B)

# --- 2. Determinantes ---
matrices = {
    "escalar x2, y3": np.array([[2, 0], [0, 3]]),
    "rotar 90°": np.array([[0, -1], [1, 0]]),
    "A": A,
    "aplastar": np.array([[1, 2], [2, 4]]),
}
print()
for nombre, M in matrices.items():
    print(f"det({nombre}) = {np.linalg.det(M):.1f}")

# --- 3. Inversa: deshacer ---
A_inv = np.linalg.inv(A)
print("\nA⁻¹ =\n", A_inv)
print("A @ A⁻¹ =\n", (A @ A_inv).round(6))

x = np.array([3, 1])
y = A @ x
print("x =", x, "→ A @ x =", y, "→ A⁻¹ @ y =", A_inv @ y)

try:
    np.linalg.inv(np.array([[1, 2], [2, 4]]))
except np.linalg.LinAlgError as e:
    print("Invertir 'aplastar':", e)

# --- 4. Broadcasting ---
salidas = np.array([[1, 2, 3], [4, 5, 6]])
sesgo = np.array([10, 20, 30])
print("\nsalidas + sesgo =\n", salidas + sesgo)


# --- 5. Red de dos capas: 3 entradas -> 4 ocultas -> 2 salidas ---
def relu(z):
    return np.maximum(0, z)


rng = np.random.default_rng(42)
X = rng.normal(size=(5, 3))                  # lote: 5 ejemplos de 3 características
W1, b1 = rng.normal(size=(3, 4)) * 0.5, np.zeros(4)
W2, b2 = rng.normal(size=(4, 2)) * 0.5, np.zeros(2)

H = relu(X @ W1 + b1)                        # capa oculta
Y = H @ W2 + b2                              # capa de salida

print(f"\nX {X.shape} @ W1 {W1.shape} -> H {H.shape}")
print(f"H {H.shape} @ W2 {W2.shape} -> Y {Y.shape}")
print("Parámetros totales:", W1.size + b1.size + W2.size + b2.size)
print("¿Algún cero en H por la ReLU?", (H == 0).any())