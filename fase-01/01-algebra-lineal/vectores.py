import random


class Vector:
    def __init__(self, componentes):
        self.c = list(componentes)

    def __add__(self, otro):                      # a + b
        return Vector([x + y for x, y in zip(self.c, otro.c)])

    def dot(self, otro):                          # producto escalar
        return sum(x * y for x, y in zip(self.c, otro.c))

    def magnitud(self):                           # Pitágoras
        return sum(x ** 2 for x in self.c) ** 0.5

    def coseno(self, otro):                       # similitud coseno
        return self.dot(otro) / (self.magnitud() * otro.magnitud())

    def __repr__(self):
        return f"Vector({[round(x, 3) for x in self.c]})"


class Matriz:
    def __init__(self, filas):
        self.filas = [list(f) for f in filas]

    def __matmul__(self, v):                      # M @ v
        return Vector([sum(m * x for m, x in zip(fila, v.c)) for fila in self.filas])


# --- 1. Operaciones básicas ---
a = Vector([1, 2, 3])
b = Vector([4, 5, 6])
print("a + b =", a + b)
print("a · b =", a.dot(b))
print(f"|a| = {a.magnitud():.4f}")
print(f"coseno(a, b) = {a.coseno(b):.4f}")

# --- 2. Una matriz que rota 90 grados ---
rotar_90 = Matriz([[0, -1], [1, 0]])
print("\n[3, 1] rotado 90° =", rotar_90 @ Vector([3, 1]))

# --- 3. "Embeddings" de juguete: [realeza, persona, fruta] ---
rey = Vector([0.90, 0.80, 0.10])
reina = Vector([0.88, 0.85, 0.05])
manzana = Vector([0.05, 0.10, 0.95])
print(f"\ncoseno(rey, reina)   = {rey.coseno(reina):.3f}")
print(f"coseno(rey, manzana) = {rey.coseno(manzana):.3f}")

# --- 4. Una capa de red neuronal: 3 entradas -> 2 salidas ---
random.seed(42)
pesos = Matriz([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
entrada = Vector([1.0, 0.5, -0.3])
print("\nEntrada (3D):", entrada)
print("Salida  (2D):", pesos @ entrada)