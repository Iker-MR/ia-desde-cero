import numpy as np

# ---------------------------------------------
# 1. Tres formas de medir la distancia
# ---------------------------------------------
def l1(a, b):
    """Taxi: solo puede ir por calles (sumar cada lado)."""
    return np.sum(np.abs(a - b))

def l2(a, b):
    """Paloma: línea recta (Pitágoras)."""
    return np.sqrt(np.sum((a - b) ** 2))

def linf(a, b):
    """Rey de ajedrez: manda el lado más largo."""
    return np.max(np.abs(a - b))

A = np.array([1, 1])
B = np.array([4, 5])

print("=== DE A=(1,1) A B=(4,5) ===")
print(f"L1   (taxi):    {l1(A, B):.2f}")
print(f"L2   (paloma):  {l2(A, B):.2f}")
print(f"Linf (rey):     {linf(A, B):.2f}")

# Lo mismo con numpy, que es lo que usarás en la práctica
print("\nCon np.linalg.norm:")
d = B - A
print(f"ord=1: {np.linalg.norm(d, ord=1):.2f} | ord=2: {np.linalg.norm(d):.2f} | ord=inf: {np.linalg.norm(d, ord=np.inf):.2f}")

# ---------------------------------------------
# 2. ¿Siempre Linf <= L2 <= L1? Probamos 10.000 parejas al azar
# ---------------------------------------------
rng = np.random.default_rng(0)
cumple = 0
for _ in range(10_000):
    a, b = rng.normal(size=5), rng.normal(size=5)
    if linf(a, b) <= l2(a, b) <= l1(a, b):
        cumple += 1
print(f"\n=== ORDEN Linf <= L2 <= L1 ===\nSe cumple en {cumple} de 10000 parejas")

# ---------------------------------------------
# 3. ¿Qué puntos están "a distancia 1 o menos" del origen?
# ---------------------------------------------
print("\n=== ¿DENTRO DEL CÍRCULO DE RADIO 1? ===")
origen = np.array([0.0, 0.0])
for p in [(0.6, 0.6), (1.0, 0.0), (0.9, 0.9)]:
    p = np.array(p)
    fila = []
    for nombre, f in [("L1", l1), ("L2", l2), ("Linf", linf)]:
        dist = f(p, origen)
        fila.append(f"{nombre}={dist:.2f} {'dentro' if dist <= 1 else 'fuera '}")
    print(f"({p[0]}, {p[1]}): " + " | ".join(fila))

# ---------------------------------------------
# 4. Del mismo modo se miden los errores: MAE (L1) y MSE (L2)
# ---------------------------------------------
print("\n=== ERRORES DE UN MODELO: MAE vs MSE ===")
errores = np.array([1, 1, 1, 1, 10])   # un error muy grande (un outlier)
mae = np.mean(np.abs(errores))
mse = np.mean(errores ** 2)
print(f"Errores: {errores}")
print(f"MAE (L1): {mae:.1f}  -> el error grande aporta {10 / np.sum(np.abs(errores)):.0%}")
print(f"MSE (L2): {mse:.1f} -> el error grande aporta {100 / np.sum(errores ** 2):.0%}")