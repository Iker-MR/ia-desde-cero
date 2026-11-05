import numpy as np

# =============================================
# 1. JACCARD: ¿cuánto se parecen dos CONJUNTOS?
# =============================================
def jaccard(a, b):
    """Lo que tienen en común / todo lo que hay entre los dos."""
    return len(a & b) / len(a | b)

tu = {"acción", "comedia", "ciencia ficción"}
amigo = {"acción", "ciencia ficción", "terror", "drama"}

print("=== JACCARD: GÉNEROS FAVORITOS ===")
print(f"En común:       {sorted(tu & amigo)}  -> {len(tu & amigo)}")
print(f"Entre los dos:  {len(tu | amigo)} géneros distintos")
print(f"Jaccard = {len(tu & amigo)}/{len(tu | amigo)} = {jaccard(tu, amigo):.2f}")

# =============================================
# 2. DISTANCIA DE EDICIÓN: ¿cuántos cambios de letra?
# =============================================
def edicion(a, b):
    """Mínimo de letras a cambiar, borrar o añadir para pasar de a a b."""
    tabla = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    tabla[:, 0] = range(len(a) + 1)
    tabla[0, :] = range(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cambio = 0 if a[i - 1] == b[j - 1] else 1
            tabla[i, j] = min(tabla[i - 1, j] + 1,           # borrar
                              tabla[i, j - 1] + 1,           # añadir
                              tabla[i - 1, j - 1] + cambio)  # cambiar
    return tabla[len(a), len(b)]

print("\n=== DISTANCIA DE EDICIÓN ===")
for a, b in [("gato", "pato"), ("gato", "gatos"), ("pyhton", "python"), ("kitten", "sitting")]:
    print(f"{a:>8} -> {b:<8} {edicion(a, b)} cambios")

# Un corrector: ¿qué palabra del diccionario está más cerca?
diccionario = ["python", "pytorch", "pandas", "numpy"]
escrito = "pyton"
mejor = min(diccionario, key=lambda p: edicion(escrito, p))
print(f"\nHas escrito '{escrito}'. ¿Quisiste decir '{mejor}'?")

# =============================================
# 3. MAHALANOBIS: distancia que sabe que los datos van JUNTOS
# =============================================
rng = np.random.default_rng(0)
altura = rng.normal(170, 10, 1000)
peso = 70 + 0.9 * (altura - 170) + rng.normal(0, 5, 1000)  # más alto -> más peso
datos = np.column_stack([altura, peso])

media = datos.mean(axis=0)
cov_inv = np.linalg.inv(np.cov(datos, rowvar=False))

def mahalanobis(x):
    d = x - media
    return np.sqrt(d @ cov_inv @ d)

print("\n=== MAHALANOBIS: ALTURA Y PESO ===")
print(f"Persona media: {media[0]:.0f} cm, {media[1]:.0f} kg")
for nombre, p in [("P1: 185 cm, 85 kg", (185, 85)), ("P2: 155 cm, 85 kg", (155, 85))]:
    p = np.array(p)
    print(f"{nombre} | L2 = {np.linalg.norm(p - media):5.1f} | Mahalanobis = {mahalanobis(p):4.1f}")