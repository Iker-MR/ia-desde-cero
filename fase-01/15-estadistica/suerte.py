import numpy as np

rng = np.random.default_rng(42)

def pearson(x, y):
    return np.corrcoef(x, y)[0, 1]

def spearman(x, y):
    """Pearson sobre los PUESTOS (1.º, 2.º, 3.º...) en vez de los valores."""
    puesto_x = np.argsort(np.argsort(x))
    puesto_y = np.argsort(np.argsort(y))
    return pearson(puesto_x, puesto_y)

# =============================================
# 1. CORRELACIÓN: ¿suben juntos?
# =============================================
print("=== CORRELACIÓN ===")
horas = np.array([1, 2, 3, 4, 5, 6, 7, 8])
nota = np.array([3, 4, 4, 5, 6, 7, 8, 9])
print(f"Horas de estudio vs nota:    Pearson = {pearson(horas, nota):+.2f}")
print(f"Horas de móvil vs nota:      Pearson = {pearson(horas, nota[::-1]):+.2f}")

x = rng.normal(0, 1, 20)        # 20 puntos al azar
y = rng.normal(0, 1, 20)        # sin ninguna relación
print(f"\n20 puntos al azar:           Pearson = {pearson(x, y):+.2f} | Spearman = {spearman(x, y):+.2f}")
x2, y2 = np.append(x, 100), np.append(y, 100)   # UN punto extremo
print(f"+ un solo punto en (100,100): Pearson = {pearson(x2, y2):+.2f} | Spearman = {spearman(x2, y2):+.2f}")

# =============================================
# 2. P-VALOR: ¿la moneda está trucada?
# =============================================
print("\n=== ¿MONEDA TRUCADA? (100.000 simulaciones con una moneda NORMAL) ===")
for tiradas, caras in [(10, 7), (100, 70)]:
    simuladas = rng.binomial(tiradas, 0.5, 100_000)   # monedas normales
    p = np.mean(simuladas >= caras)
    print(f"{caras} caras de {tiradas}: p-valor = {p:.5f} -> {'¡difícil que sea suerte!' if p < 0.05 else 'puede ser pura suerte'}")

# =============================================
# 3. DOS MODELOS: 0,87 vs 0,89, ¿mejora real o suerte?
# =============================================
print("\n=== DOS MODELOS IGUAL DE BUENOS (0,88 de verdad) ===")
for n in [200, 10_000]:
    a = rng.binomial(n, 0.88, 100_000) / n    # modelo A en 100.000 test distintos
    b = rng.binomial(n, 0.88, 100_000) / n    # modelo B, igual de bueno
    veces = np.mean(np.abs(b - a) >= 0.02)
    print(f"Test de {n:>6} ejemplos: diferencia de 0,02 o más por pura suerte el {veces:.1%} de las veces")