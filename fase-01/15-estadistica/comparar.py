import math
import numpy as np

rng = np.random.default_rng(7)

# =============================================
# 1. BOOTSTRAP: ¿entre qué valores está de verdad mi 0,88?
# =============================================
aciertos = np.array([1] * 176 + [0] * 24)    # 176 aciertos de 200 = 0,88
print("=== BOOTSTRAP: MI MODELO SACA 0,88 CON 200 EJEMPLOS ===")

medias = []
for _ in range(10_000):
    # sacar 200 ejemplos CON REPETICIÓN (algunos salen dos veces, otros ninguna)
    muestra = rng.choice(aciertos, size=len(aciertos), replace=True)
    medias.append(muestra.mean())
bajo, alto = np.percentile(medias, [2.5, 97.5])
print(f"Precisión medida: {aciertos.mean():.3f}")
print(f"Intervalo del 95 %: entre {bajo:.3f} y {alto:.3f}")

# Dos modelos sobre los MISMOS 200 ejemplos
# (168 los aciertan los dos, 6 solo A, 10 solo B, 16 ninguno)
a = np.array([1] * 168 + [1] * 6 + [0] * 10 + [0] * 16)
b = np.array([1] * 168 + [0] * 6 + [1] * 10 + [0] * 16)
difs = []
for _ in range(10_000):
    idx = rng.integers(0, 200, 200)          # mismos ejemplos para los dos
    difs.append(b[idx].mean() - a[idx].mean())
bajo, alto = np.percentile(difs, [2.5, 97.5])
print(f"\nModelo A: {a.mean():.3f} | Modelo B: {b.mean():.3f} | B - A = {b.mean() - a.mean():+.3f}")
print(f"Intervalo del 95 % de B - A: entre {bajo:+.3f} y {alto:+.3f}")
print("¿Incluye el 0? ->", "SÍ: puede ser suerte" if bajo <= 0 <= alto else "NO: la mejora es real")

# =============================================
# 2. REAL NO SIGNIFICA QUE IMPORTE
# =============================================
def p_valor(p1, p2, n):
    """¿Puede ser suerte una diferencia entre dos precisiones medidas con n ejemplos?"""
    p = (p1 + p2) / 2
    error = math.sqrt(2 * p * (1 - p) / n)
    z = abs(p2 - p1) / error
    return math.erfc(z / math.sqrt(2))

print("\n=== 0,923 vs 0,924 (solo 0,001 de diferencia) ===")
for n in [1_000, 10_000_000]:
    p = p_valor(0.923, 0.924, n)
    print(f"Con {n:>10} ejemplos: p-valor = {p:.4f} -> {'real' if p < 0.05 else 'puede ser suerte'}")

# =============================================
# 3. PROBAR MUCHAS COSAS: los falsos positivos
# =============================================
print("\n=== PROBAR 20 TRUCOS QUE NO SIRVEN PARA NADA ===")
experimentos = 10_000
p_valores = rng.uniform(0, 1, (experimentos, 20))   # si no sirven, el p-valor es puro azar
alguno = np.mean((p_valores < 0.05).any(axis=1))
print(f"Umbral 0,05:   al menos uno 'funciona' por suerte el {alguno:.0%} de las veces")
print(f"Fórmula:       1 - 0,95^20 = {1 - 0.95 ** 20:.0%}")
alguno_b = np.mean((p_valores < 0.05 / 20).any(axis=1))
print(f"Bonferroni (umbral 0,05 / 20 = 0,0025): {alguno_b:.0%}")