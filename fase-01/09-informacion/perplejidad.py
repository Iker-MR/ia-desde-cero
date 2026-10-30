import math
from collections import Counter


# --- 1. Perplejidad: ¿entre cuántas palabras "duda" el modelo? ---
def perplejidad(probs_palabra_correcta):
    entropia_cruzada = sum(-math.log2(p) for p in probs_palabra_correcta) / len(probs_palabra_correcta)
    return entropia_cruzada, 2 ** entropia_cruzada


# Probabilidad que cada modelo dio a la palabra que DE VERDAD venía, en una frase de 6 palabras
modelos = {
    "Al azar (10.000 palabras)": [1 / 10_000] * 6,
    "Modelo flojo": [0.05, 0.10, 0.02, 0.08, 0.04, 0.10],
    "Modelo bueno": [0.40, 0.60, 0.30, 0.50, 0.70, 0.45],
    "Modelo perfecto": [1.0] * 6,
}
print("Perplejidad (menos es mejor):")
for nombre, probs in modelos.items():
    ec, pp = perplejidad(probs)
    print(f"  {nombre:26} entropía cruzada = {ec:5.2f} bits  ->  perplejidad = {pp:8.1f}")


# --- 2. Información mutua: ¿cuánto me ayuda saber X para adivinar Y? ---
def entropia(contador):
    total = sum(contador.values())
    return sum(-(n / total) * math.log2(n / total) for n in contador.values())


def info_mutua(pares):
    xs = Counter(x for x, _ in pares)
    ys = Counter(y for _, y in pares)
    juntos = Counter(pares)
    return entropia(xs) + entropia(ys) - entropia(juntos)


def correlacion(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


# 100 días: ¿la gente lleva paraguas?
tiempo_paraguas = ([("lluvia", "paraguas")] * 45 + [("lluvia", "no")] * 5 +
                   [("sol", "paraguas")] * 10 + [("sol", "no")] * 40)
zapatos_paraguas = ([("rojos", "paraguas")] * 28 + [("rojos", "no")] * 22 +
                    [("negros", "paraguas")] * 27 + [("negros", "no")] * 23)

print("\nInformación mutua con 'lleva paraguas' (bits):")
print(f"  saber si llueve:            {info_mutua(tiempo_paraguas):.3f}")
print(f"  saber el color de zapatos:  {info_mutua(zapatos_paraguas):.3f}")

# Una relación que la correlación NO ve: y = x²
x = [-2, -1, 0, 1, 2] * 20
y = [v ** 2 for v in x]
print("\ny = x² (si sabes x, sabes y exactamente):")
print(f"  correlación:        {correlacion(x, y):.3f}")
print(f"  información mutua:  {info_mutua(list(zip(x, y))):.3f} bits")