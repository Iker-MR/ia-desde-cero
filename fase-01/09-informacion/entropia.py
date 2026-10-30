import math


def sorpresa(p):
    """Bits de sorpresa: cuántas preguntas de sí o no 'vale' que ocurra algo de probabilidad p."""
    return -math.log2(p) + 0.0          # el + 0.0 evita que salga "-0.00"


def entropia(probs):
    """Sorpresa media: cada resultado pesa según lo probable que es."""
    return sum(p * sorpresa(p) for p in probs if p > 0)


# --- 1. Sorpresa de distintos sucesos ---
print("Sorpresa (bits):")
for nombre, p in [("Seguro que pasa", 1.0), ("Cara en una moneda", 1/2), ("Un 6 en un dado", 1/6),
                  ("Número del 1 al 16", 1/16), ("1 entre 1.000", 1/1000)]:
    print(f"  {nombre:20} P = {p:<7.4f} -> {sorpresa(p):5.2f} bits")

# --- 2. Entropía: cuánta incertidumbre hay ---
print("\nEntropía (sorpresa media, en bits):")
casos = [
    ("Moneda normal", [0.5, 0.5]),
    ("Moneda trucada 99 %", [0.99, 0.01]),
    ("Dado normal", [1/6] * 6),
    ("Dado trucado (6 al 90 %)", [0.02] * 5 + [0.9]),
    ("Sevilla en agosto", [0.97, 0.02, 0.01]),       # sol, nubes, lluvia
]
for nombre, probs in casos:
    print(f"  {nombre:26} {entropia(probs):5.2f} bits")

# --- 3. ¿Cuál es la entropía máxima? Cuando todo es igual de probable ---
print("\nUn dado de 6 caras, cada vez más trucado hacia el 6:")
for p6 in [1/6, 0.3, 0.5, 0.7, 0.9, 1.0]:
    resto = (1 - p6) / 5
    print(f"  P(6) = {p6:.2f}  ->  entropía = {entropia([resto] * 5 + [p6]):.2f} bits")