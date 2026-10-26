import random

random.seed(0)

def dado():
    return random.randint(1, 6)

# --- 1. Más tiradas = frecuencia más cerca de la probabilidad ---
print("P(sacar más de 4) — teoría: 2/6 = 0,3333")
for n in [10, 1_000, 100_000]:
    exitos = sum(1 for _ in range(n) if dado() > 4)
    print(f"  {n:>7} tiradas: {exitos / n:.4f}")

# --- 2. Probabilidad condicionada: me quedo solo con las tiradas impares ---
tiradas = [dado() for _ in range(100_000)]
impares = [t for t in tiradas if t % 2 == 1]          # "sabiendo que ha salido impar"
p_5_si_impar = sum(1 for t in impares if t == 5) / len(impares)
print(f"\nP(5 | impar) — teoría: 1/3 = 0,3333  |  simulado: {p_5_si_impar:.4f}")

# --- 3. Tres monedas independientes ---
n = 100_000
tres_caras = sum(1 for _ in range(n)
                 if random.random() < 0.5 and random.random() < 0.5 and random.random() < 0.5)
print(f"P(tres caras) — teoría: 0,5 · 0,5 · 0,5 = 0,125  |  simulado: {tres_caras / n:.4f}")

# --- 4. Cartas sin devolver: la segunda extracción depende de la primera ---
baraja = ["as"] * 4 + ["otra"] * 36
casos_primer_as = 0
casos_dos_ases = 0
for _ in range(400_000):
    mano = random.sample(baraja, 2)        # saca 2 cartas SIN devolver
    if mano[0] == "as":
        casos_primer_as += 1
        if mano[1] == "as":
            casos_dos_ases += 1
print(f"P(2º as | 1º as) — teoría: 3/39 = {3/39:.4f}  |  simulado: {casos_dos_ases / casos_primer_as:.4f}")
print(f"P(as) sin saber nada — 4/40 = {4/40:.4f}  ->  no son independientes")