import math


def entropia(p):
    return sum(-pi * math.log2(pi) for pi in p if pi > 0)


def entropia_cruzada(p, q):
    """Sorpresa media si la realidad es p pero tú esperas q."""
    return sum(-pi * math.log2(qi) for pi, qi in zip(p, q) if pi > 0)


def kl(p, q):
    """La sorpresa EXTRA por esperar q en vez de p."""
    return entropia_cruzada(p, q) - entropia(p)


# --- 1. Agosto en Sevilla: sol, nubes, lluvia ---
realidad = [0.97, 0.02, 0.01]
print(f"Entropía del tiempo en Sevilla (el suelo): {entropia(realidad):.2f} bits\n")
print(f"{'':20} {'esperaba':22} {'entropía cruzada':>17} {'KL (sorpresa extra)':>21}")
for nombre, espera in [("Sevillano", [0.97, 0.02, 0.01]),
                       ("Turista prudente", [0.80, 0.15, 0.05]),
                       ("Turista de Londres", [0.30, 0.30, 0.40])]:
    texto = " / ".join(f"{x:.0%}" for x in espera)
    print(f"{nombre:20} {texto:22} {entropia_cruzada(realidad, espera):12.2f} bits"
          f" {kl(realidad, espera):16.2f} bits")

# --- 2. Un clasificador: la realidad es 'gato' seguro (100 %) ---
print("\nLa foto es un gato. Realidad = [gato 100 %, perro 0 %, pájaro 0 %]")
gato = [1.0, 0.0, 0.0]
for prediccion in [[0.25, 0.5, 0.25], [0.66, 0.24, 0.10], [0.99, 0.005, 0.005]]:
    print(f"  modelo dice gato {prediccion[0]:.0%}:  entropía cruzada = {entropia_cruzada(gato, prediccion):.2f} bits"
          f"  =  -log2({prediccion[0]})")

# --- 3. Entrenar = acercar lo que espera el modelo a la realidad ---
print("\nEl turista de Londres se va acostumbrando, día a día:")
espera = [0.30, 0.30, 0.40]
for dia in range(0, 31, 5):
    print(f"  día {dia:2}: espera sol {espera[0]:.0%}  ->  KL = {kl(realidad, espera):.3f} bits")
    for _ in range(5):
        espera = [0.8 * e + 0.2 * r for e, r in zip(espera, realidad)]   # se acerca un 20 % cada día