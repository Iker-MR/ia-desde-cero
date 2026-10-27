import random

random.seed(7)


class Libreta:
    """Lo que creemos sobre una tasa (de caras, de clics...), guardado como dos contadores."""

    def __init__(self, exitos=1, fallos=1):        # empezamos con 1 y 1 imaginarios: "no sé nada"
        self.exitos = exitos
        self.fallos = fallos

    def actualizar(self, nuevos_exitos, nuevos_fallos):
        self.exitos += nuevos_exitos
        self.fallos += nuevos_fallos

    def media(self):
        return self.exitos / (self.exitos + self.fallos)

    def muestra(self):
        # Un valor posible de la tasa real, según lo que creemos (distribución Beta)
        return random.betavariate(self.exitos, self.fallos)


# --- 1. Aprender día a día: ¿la moneda está trucada? ---
moneda = Libreta()
print(f"Día 1, sin datos:            libreta {moneda.exitos}/{moneda.fallos}  ->  P(cara) ≈ {moneda.media():.2f}")
moneda.actualizar(7, 3)
print(f"Día 2, 7 caras y 3 cruces:   libreta {moneda.exitos}/{moneda.fallos}  ->  P(cara) ≈ {moneda.media():.2f}")
moneda.actualizar(5, 5)
print(f"Día 3, 5 caras y 5 cruces:   libreta {moneda.exitos}/{moneda.fallos} ->  P(cara) ≈ {moneda.media():.2f}")

de_golpe = Libreta()
de_golpe.actualizar(12, 8)
print(f"Todo de golpe (12 y 8):      libreta {de_golpe.exitos}/{de_golpe.fallos} ->  P(cara) ≈ {de_golpe.media():.2f}")


# --- 2. Test A/B: ¿qué botón consigue más clics? ---
def prob_b_mejor(clics_a, visitas_a, clics_b, visitas_b, simulaciones=100_000):
    a = Libreta()
    a.actualizar(clics_a, visitas_a - clics_a)
    b = Libreta()
    b.actualizar(clics_b, visitas_b - clics_b)
    gana_b = sum(1 for _ in range(simulaciones) if b.muestra() > a.muestra())
    return gana_b / simulaciones


print("\nBotón azul (A) contra botón verde (B):")
for clics_a, clics_b, visitas in [(5, 7, 100), (50, 65, 1000), (500, 650, 10_000)]:
    p = prob_b_mejor(clics_a, visitas, clics_b, visitas)
    if p > 0.95:
        decision = "lanzar el verde"
    elif p < 0.05:
        decision = "lanzar el azul"
    else:
        decision = "seguir recogiendo datos"
    print(f"  {visitas:>6} visitas cada uno: A {clics_a / visitas:.1%} | B {clics_b / visitas:.1%}"
          f"  ->  P(B mejor que A) = {p:.1%}  ->  {decision}")