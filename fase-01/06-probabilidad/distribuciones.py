import random
import statistics

random.seed(1)

# --- 1. Valor esperado: teoría contra simulación ---
def valor_esperado(valores, probs):
    return sum(v * p for v, p in zip(valores, probs))

teoria = valor_esperado([6, -1], [1/6, 5/6])
partidas = [6 if random.randint(1, 6) == 6 else -1 for _ in range(100_000)]
print(f"Juego (6 € con un 6, -1 € si no): teoría {teoria:.4f} €/partida | simulado {statistics.mean(partidas):.4f}")

# --- 2. Media y varianza de las tres clases ---
for nombre, notas in [("A", [5, 5, 5, 5]), ("B", [0, 10, 0, 10]), ("C", [4, 6, 4, 6])]:
    media = statistics.mean(notas)
    var = statistics.pvariance(notas)          # media de los cuadrados de las distancias
    print(f"Clase {nombre} {notas}: media = {media}, varianza = {var}, desviación = {var ** 0.5}")

# --- 3. Alturas con forma de campana: ¿se cumple la regla 68 / 95 / 99,7? ---
alturas = [random.gauss(175, 7) for _ in range(100_000)]
for k in (1, 2, 3):
    dentro = sum(1 for a in alturas if 175 - k * 7 <= a <= 175 + k * 7) / len(alturas)
    print(f"Entre {175 - k*7} y {175 + k*7} cm (± {k} desviación): {dentro:.1%}")
print(f"Más de 182 cm: {sum(1 for a in alturas if a > 182) / len(alturas):.1%}")
print(f"Más de 189 cm: {sum(1 for a in alturas if a > 189) / len(alturas):.1%}")

# --- 4. Teorema central del límite: la media de muchos dados ---
def histograma(datos, desde, hasta, n_cajas, titulo):
    print(f"\n{titulo}")
    ancho = (hasta - desde) / n_cajas
    cajas = [0] * n_cajas
    for d in datos:
        i = int((d - desde) / ancho)
        if 0 <= i < n_cajas:
            cajas[i] += 1
    mayor = max(cajas)
    for i, n in enumerate(cajas):
        centro = desde + (i + 0.5) * ancho
        print(f"  {centro:4.2f} | {'█' * round(n / mayor * 40)}")

repeticiones = 10_000
medias_1 = [random.randint(1, 6) for _ in range(repeticiones)]
medias_2 = [statistics.mean(random.randint(1, 6) for _ in range(2)) for _ in range(repeticiones)]
medias_30 = [statistics.mean(random.randint(1, 6) for _ in range(30)) for _ in range(repeticiones)]

histograma(medias_1, 0.5, 6.5, 6, "1 dado (10.000 tiradas):")
histograma(medias_2, 0.75, 6.25, 11, "Media de 2 dados (10.000 veces):")
histograma(medias_30, 2.75, 4.25, 15, "Media de 30 dados (10.000 veces), vista de cerca:")