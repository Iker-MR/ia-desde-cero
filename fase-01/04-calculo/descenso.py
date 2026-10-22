def derivada(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)       # el truco de mirar a cada lado


def f(x):
    return x ** 2


print("Derivada numérica de x² en x = 3:", round(derivada(f, 3.0), 6))

for tasa in [0.1, 0.5, 0.9, 1.1]:
    x = 3.0
    recorrido = [x]
    for paso in range(8):
        x = x - tasa * derivada(f, x)            # descenso de gradiente
        recorrido.append(x)
    print(f"\ntasa = {tasa}:")
    print("  ", [round(v, 3) for v in recorrido])