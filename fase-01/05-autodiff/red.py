import random
from valor import Valor


class Neurona:
    def __init__(self, n_entradas):
        self.w = [Valor(random.uniform(-1, 1)) for _ in range(n_entradas)]
        self.b = Valor(0.0)

    def __call__(self, x):
        suma = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)   # w1·x1 + w2·x2 + ... + b
        return suma.tanh()

    def parametros(self):
        return self.w + [self.b]


class Capa:
    def __init__(self, n_entradas, n_neuronas):
        self.neuronas = [Neurona(n_entradas) for _ in range(n_neuronas)]

    def __call__(self, x):
        return [n(x) for n in self.neuronas]

    def parametros(self):
        return [p for n in self.neuronas for p in n.parametros()]


class Red:
    def __init__(self, tamanos):
        self.capas = [Capa(tamanos[i], tamanos[i + 1]) for i in range(len(tamanos) - 1)]

    def __call__(self, x):
        for capa in self.capas:
            x = capa(x)
        return x[0] if len(x) == 1 else x

    def parametros(self):
        return [p for capa in self.capas for p in capa.parametros()]


if __name__ == "__main__":
    random.seed(42)
    red = Red([2, 4,1])            # 2 entradas, 4 neuronas ocultas, 1 salida
    print("Parámetros de la red:", len(red.parametros()))

    # XOR: la salida es 1 si las entradas son distintas, -1 si son iguales
    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ys = [-1, 1, 1, -1]

    tasa = 0.05
    for paso in range(101):
        # 1) Hacia delante: predicciones y pérdida
        preds = [red(x) for x in xs]
        perdida = sum((p - y) ** 2 for p, y in zip(preds, ys))

        # 2) Poner los gradientes a 0 (por el +=)
        for p in red.parametros():
            p.grad = 0.0

        # 3) Hacia atrás: TODOS los gradientes de golpe
        perdida.backward()

        # 4) Actualizar: cada peso, un pasito en contra de su gradiente
        for p in red.parametros():
            p.dato -= tasa * p.grad

        if paso % 20 == 0:
            print(f"paso {paso:3d}   pérdida = {perdida.dato:.4f}")

    print("\nPredicciones tras entrenar:")
    for x, y in zip(xs, ys):
        print(f"  entrada = {x}   objetivo = {y:2d}   predicción = {red(x).dato:6.3f}")