import math


# El "plátano" de Rosenbrock: un valle estrecho y CURVO. El fondo está en (1, 1).
def platano(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def gradiente(p):
    x, y = p
    return [-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)]


class Descenso:
    def __init__(self, tasa):
        self.tasa = tasa

    def paso(self, params, grads):
        return [p - self.tasa * g for p, g in zip(params, grads)]


class Momento:
    def __init__(self, tasa, momento=0.9):
        self.tasa, self.momento = tasa, momento
        self.velocidad = None

    def paso(self, params, grads):
        if self.velocidad is None:
            self.velocidad = [0.0] * len(params)
        self.velocidad = [self.momento * v + g for v, g in zip(self.velocidad, grads)]
        return [p - self.tasa * v for p, v in zip(params, self.velocidad)]


class Adam:
    def __init__(self, tasa, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.tasa, self.beta1, self.beta2, self.epsilon = tasa, beta1, beta2, epsilon
        self.m = None          # media de las pendientes (dirección, como el momento)
        self.v = None          # media de las pendientes AL CUADRADO (lo fuertes que suelen ser)
        self.t = 0

    def paso(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)
        self.t += 1
        self.m = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(self.m, grads)]
        self.v = [self.beta2 * v + (1 - self.beta2) * g ** 2 for v, g in zip(self.v, grads)]
        m_corr = [m / (1 - self.beta1 ** self.t) for m in self.m]      # corrección del arranque en 0
        v_corr = [v / (1 - self.beta2 ** self.t) for v in self.v]
        # Cada peso divide su paso entre lo fuertes que suelen ser SUS pendientes
        return [p - self.tasa * m / (math.sqrt(v) + self.epsilon)
                for p, m, v in zip(params, m_corr, v_corr)]


def optimizar(optimizador, pasos):
    params = [-1.0, 1.0]
    llegada = None
    for i in range(1, pasos + 1):
        params = optimizador.paso(params, gradiente(params))
        if llegada is None and platano(params) < 0.0001:
            llegada = i
    return params, llegada


print("Del punto (-1, 1) al fondo (1, 1) del plátano de Rosenbrock:\n")
for nombre, opt in [("Descenso normal", Descenso(tasa=0.0005)),
                    ("Con momento    ", Momento(tasa=0.0001)),
                    ("Adam           ", Adam(tasa=0.01))]:
    final, llegada = optimizar(opt, pasos=20_000)
    print(f"  {nombre}: llega al fondo en {llegada:>6} pasos")

print("\nDónde está cada uno tras 5.000 pasos:")
for nombre, opt in [("Descenso normal", Descenso(tasa=0.0005)),
                    ("Con momento    ", Momento(tasa=0.0001)),
                    ("Adam           ", Adam(tasa=0.01))]:
    final, _ = optimizar(opt, pasos=5_000)
    print(f"  {nombre}: x = {final[0]:.3f}, y = {final[1]:.3f}   pérdida = {platano(final):.6f}")