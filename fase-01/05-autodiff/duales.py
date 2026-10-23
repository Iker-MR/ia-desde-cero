import math
from valor import Valor


class Dual:
    """Un número que viaja con su 'mochila': cuánto cambia si x sube un poquito."""

    def __init__(self, val, der=0.0):
        self.val = val      # el valor de siempre
        self.der = der      # la derivada respecto a x (la mochila)

    def __repr__(self):
        return f"Dual(val={self.val:.4f}, der={self.der:.4f})"

    def __add__(self, otro):
        otro = otro if isinstance(otro, Dual) else Dual(otro)
        return Dual(self.val + otro.val, self.der + otro.der)

    def __mul__(self, otro):
        otro = otro if isinstance(otro, Dual) else Dual(otro)
        return Dual(self.val * otro.val,
                    self.der * otro.val + self.val * otro.der)   # regla del producto

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, n):
        return Dual(self.val ** n, n * self.val ** (n - 1) * self.der)

    def tanh(self):
        t = math.tanh(self.val)
        return Dual(t, (1 - t ** 2) * self.der)


# --- 1. y = x · x con x = 3: la mochila empieza en 1 ("x respecto a x") ---
x = Dual(3.0, 1.0)
print("x · x con x = 3:", x * x)

# --- 2. Modo hacia delante (Dual) contra modo hacia atrás (Valor) ---
def f(x):
    return (x ** 3 + 2 * x + 1).tanh()

adelante = f(Dual(0.5, 1.0)).der

v = Valor(0.5)
f(v).backward()
atras = v.grad

print(f"\ntanh(x³ + 2x + 1) en x = 0.5")
print(f"  hacia delante (Dual):  {adelante:.6f}")
print(f"  hacia atrás   (Valor): {atras:.6f}")

# --- 3. Con 2 entradas: el modo hacia delante necesita una pasada POR ENTRADA ---
def g(x1, x2):
    return x1 * x2 + x1

d1 = g(Dual(2.0, 1.0), Dual(3.0, 0.0)).der   # pasada 1: mochila en x1
d2 = g(Dual(2.0, 0.0), Dual(3.0, 1.0)).der   # pasada 2: mochila en x2
print(f"\ng = x1 · x2 + x1 con x1 = 2, x2 = 3")
print(f"  hacia delante: 2 pasadas  ->  dg/dx1 = {d1}, dg/dx2 = {d2}")

a, b = Valor(2.0), Valor(3.0)
g(a, b).backward()
print(f"  hacia atrás:   1 pasada   ->  dg/dx1 = {a.grad}, dg/dx2 = {b.grad}")