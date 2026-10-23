import math

class Valor:
    """Un número que recuerda de dónde viene y guarda su gradiente."""

    def __init__(self, dato, hijos=(), op=""):
        self.dato = dato
        self.grad = 0.0                  # dy/d(este valor); empieza en 0
        self._backward = lambda: None    # cómo pasar el gradiente a sus hijos
        self._hijos = set(hijos)         # los valores que lo produjeron
        self._op = op

    def __repr__(self):
        return f"Valor(dato={self.dato}, grad={self.grad})"

    def __add__(self, otro):
        otro = otro if isinstance(otro, Valor) else Valor(otro)
        sal = Valor(self.dato + otro.dato, (self, otro), "+")
        def _backward():
            self.grad += 1.0 * sal.grad          # derivada local de la suma: 1
            otro.grad += 1.0 * sal.grad
        sal._backward = _backward
        return sal

    def __mul__(self, otro):
        otro = otro if isinstance(otro, Valor) else Valor(otro)
        sal = Valor(self.dato * otro.dato, (self, otro), "*")
        def _backward():
            self.grad += otro.dato * sal.grad    # derivada local: el OTRO factor
            otro.grad += self.dato * sal.grad
        sal._backward = _backward
        return sal

    def relu(self):
        sal = Valor(max(0, self.dato), (self,), "relu")
        def _backward():
            self.grad += (1.0 if sal.dato > 0 else 0.0) * sal.grad
        sal._backward = _backward
        return sal

    # ---------- Bloque 2: más operaciones ----------
    def __neg__(self):                 # -x  es  x * (-1)
        return self * -1

    def __sub__(self, otro):           # a - b  es  a + (-b)
        return self + (-otro)

    def __radd__(self, otro):          # 2 + x  (el número va delante)
        return self + otro

    def __rmul__(self, otro):          # 2 * x
        return self * otro

    def __pow__(self, n):              # x ** n, con n un número normal
        sal = Valor(self.dato ** n, (self,), f"**{n}")
        def _backward():
            self.grad += n * self.dato ** (n - 1) * sal.grad
        sal._backward = _backward
        return sal

    def tanh(self):
        t = math.tanh(self.dato)
        sal = Valor(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t**2) * sal.grad
        sal._backward = _backward
        return sal

    def backward(self):
        # 1) Ordenar el grafo: cada nodo después de los que lo producen
        orden, visitados = [], set()
        def construir(v):
            if v not in visitados:
                visitados.add(v)
                for h in v._hijos:
                    construir(h)
                orden.append(v)
        construir(self)
        # 2) Semilla dy/dy = 1 y recorrer al revés aplicando la regla de la cadena
        self.grad = 1.0
        for v in reversed(orden):
            v._backward()


if __name__ == "__main__":
    # y = relu(x1 * x2 + b)
    x1, x2, b = Valor(2.0), Valor(3.0), Valor(1.0)
    a = x1 * x2
    c = a + b
    y = c.relu()
    print("Hacia delante:  a =", a.dato, "| c =", c.dato, "| y =", y.dato)

    y.backward()
    print("Hacia atrás:    dy/dx1 =", x1.grad, "| dy/dx2 =", x2.grad, "| dy/db =", b.grad)

    # Experimento: ¿qué pasa si la entrada a la relu es negativa?
    x1, x2, b = Valor(2.0), Valor(-3.0), Valor(1.0)
    y = (x1 * x2 + b).relu()
    y.backward()
    print("\nCon x2 = -3:    y =", y.dato, "| dy/dx1 =", x1.grad, "| dy/dx2 =", x2.grad, "| dy/db =", b.grad)

    # y = x * x  ->  la misma x por dos caminos
    x = Valor(3.0)
    y = x * x
    y.backward()
    print("\ny = x · x con x = 3:  dy/dx =", x.grad, "(debería ser 2x = 6)")