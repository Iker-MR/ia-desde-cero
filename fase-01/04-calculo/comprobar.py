# Comprobación de gradientes: ¿la fórmula de la regla de la cadena coincide con el truco numérico?
x, y = 3.0, 7.0          # un dato
w, b = 0.5, 0.2          # unos pesos cualesquiera
h = 1e-5


def perdida(w, b):
    return (w * x + b - y) ** 2


error = w * x + b - y
print("Con la regla de la cadena:")
print(f"  dL/dw = 2·error·x = {2 * error * x:.6f}")
print(f"  dL/db = 2·error   = {2 * error:.6f}")

print("Con el truco numérico:")
print(f"  dL/dw ≈ {(perdida(w + h, b) - perdida(w - h, b)) / (2 * h):.6f}")
print(f"  dL/db ≈ {(perdida(w, b + h) - perdida(w, b - h)) / (2 * h):.6f}")

# Taylor: aproximar f(x) = x² cerca de x = 3
f = lambda t: t ** 2
paso = 0.1
print(f"\nf(3.1) real                    = {f(3.1):.4f}")
print(f"Aproximación con la pendiente  = {f(3) + 2 * 3 * paso:.4f}")
print(f"Añadiendo la curvatura         = {f(3) + 2 * 3 * paso + 0.5 * 2 * paso ** 2:.4f}")