from valor import Valor

# --- 1. Resta ---
a, b = Valor(5.0), Valor(2.0)
y = a - b
y.backward()
print("Resta      y = a - b       =", y.dato, "| dy/da =", a.grad, "| dy/db =", b.grad)

# --- 2. Potencias ---
x = Valor(3.0)
y = x ** 2
y.backward()
print("Potencia   y = x²  (x = 3) =", y.dato, "| dy/dx =", x.grad)

x = Valor(2.0)
y = x ** 3
y.backward()
print("Potencia   y = x³  (x = 2) =", y.dato, "| dy/dx =", x.grad)

# --- 3. tanh: cerca de 0 y lejos de 0 ---
for v in [0.0, 2.0, 5.0]:
    x = Valor(v)
    y = x.tanh()
    y.backward()
    print(f"tanh       x = {v}:  tanh = {y.dato:.4f} | dy/dx = {x.grad:.4f}")

# --- 4. Comprobación de gradientes (gradient check) ---
def comprobar(nombre, formula, x0, h=1e-6):
    x = Valor(x0)
    formula(x).backward()
    autodiff = x.grad
    numerico = (formula(Valor(x0 + h)).dato - formula(Valor(x0 - h)).dato) / (2 * h)
    diferencia = abs(autodiff - numerico)
    estado = "OK" if diferencia < 1e-5 else "¡FALLO!"
    print(f"{nombre:28} autodiff = {autodiff:.6f} | numérico = {numerico:.6f} | dif = {diferencia:.1e}  {estado}")

print("\nComprobación de gradientes:")
comprobar("x³ + 2x + 1", lambda x: x ** 3 + 2 * x + 1, 0.5)
comprobar("tanh(x³ + 2x + 1)", lambda x: (x ** 3 + 2 * x + 1).tanh(), 0.5)
comprobar("(x - 3)² · tanh(x)", lambda x: (x - 3) ** 2 * x.tanh(), 1.5)