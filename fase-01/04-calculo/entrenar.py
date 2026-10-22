# Datos: el "secreto" que el modelo debe descubrir es y = 2x + 1
xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

w, b = 0.0, 0.0          # el modelo empieza sin saber nada
tasa = 0.01

for epoca in range(1001):
    dw, db, perdida = 0.0, 0.0, 0.0
    for x, y in zip(xs, ys):
        pred = w * x + b             # 1. predecir
        error = pred - y
        perdida += error ** 2        # 2. medir el error (MSE)
        dw += 2 * error * x          # 3. gradiente respecto a w
        db += 2 * error              #    gradiente respecto a b
    n = len(xs)
    perdida, dw, db = perdida / n, dw / n, db / n
    w -= tasa * dw                   # 4. dar un paso cuesta abajo
    b -= tasa * db
    if epoca in (0, 1, 10, 100, 500, 1000):
        print(f"época {epoca:4}  w = {w:.4f}  b = {b:.4f}  pérdida = {perdida:.6f}")

print(f"\nAprendido: y = {w:.2f}x + {b:.2f}")
print(f"Predicción para x = 10: {w * 10 + b:.2f}")