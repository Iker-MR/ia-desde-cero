# Un valle estrecho: muy empinado de lado a lado (y), casi plano a lo largo (x).
# El fondo está en x = 0, y = 0.
def perdida(x, y):
    return x ** 2 + 100 * y ** 2

def gradiente(x, y):
    return 2 * x, 200 * y


def descenso(tasa, momento=0.0, pasos=2000, ver=0):
    x, y = 10.0, 1.0
    vx = vy = 0.0                                  # la "velocidad" acumulada
    for paso in range(1, pasos + 1):
        gx, gy = gradiente(x, y)
        vx = momento * vx + gx                     # con momento = 0 es el descenso normal
        vy = momento * vy + gy
        x, y = x - tasa * vx, y - tasa * vy
        if paso <= ver:
            print(f"    paso {paso}: x = {x:6.2f}   y = {y:6.2f}")
        if perdida(x, y) < 0.001:
            return paso
        if abs(x) > 1e6 or abs(y) > 1e6:
            return None
    return None


print("Descenso normal (tasa 0.009), primeros pasos:")
descenso(0.009, ver=6)

print("\nPasos hasta llegar al fondo (pérdida < 0,001):")
for nombre, tasa, momento in [("Normal,      tasa 0.009 ", 0.009, 0.0),
                              ("Normal,      tasa 0.0101", 0.0101, 0.0),
                              ("Con momento, tasa 0.005 ", 0.005, 0.9)]:
    pasos = descenso(tasa, momento)
    print(f"  {nombre}: {pasos if pasos else 'no llega (se dispara)'}")