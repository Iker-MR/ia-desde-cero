import math
import numpy as np

np.seterr(over="ignore")                      # no mostrar avisos: queremos ver el resultado

# --- 1. 0,1 + 0,2 ---
print("0.1 + 0.2 =", 0.1 + 0.2)
print("¿0.1 + 0.2 == 0.3?", 0.1 + 0.2 == 0.3)
print("¿math.isclose(0.1 + 0.2, 0.3)?", math.isclose(0.1 + 0.2, 0.3))

# --- 2. Cuántas cifras guarda cada formato ---
print("\nEl número 1/3 guardado en cada formato:")
for tipo in [np.float64, np.float32, np.float16]:
    print(f"  {tipo.__name__:8} {tipo(1) / tipo(3):.20f}")

# --- 3. Sumar algo pequeño a algo grande ---
print(f"\n1.000.000 + 0,01 en float32: {np.float32(1_000_000) + np.float32(0.01):.2f}")
total = np.float32(1_000_000)
for _ in range(1000):
    total += np.float32(0.01)                 # sumar 0,01 mil veces = sumar 10
print(f"1.000.000 + (0,01 sumado mil veces) en float32: {total:.2f}  (debería ser 1000010.00)")

# --- 4. Números demasiado grandes o demasiado pequeños ---
print("\nEl número más grande que cabe en float16:", int(np.finfo(np.float16).max))
print("70.000 en float16:", np.float16(70_000))
print("exp(89) en float32:", np.exp(np.float32(89)))
print("exp(-110) en float32:", np.exp(np.float32(-110)), "  (demasiado pequeño: se redondea a 0)")
print("0.0001 en float16:", np.float16(0.0001), " | 0.00000001 en float16:", np.float16(0.00000001))

# --- 5. Cancelación catastrófica: restar dos números casi iguales ---
datos = np.array([1_000_000.1, 1_000_000.2, 1_000_000.3], dtype=np.float32)   # varianza: unos 0,0065
media = datos.mean()
formula_rapida = np.mean(datos ** 2) - media ** 2            # E[x²] - E[x]²: resta dos números enormes casi iguales
formula_segura = np.mean((datos - media) ** 2)               # restar la media primero
print(f"\nVarianza de 3 datos cerca de un millón (debería salir unos 0,0065):")
print(f"  restando dos números enormes: {formula_rapida}")
print(f"  centrando primero:            {formula_segura:.5f}")