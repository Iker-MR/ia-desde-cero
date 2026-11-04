import numpy as np

np.seterr(over="ignore", invalid="ignore")


def a_bfloat16(x):
    """NumPy no trae bfloat16: lo imitamos quedándonos con los 16 primeros bits de un float32."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    redondeo = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + redondeo) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def mostrar(nombre, valor):
    f32 = np.float32(valor)
    f16 = np.float16(valor)
    bf16 = a_bfloat16(valor)
    print(f"  {nombre:26} float32 = {f32:<14.8g} float16 = {float(f16):<14.8g} bfloat16 = {float(bf16):.8g}")


# --- 1. Rango contra precisión ---
print("El mismo número en cada formato:")
mostrar("70.000 (un logit grande)", 70_000)
mostrar("0,00000001 (un gradiente)", 1e-8)
mostrar("pi", np.pi)

# --- 2. Gradientes diminutos y el escalado de la pérdida ---
gradiente = 1e-8
print(f"\nGradiente real: {gradiente}")
print(f"  en float16:                        {float(np.float16(gradiente))}   <- ¡se ha perdido!")
escala = 65536
escalado = np.float16(gradiente * escala)                       # multiplicar ANTES de bajar a float16
print(f"  multiplicado por {escala} en float16: {float(escalado):.6f}")
print(f"  dividido después en float32:       {np.float32(escalado) / escala:.3g}   <- recuperado")

# --- 3. Por qué los pesos se guardan en float32 ---
print("\nUn peso de 1.0 recibe 1.000 actualizaciones de +0,0001 (debería acabar en 1.1):")
for nombre, convertir in [("float16", np.float16), ("bfloat16", a_bfloat16), ("float32", np.float32)]:
    peso = convertir(1.0)
    for _ in range(1000):
        peso = convertir(peso + convertir(0.0001))
    print(f"  pesos en {nombre:9} -> {float(peso):.4f}")

# --- 4. Cazar el primer NaN ---
perdidas = [2.3, 1.9, 1.4, 1.1, 0.9, np.inf, np.nan, np.nan, np.nan]   # así suele verse en el registro
perdidas = np.array(perdidas)
primero = int(np.argmax(~np.isfinite(perdidas)))
print(f"\nPérdidas: {perdidas}")
print(f"El primer valor no válido aparece en el paso {primero}: {perdidas[primero]}  (luego todo es nan)")