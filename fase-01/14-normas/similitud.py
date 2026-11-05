import numpy as np

# ---------------------------------------------
# Cada documento = cuántas veces dice [gatos, fútbol]
# ---------------------------------------------
docs = {
    "A: nota corta de gatos":      np.array([2, 0]),
    "B: artículo LARGO de gatos":  np.array([10, 1]),
    "C: crónica de fútbol":        np.array([1, 3]),
}
pregunta = np.array([1, 0])   # "háblame de gatos"

def l2(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def producto_punto(a, b):
    """Multiplicar por parejas y sumar."""
    return np.sum(a * b)

def coseno(a, b):
    """Producto punto dividido entre las dos longitudes: solo mira la DIRECCIÓN."""
    return producto_punto(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------------------------------------
# 1. El buscador con L2 (la paloma)
# ---------------------------------------------
print("=== BUSCADOR CON L2 (menor = más parecido) ===")
for nombre, v in sorted(docs.items(), key=lambda kv: l2(pregunta, kv[1])):
    print(f"  {l2(pregunta, v):5.2f}  {nombre}")

# ---------------------------------------------
# 2. El buscador con coseno
# ---------------------------------------------
print("\n=== BUSCADOR CON COSENO (mayor = más parecido) ===")
for nombre, v in sorted(docs.items(), key=lambda kv: -coseno(pregunta, kv[1])):
    print(f"  {coseno(pregunta, v):5.3f}  {nombre}")

# ---------------------------------------------
# 3. Misma dirección, distinta longitud
# ---------------------------------------------
print("\n=== MISMA DIRECCIÓN, DISTINTA LONGITUD ===")
a, b = np.array([1, 1]), np.array([5, 5])
print(f"(1,1) y (5,5): L2 = {l2(a, b):.2f} | coseno = {coseno(a, b):.3f}")
a, c = np.array([1, 0]), np.array([0, 1])
print(f"(1,0) y (0,1): L2 = {l2(a, c):.2f} | coseno = {coseno(a, c):.3f}  (perpendiculares)")
d = np.array([-1, 0])
print(f"(1,0) y (-1,0): L2 = {l2(a, d):.2f} | coseno = {coseno(a, d):.3f} (opuestos)")

# ---------------------------------------------
# 4. Normalizar: encoger cada flecha a longitud 1
# ---------------------------------------------
print("\n=== NORMALIZAR (longitud 1) ===")
B = docs["B: artículo LARGO de gatos"]
B_norm = B / np.linalg.norm(B)
p_norm = pregunta / np.linalg.norm(pregunta)
print(f"B original:    {B}  longitud {np.linalg.norm(B):.2f}")
print(f"B normalizado: {np.round(B_norm, 3)}  longitud {np.linalg.norm(B_norm):.2f}")
print(f"Producto punto ya normalizados: {producto_punto(p_norm, B_norm):.3f}")
print(f"Coseno de los originales:       {coseno(pregunta, B):.3f}")