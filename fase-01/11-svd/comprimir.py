from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

carpeta = Path(__file__).parent

# --- 1. Una foto es una matriz: cada número es el brillo de un píxel ---
foto = load_sample_image("china.jpg").mean(axis=2)      # a blanco y negro
alto, ancho = foto.shape
print(f"Foto: {alto} x {ancho} = {alto * ancho:,} números".replace(",", "."))

U, S, Vt = np.linalg.svd(foto, full_matrices=False)
print("\nLos 6 primeros valores singulares:", np.round(S[:6]).astype(int))
print("Los valores 100, 200 y 400:       ", np.round(S[[99, 199, 399]]).astype(int))

# --- 2. Comprimir: quedarse con las k capas más importantes ---
fig, ejes = plt.subplots(1, 4, figsize=(16, 3.6))
for eje, k in zip(ejes, [5, 20, 50, alto]):
    aprox = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    numeros = k * (alto + ancho + 1)
    tamano = numeros / (alto * ancho)
    error = np.abs(foto - aprox).mean()
    if k < alto:
        print(f"  {k:3} capas: {tamano:6.1%} de los números | error medio {error:5.1f} (sobre 255)")
    eje.imshow(aprox, cmap="gray", vmin=0, vmax=255)
    eje.set_title("Original" if k == alto else f"{k} capas ({tamano:.0%} del tamaño)")
    eje.axis("off")
plt.tight_layout()
plt.savefig(carpeta / "foto_comprimida.png", dpi=90)
print("Imágenes guardadas en foto_comprimida.png")

# --- 3. LoRA: el "ajuste" de una capa de un LLM casi siempre tiene pocas capas ---
rng = np.random.default_rng(0)
d = 1000                                                  # una capa de 1.000 x 1.000 pesos
cambio = rng.normal(size=(d, 4)) @ rng.normal(size=(4, d)) + rng.normal(scale=0.1, size=(d, d))
S_cambio = np.linalg.svd(cambio, compute_uv=False)
print("\nAjuste de una capa de 1.000 x 1.000. Primeros 6 valores singulares:", np.round(S_cambio[:6]).astype(int))

r = 4
print(f"Actualizar la capa entera: {d * d:>9,} números".replace(",", "."))
print(f"LoRA con r = {r} (A: {d}x{r} y B: {r}x{d}): {2 * d * r:>6,} números".replace(",", ".") +
      f"  ->  {2 * d * r / (d * d):.1%} de los pesos")