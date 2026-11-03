from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # dibujar en un archivo, sin abrir ventanas
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

digitos = load_digits()
X, y = digitos.data, digitos.target       # 1.797 imágenes de 8x8 píxeles = 64 columnas
print("Datos:", X.shape, "→ cada dígito es un punto en 64 dimensiones")

# --- 1. PCA (lo mismo del bloque 1, ahora con 64 columnas) ---
media = X.mean(axis=0)
Xc = X - media
valores, vectores = np.linalg.eigh(np.cov(Xc, rowvar=False))
orden = np.argsort(valores)[::-1]
valores, vectores = valores[orden], vectores[:, orden]
acumulada = np.cumsum(valores) / valores.sum()

print("\n¿Cuántas componentes hacen falta?")
for objetivo in [0.5, 0.8, 0.9, 0.95]:
    k = int(np.argmax(acumulada >= objetivo)) + 1
    print(f"  para guardar el {objetivo:.0%} de la información: {k:2} de 64")

# --- 2. Reconstruir un dígito con pocas componentes ---
def dibujar(fila):
    tonos = " .:-=+*#%@"
    for i in range(8):
        print("    " + "".join(tonos[int(np.clip(v, 0, 16) / 16 * 9)] * 2 for v in fila[i * 8:(i + 1) * 8]))

n = 0                                     # el primer dígito del dataset
print(f"\nEl dígito {y[n]}, original (64 números):")
dibujar(X[n])
for k in [2, 10, 30]:
    W = vectores[:, :k]
    reconstruido = (Xc[n] @ W) @ W.T + media
    print(f"\nReconstruido con solo {k} números ({acumulada[k - 1]:.0%} de la información):")
    dibujar(reconstruido)

# --- 3. Dibujar los 1.797 dígitos en 2D: PCA contra t-SNE ---
pca_2d = Xc @ vectores[:, :2]
tsne_2d = TSNE(n_components=2, random_state=0, init="pca").fit_transform(X)

def acierto_vecinos(puntos, etiquetas, k=5):
    """¿Los 5 vecinos más cercanos en el dibujo son del mismo número? (más alto = grupos más separados)"""
    d = ((puntos[:, None, :] - puntos[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    vecinos = np.argsort(d, axis=1)[:, :k]
    votos = etiquetas[vecinos]
    prediccion = np.array([np.bincount(v, minlength=10).argmax() for v in votos])
    return (prediccion == etiquetas).mean()

print(f"\nEn el dibujo 2D, ¿los vecinos de cada punto son del mismo número?")
print(f"  PCA:   {acierto_vecinos(pca_2d, y):.1%}")
print(f"  t-SNE: {acierto_vecinos(tsne_2d, y):.1%}")

fig, ejes = plt.subplots(1, 2, figsize=(12, 5.5))
for eje, puntos, titulo in [(ejes[0], pca_2d, "PCA (2 componentes)"), (ejes[1], tsne_2d, "t-SNE")]:
    sc = eje.scatter(puntos[:, 0], puntos[:, 1], c=y, cmap="tab10", s=8)
    for d in range(10):
        cx, cy = np.median(puntos[y == d], axis=0)
        eje.text(cx, cy, str(d), fontsize=16, weight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="circle", fc="white", alpha=0.8))
    eje.set_title(titulo)
    eje.set_xticks([]); eje.set_yticks([])
plt.tight_layout()
archivo = Path(__file__).parent / "digitos_2d.png"     # junto a este script
plt.savefig(archivo, dpi=110)
print(f"\nGráfico guardado en {archivo}")