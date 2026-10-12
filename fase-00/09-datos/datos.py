from pathlib import Path

from datasets import load_dataset

CARPETA = Path(__file__).parent / "data"
CARPETA.mkdir(exist_ok=True)

# 1. Cargar el dataset (la primera vez lo descarga; luego usa la caché)
ds = load_dataset("stanfordnlp/imdb")
print(ds)

train = ds["train"]
print("\nPrimera reseña:")
print("  etiqueta:", train[0]["label"], "(0 = negativa, 1 = positiva)")
print("  texto:", train[0]["text"][:200], "...")
print("Etiqueta de la ÚLTIMA reseña:", train[len(train) - 1]["label"])

# 2. Guardar en tres formatos y comparar tamaños
train.to_csv(str(CARPETA / "imdb_train.csv"))
train.to_json(str(CARPETA / "imdb_train.json"))
train.to_parquet(str(CARPETA / "imdb_train.parquet"))

print("\nTamaño en disco:")
for ext in ["csv", "json", "parquet"]:
    archivo = CARPETA / f"imdb_train.{ext}"
    print(f"  {ext:8} {archivo.stat().st_size / 1e6:6.1f} MB")

# 3. Dividir en train / validation / test (80 / 10 / 10) con semilla fija
paso1 = train.train_test_split(test_size=0.2, seed=42)
paso2 = paso1["test"].train_test_split(test_size=0.5, seed=42)

train_ds = paso1["train"]
val_ds = paso2["train"]
test_ds = paso2["test"]

print("\nDivisiones:")
for nombre, d in [("train", train_ds), ("validation", val_ds), ("test", test_ds)]:
    positivas = sum(d["label"]) / len(d)
    print(f"  {nombre:11} {len(d):6} reseñas | {positivas:.0%} positivas") 

# 4. Experimento: dividir SIN mezclar (shuffle=False)
malo = train.train_test_split(test_size=0.2, shuffle=False)
print("\nSin mezclar:")
for nombre, d in [("train", malo["train"]), ("test", malo["test"])]:
    positivas = sum(d["label"]) / len(d)
    print(f"  {nombre:6} {len(d):6} reseñas | {positivas:.0%} positivas")