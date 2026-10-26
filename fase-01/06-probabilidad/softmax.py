import math


def softmax_ingenua(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def softmax(logits):
    mayor = max(logits)
    exps = [math.exp(z - mayor) for z in logits]     # el truco: restar el máximo
    total = sum(exps)
    return [e / total for e in exps]


def entropia_cruzada(probs, correcta):
    return -math.log(probs[correcta])


def mostrar(nombre, probs):
    print(f"  {nombre:24}", "  ".join(f"{p:.3f}" for p in probs))


clases = ["gato", "perro", "pájaro"]

# --- 1. De puntuaciones (logits) a probabilidades ---
print("Softmax (gato, perro, pájaro):")
mostrar("[2, 1, 0.1]", softmax([2, 1, 0.1]))
mostrar("[3, 3, 3]", softmax([3, 3, 3]))
mostrar("[102, 101, 100.1]", softmax([102, 101, 100.1]))

# --- 2. ¿Por qué restar el máximo? ---
print("\nLogits enormes [1000, 999, 998]:")
mostrar("con el truco", softmax([1000, 999, 998]))
try:
    softmax_ingenua([1000, 999, 998])
except OverflowError as e:
    print(f"  sin el truco             ERROR: {e}")

# --- 3. Entropía cruzada: la foto era un GATO (índice 0) ---
print("\nPérdida si la respuesta correcta es 'gato':")
for p_gato in [0.99, 0.66, 0.5, 0.1, 0.01, 0.0001]:
    resto = (1 - p_gato) / 2
    perdida = entropia_cruzada([p_gato, resto, resto], 0)
    print(f"  P(gato) = {p_gato:<7} ->  pérdida = {perdida:.2f}")
try:
    entropia_cruzada([0.0, 0.5, 0.5], 0)
except ValueError as e:
    print(f"  P(gato) = 0       ->  ERROR: {e}")

# --- 4. Temperatura: el parámetro 'temperature' de la API de Claude ---
print("\nTemperatura (los logits se dividen entre T antes de la softmax):")
for t in [0.2, 1.0, 5.0]:
    mostrar(f"T = {t}", softmax([z / t for z in [2, 1, 0.1]]))

# --- 5. Por qué se trabaja con logaritmos ---
print("\nProbabilidad de un texto en el que cada palabra tiene P = 0,01:")
for n in [10, 100, 200]:
    producto = 1.0
    suma_logs = 0.0
    for _ in range(n):
        producto *= 0.01
        suma_logs += math.log(0.01)
    print(f"  {n:3} palabras:  multiplicando = {producto:.3e}   |   sumando logs = {suma_logs:.1f}")