# --- Error 1: dos nombres, una sola lista ---
pesos = [0.5, 0.5, 0.5]
copia = pesos.copy()
copia[0] = 99
print("1) pesos:", pesos)


# --- Error 2: argumento por defecto que es una lista ---
def agregar_ejemplo(x, lote=None):
    if lote is None:
        lote = []
    lote.append(x)
    return lote


print("2) primera llamada:", agregar_ejemplo(1))
print("2) segunda llamada:", agregar_ejemplo(2))


# --- Error 3: un generador que se agota ---
def leer_datos():
    for i in range(3):
        yield i


datos = leer_datos()
for epoca in range(2):
    datos = leer_datos()          # generador nuevo cada vez
    vistos = [x for x in datos]
    print(f"3) época {epoca}: {vistos}")