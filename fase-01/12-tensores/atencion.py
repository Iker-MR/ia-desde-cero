import numpy as np

np.set_printoptions(precision=2, suppress=True)
rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)          # el truco de la 1.06
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


B, T, E, H = 2, 5, 8, 2       # 2 frases, 5 tokens cada una, 8 números por token, 2 cabezas
D = E // H                    # cada cabeza trabaja con 4 de los 8 números

X = rng.normal(size=(B, T, E))                       # los embeddings de los tokens
W_q, W_k, W_v, W_o = (rng.normal(size=(E, E)) * 0.5 for _ in range(4))
print(f"0. Entrada X                                   {X.shape}   (frases, tokens, dimensiones)")

# 1. Cada token genera su pregunta (Q), su etiqueta (K) y su contenido (V)
Q, K, V = X @ W_q, X @ W_k, X @ W_v
print(f"1. Q, K y V = X @ W                            {Q.shape}")

# 2. Repartir los 8 números entre 2 cabezas de 4: reshape + transpose
def separar_cabezas(t):
    return t.reshape(B, T, H, D).transpose(0, 2, 1, 3)
Q, K, V = separar_cabezas(Q), separar_cabezas(K), separar_cabezas(V)
print(f"2. Separar en cabezas                          {Q.shape}   (frases, cabezas, tokens, D)")

# 3. Cada token compara su pregunta con la etiqueta de todos los tokens (producto escalar)
puntuaciones = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
print(f"3. Puntuaciones = Q · K                        {puntuaciones.shape}   (frases, cabezas, token, token)")

# 4. Softmax sobre el último eje: cuánta atención presta cada token a cada uno de los demás
pesos = softmax(puntuaciones, axis=-1)
print(f"4. Pesos = softmax(puntuaciones)               {pesos.shape}")
print("   Cada fila suma 1:", pesos[0, 0].sum(axis=-1))

# 5. Cada token recoge el contenido (V) de los demás, según esos pesos
salida = np.einsum("bhts,bhsd->bhtd", pesos, V)
print(f"5. Salida = pesos · V                          {salida.shape}")

# 6. Volver a juntar las cabezas y mezclarlas
juntas = salida.transpose(0, 2, 1, 3).reshape(B, T, E)
final = juntas @ W_o
print(f"6. Juntar cabezas y W_o                        {final.shape}   ¡la misma forma que la entrada!")

print("\nAtención de la frase 1, cabeza 1 (fila = token que mira, columna = token mirado):")
print(pesos[0, 0])