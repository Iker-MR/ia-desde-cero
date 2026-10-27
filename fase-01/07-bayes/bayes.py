def bayes(prior, acierta_si_es_verdad, falsa_alarma):
    """Probabilidad de que algo sea verdad DESPUÉS de ver una señal positiva."""
    positivos_verdaderos = acierta_si_es_verdad * prior
    positivos_falsos = falsa_alarma * (1 - prior)
    return positivos_verdaderos / (positivos_verdaderos + positivos_falsos)


# --- 1. Los dos pueblos: mismo test, distinta enfermedad ---
print("Pueblo con 10 enfermos de 1.000: ", f"{bayes(10/1000, 1.0, 0.1):.1%}")
print("Pueblo con 100 enfermos de 1.000:", f"{bayes(100/1000, 1.0, 0.1):.1%}")

# --- 2. La enfermedad rara (1 de cada 10.000) y un test del 99 % ---
p = 1 / 10_000
print(f"\nAntes de ningún test:        {p:.2%}")
for prueba in [1, 2, 3]:
    p = bayes(p, 0.99, 0.01)          # lo que crees ahora se convierte en tu nuevo punto de partida
    print(f"Tras el positivo número {prueba}:   {p:.2%}")

# --- 3. Filtro de spam con UNA palabra ---
print(f"\nEmail con 'lotería': {bayes(300/1000, 1/20, 1/1000):.1%} de que sea spam (antes de leerlo: 30%)")