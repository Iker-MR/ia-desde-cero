import numpy as np

# =============================================
# 1. MEDIA vs MEDIANA: los sueldos de una startup
# =============================================
sueldos = np.array([1800, 2000, 2200, 2500, 50000])   # el último es el CEO

print("=== SUELDOS DE UNA STARTUP ===")
print(f"Sueldos: {sueldos}")
print(f"Media:   {np.mean(sueldos):>8.0f} €")
print(f"Mediana: {np.median(sueldos):>8.0f} €")

sin_ceo = sueldos[:-1]
print(f"\nSin el CEO -> media {np.mean(sin_ceo):.0f} € | mediana {np.median(sin_ceo):.0f} €")

# =============================================
# 2. PERCENTILES: lo que tarda en responder un chatbot
# =============================================
rng = np.random.default_rng(0)
rapidas = rng.normal(200, 20, 970)    # 970 respuestas normales (~200 ms)
lentas = rng.normal(3000, 300, 30)    # 30 respuestas atascadas (~3 s)
latencias = np.concatenate([rapidas, lentas])

print("\n=== TIEMPO DE RESPUESTA DE UN CHATBOT (1.000 peticiones) ===")
print(f"Media: {np.mean(latencias):6.0f} ms")
for p in [50, 95, 99]:
    print(f"P{p}:   {np.percentile(latencias, p):6.0f} ms  -> el {p} % de usuarios espera esto o menos")

# =============================================
# 3. DESVIACIÓN TÍPICA: dos modelos con la misma media
# =============================================
modelo_a = np.array([0.85, 0.86, 0.84, 0.85, 0.86])   # 5 entrenamientos
modelo_b = np.array([0.70, 0.98, 0.79, 0.95, 0.84])

print("\n=== PRECISIÓN EN 5 ENTRENAMIENTOS ===")
for nombre, m in [("Modelo A", modelo_a), ("Modelo B", modelo_b)]:
    print(f"{nombre}: {m} -> media {np.mean(m):.3f} | desviación típica {np.std(m):.3f}")
print(f"La desviación de B es {np.std(modelo_b) / np.std(modelo_a):.0f} veces la de A")