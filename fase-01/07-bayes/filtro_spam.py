import math
from collections import defaultdict


class NaiveBayes:
    def __init__(self, suavizado=1.0):
        self.suavizado = suavizado
        self.emails_por_clase = defaultdict(int)                      # cuántos spam y cuántos normales
        self.veces_palabra = defaultdict(lambda: defaultdict(int))    # veces que sale cada palabra en cada clase
        self.palabras_por_clase = defaultdict(int)                    # total de palabras de cada clase
        self.vocabulario = set()

    def entrenar(self, emails, etiquetas):
        for email, clase in zip(emails, etiquetas):
            self.emails_por_clase[clase] += 1
            for palabra in email.lower().split():
                self.veces_palabra[clase][palabra] += 1
                self.palabras_por_clase[clase] += 1
                self.vocabulario.add(palabra)

    def prob_palabra(self, palabra, clase):
        # Contar y sumar 'suavizado' para que ninguna palabra tenga probabilidad 0
        veces = self.veces_palabra[clase].get(palabra, 0)
        total = self.palabras_por_clase[clase]
        return (veces + self.suavizado) / (total + self.suavizado * len(self.vocabulario))

    def puntuaciones(self, email):
        total_emails = sum(self.emails_por_clase.values())
        resultado = {}
        for clase in self.emails_por_clase:
            puntos = math.log(self.emails_por_clase[clase] / total_emails)       # el prior
            for palabra in email.lower().split():
                puntos += math.log(self.prob_palabra(palabra, clase))           # cada palabra suma su prueba
            resultado[clase] = puntos
        return resultado

    def predecir(self, email):
        p = self.puntuaciones(email)
        return max(p, key=p.get)


emails = [
    "gana dinero gratis ahora",
    "lotería gratis billete ganador",
    "reclama tu premio hoy gratis",
    "oferta urgente dinero gratis",
    "enhorabuena has ganado gratis",
    "reunión mañana a mediodía",
    "te adjunto el informe del proyecto",
    "podemos hacer una llamada",
    "revisión del informe trimestral",
    "comemos el jueves",
    "notas de la reunión de equipo",
    "revisa el pull request por favor",
]
etiquetas = ["spam"] * 5 + ["normal"] * 7

filtro = NaiveBayes()
filtro.entrenar(emails, etiquetas)

# --- 1. Clasificar emails que nunca ha visto ---
print("Emails nuevos:")
for email in ["dinero gratis para ti",
              "la reunión pasa al viernes",
              "has ganado un premio gratis",
              "revisa el informe adjunto por favor",
              "reunión gratis"]:
    print(f"  {email:38} -> {filtro.predecir(email)}")

# --- 2. ¿Cuánto 'apunta' cada palabra a spam? ---
print("\nCuántas veces más típica es cada palabra en spam que en normal:")
for palabra in ["gratis", "dinero", "premio", "reunión", "informe"]:
    ratio = filtro.prob_palabra(palabra, "spam") / filtro.prob_palabra(palabra, "normal")
    print(f"  {palabra:10} x{ratio:.2f}")

# --- 3. Sin suavizado: una palabra nunca vista en spam lo bloquea todo ---
sin_suavizar = NaiveBayes(suavizado=0)
sin_suavizar.entrenar(emails, etiquetas)
email = "gratis gratis dinero premio reunión"
print(f"\nSin suavizado, '{email}':")
try:
    print("  ->", sin_suavizar.predecir(email))
except ValueError as e:
    print(f"  ERROR: {e}  ('reunión' nunca salió en spam: probabilidad 0 y log(0))")
print(f"Con suavizado: -> {filtro.predecir(email)}")