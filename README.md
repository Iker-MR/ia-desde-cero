# IA desde cero

Mi recorrido como desarrollador full stack aprendiendo ingeniería de IA desde los fundamentos, siguiendo el curso abierto [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch).

Implemento cada algoritmo a mano, con matemáticas y Python puro, y después lo contrasto con las librerías de producción (NumPy y PyTorch). Cada lección incluye mi código y mis notas, también los errores que cometí.

## Resultados destacados

| Lección | Qué construí | Resultado |
|---|---|---|
| [1.05 Autodiff](fase-01/05-autodiff) | Motor de backpropagation y red neuronal desde cero | La red `[2, 4, 1]` aprende XOR: la pérdida baja de 4,15 a 0,17 en 100 pasos. Los gradientes de la neurona de prueba coinciden con PyTorch |
| [1.08 Optimización](fase-01/08-optimizacion) | Descenso, momento y Adam sin librerías | Función de Rosenbrock: 18.599 pasos con descenso normal, 9.260 con momento y 767 con Adam |
| [1.07 Bayes](fase-01/07-bayes) | Filtro de spam Naive Bayes y test A/B bayesiano | Cada palabra actúa como multiplicador de la sospecha: «gratis» ×7,12, «reunión» ×0,40 |
| [1.09 Información](fase-01/09-informacion) | Entropía, entropía cruzada, KL y perplejidad | Con `y = x²` la correlación es 0,000 y la información mutua detecta la relación con 1,522 bits |
| [0.12 Depuración](fase-00/12-debug) | Detección de NaN silenciosos y análisis de rendimiento | El perfilado señaló la carga de datos como cuello de botella (0,412 s) |
| [0.04 APIs](fase-00/04-apis) | Llamadas a Claude con el SDK y con HTTP directo | Gestión de errores, tokens y claves fuera del código |

## Progreso

| Fase | Tema | Estado |
|---|---|---|
| 0 | Entorno, Git, GPU, APIs de LLM, datos, depuración y Python para IA | Completada |
| 1 | Fundamentos matemáticos | En curso, 9 lecciones terminadas |

**Lecciones de la fase 1:** álgebra lineal, operaciones con matrices, autovectores, cálculo, autodiff, probabilidad, Bayes, optimización y teoría de la información.

## Cómo está organizado

```text
fase-XX/
└── NN-tema/
    ├── *.py        código de la lección
    └── notas.txt   ideas, resultados y errores
```

## Ejecutar el código

```bash
git clone https://github.com/Iker-MR/ia-desde-cero.git
cd ia-desde-cero
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
python fase-01/08-optimizacion/optimizadores.py
```
