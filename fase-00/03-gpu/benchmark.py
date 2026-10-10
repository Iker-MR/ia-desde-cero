import time
import torch

# 1. ¿Qué hardware tengo?
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")

# 2. Dos matrices aleatorias de 5000 x 5000 (25 millones de números cada una)
N = 5000
a = torch.randn(N, N)
b = torch.randn(N, N)

# 3. Multiplicación en CPU
inicio = time.perf_counter()
c = a @ b
cpu = time.perf_counter() - inicio
print(f"CPU: {cpu:.3f} s")

# 4. La misma multiplicación en GPU
if torch.cuda.is_available():
    a_gpu, b_gpu = a.cuda(), b.cuda()   # copiar a la VRAM (fuera del cronómetro)
    _ = a_gpu @ b_gpu                    # calentamiento: la 1ª llamada es más lenta
    torch.cuda.synchronize()

    inicio = time.perf_counter()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()             # esperar a que la GPU termine de verdad
    gpu = time.perf_counter() - inicio

    print(f"GPU: {gpu:.3f} s")
    print(f"La GPU es {cpu / gpu:.0f}x más rápida")