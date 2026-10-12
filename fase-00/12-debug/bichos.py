import torch


def debug_print(nombre, t):
    """Resumen de un tensor: lo que hay que mirar cuando algo va mal."""
    print(f"{nombre}: forma={tuple(t.shape)}, tipo={t.dtype}, dispositivo={t.device}, "
          f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
          f"¿NaN?={t.isnan().any().item()}, ¿Inf?={t.isinf().any().item()}")


# --- Bug 1: forma incorrecta (este SÍ da error) ---
x = torch.randn(4, 3)   # 4 ejemplos con 3 características cada uno
W = torch.randn(4, 2)   # pesos con la forma equivocada (debería ser 3 x 2)
try:
    y = x @ W
except RuntimeError as e:
    print("BUG 1 - error de forma:\n ", e)

# --- Bug 2: NaN silencioso (este NO da error) ---
probabilidades = torch.tensor([0.9, 0.1, 0.0])   # el modelo dio un 0 exacto
perdida = -torch.log(probabilidades)              # log(0) = -infinito
print("\nBUG 2 - pérdida:")
debug_print("perdida", perdida)

total = perdida.sum() - perdida.sum()             # infinito - infinito
print("total:", total.item())
print("total + 1:", (total + 1).item())           # el NaN lo contagia todo

# --- La corrección habitual: no dejar que llegue un 0 exacto al log ---
segura = -torch.log(probabilidades.clamp(min=1e-8))
print("\nCORREGIDO:")
debug_print("perdida", segura)