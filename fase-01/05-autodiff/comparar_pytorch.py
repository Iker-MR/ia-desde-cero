import torch
from valor import Valor

# --- 1. Una neurona: y = relu(w1·x1 + w2·x2 + b) ---
datos = {"w1": 0.5, "w2": -1.5, "x1": 2.0, "x2": -1.0, "b": 0.3}

# Con tu clase Valor
v = {k: Valor(val) for k, val in datos.items()}
y = (v["w1"] * v["x1"] + v["w2"] * v["x2"] + v["b"]).relu()
y.backward()

# Con PyTorch
t = {k: torch.tensor(val, requires_grad=True) for k, val in datos.items()}
y_t = torch.relu(t["w1"] * t["x1"] + t["w2"] * t["x2"] + t["b"])
y_t.backward()

print(f"Neurona: y = {y.dato:.4f} (Valor) | {y_t.item():.4f} (PyTorch)\n")
print(f"{'gradiente':10} {'Valor':>10} {'PyTorch':>10}")
for k in datos:
    print(f"dy/d{k:6} {v[k].grad:10.4f} {t[k].grad.item():10.4f}")

# --- 2. La fórmula de la comprobación: tanh(x³ + 2x + 1) en x = 0.5 ---
xv = Valor(0.5)
(xv ** 3 + 2 * xv + 1).tanh().backward()

xt = torch.tensor(0.5, requires_grad=True)
torch.tanh(xt ** 3 + 2 * xt + 1).backward()

print(f"\ntanh(x³ + 2x + 1):  Valor = {xv.grad:.6f} | PyTorch = {xt.grad.item():.6f}")