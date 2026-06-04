import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
w = os.path.join(ROOT, "logs/ldo_checkpoint/biint_ic50_model.weights.h5")
hp = os.path.join(ROOT, "logs/ldo_checkpoint/hp_snapshot.json")
print(f"ROOT    : {ROOT}")
print(f"weights : {w}")
print(f"EXISTS  : {os.path.exists(w)}")
print(f"hp.json : {os.path.exists(hp)}")
print(f"size    : {os.path.getsize(w)/1e6:.1f} MB" if os.path.exists(w) else "MISSING")
