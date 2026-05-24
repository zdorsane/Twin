"""
================================================================================
  VISUALISATION DES RÉSULTATS — Bi-Int Digital Twin
  Données RÉELLES depuis CSV (pas de données simulées)
================================================================================
  Exécuter : python3 visualize_results.py
  Sortie   : figures/ (PNG sauvegardés)
  Ouvrir   : explorer.exe $(wslpath -w ~/Twin/figures)
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import io

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED, AllChem, rdMolDescriptors, DataStructs
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

print("=" * 65)
print("  VISUALISATION DES RÉSULTATS — Bi-Int Digital Twin")
print("=" * 65)

# ── Chargement des données réelles ───────────────────────────────────────────

df_cands = pd.read_csv("graphga_top_candidates.csv")
df_cands.columns = df_cands.columns.str.strip()

df_dqn = pd.read_csv("Dataset/brics_dqn_results.csv")
df_dqn.columns = df_dqn.columns.str.strip()

df_train = pd.read_csv("logs/run_gpu_main/training_log.csv")
df_train.columns = df_train.columns.str.strip()

# ── Calcul Lipinski sur les candidats réels ───────────────────────────────────
def lipinski_props(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return {
        "mw":   Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd":  rdMolDescriptors.CalcNumHBD(mol),
        "hba":  rdMolDescriptors.CalcNumHBA(mol),
        "qed":  QED.qed(mol),
    }

props_list = [lipinski_props(s) for s in df_cands["smiles"]]
props_list = [p for p in props_list if p is not None]
df_props = pd.DataFrame(props_list)

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 1 : Structures 2D des molécules (grille 2×5)
# ═══════════════════════════════════════════════════════════════════════
print("\n[1/5] Structures 2D des molécules générées (RDKit Draw)...")

mols, legends = [], []
for _, row in df_cands.iterrows():
    mol = Chem.MolFromSmiles(row["smiles"])
    if mol:
        rdDepictor.Compute2DCoords(mol)
        mols.append(mol)
        legends.append(
            f"#{int(row['rank'])}  QED={row['qed']:.3f}\n"
            f"MW={row['mw']:.0f}  LogP={row['logp']:.2f}"
        )

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=5,
    subImgSize=(360, 290),
    legends=legends,
    returnPNG=False,
)
img.save("figures/01_molecular_structures.png")
print("   ✓ figures/01_molecular_structures.png")

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 2 : Courbes train/val RMSE du QSAR (données réelles)
# ═══════════════════════════════════════════════════════════════════════
print("\n[2/5] Courbes QSAR train/val RMSE...")

epochs   = df_train["epoch"].tolist()
tr_rmse  = df_train["train_rmse"].tolist()
val_rmse = df_train["val_rmse"].tolist()
pearson  = df_train["pearson_r"].tolist()
kl_loss  = df_train["kl_loss"].tolist()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Bi-Int Digital Twin — Courbes d'Entraînement QSAR (réel)", fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(epochs, tr_rmse,  'b-o', label='Train RMSE', linewidth=2, markersize=6)
ax.plot(epochs, val_rmse, 'r--s', label='Val RMSE',  linewidth=2, markersize=6)
ax.fill_between(epochs, tr_rmse, val_rmse, alpha=0.12, color='purple')
ax.set_xlabel("Époque"); ax.set_ylabel("RMSE (log µM)")
ax.set_title("Convergence RMSE")
ax.legend(); ax.grid(True, alpha=0.3)
for ep, tr, va in zip(epochs, tr_rmse, val_rmse):
    ax.annotate(f"{tr:.3f}", (ep, tr), textcoords="offset points", xytext=(0, 6),
                fontsize=7, color='blue', ha='center')
    ax.annotate(f"{va:.3f}", (ep, va), textcoords="offset points", xytext=(0, -12),
                fontsize=7, color='red', ha='center')

ax = axes[1]
ax.plot(epochs, pearson, 'g-^', linewidth=2, markersize=6, color='darkorange')
ax.set_xlabel("Époque"); ax.set_ylabel("Pearson r")
ax.set_title("Corrélation Pearson (validation)")
ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
for ep, p in zip(epochs, pearson):
    ax.annotate(f"{p:.3f}", (ep, p), textcoords="offset points", xytext=(0, 6),
                fontsize=7, ha='center')

ax = axes[2]
ax.plot(epochs, kl_loss, 'g-^', linewidth=2, markersize=6)
ax.fill_between(epochs, kl_loss, alpha=0.2, color='green')
ax.set_xlabel("Époque"); ax.set_ylabel("KL Divergence")
ax.set_title("KL Loss (Régularisation VAE)"); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("figures/02_training_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ figures/02_training_curves.png")

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 3 : Courbes de reward DQN par épisode (données réelles)
# ═══════════════════════════════════════════════════════════════════════
print("\n[3/5] Courbes de reward DQN...")

valid_mask   = df_dqn["valid"].astype(str).str.lower() == "true"
rewards_all  = df_dqn["reward"].values
episodes_all = df_dqn["episode"].values

# Rolling mean sur 50 épisodes
window = 50
rewards_series = pd.Series(rewards_all)
rolling_mean = rewards_series.rolling(window, min_periods=1).mean().values

valid_ep  = episodes_all[valid_mask]
valid_rew = rewards_all[valid_mask]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("BRICS-DQN — Reward par Épisode (5 000 épisodes réels)", fontsize=13, fontweight='bold')

ax = axes[0]
ax.scatter(episodes_all[valid_mask],  rewards_all[valid_mask],
           s=2, alpha=0.25, color='steelblue', label='Valide')
ax.scatter(episodes_all[~valid_mask], rewards_all[~valid_mask],
           s=2, alpha=0.15, color='salmon',    label='Invalide')
ax.plot(episodes_all, rolling_mean, 'k-', linewidth=1.8,
        label=f'Moyenne glissante (w={window})', zorder=5)
ax.set_xlabel("Épisode"); ax.set_ylabel("Reward")
ax.set_title("Reward DQN (brut + moyenne glissante)")
ax.legend(markerscale=4, fontsize=8); ax.grid(True, alpha=0.2)
ax.annotate(f"Best: {rewards_all.max():.3f}",
            xy=(episodes_all[rewards_all.argmax()], rewards_all.max()),
            xytext=(20, -20), textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle='->', color='red'), color='red')

ax = axes[1]
bin_size = 100
n_bins = len(rewards_all) // bin_size
bin_means = [rewards_all[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)]
bin_stds  = [rewards_all[i*bin_size:(i+1)*bin_size].std()  for i in range(n_bins)]
bin_x     = [(i + 0.5) * bin_size for i in range(n_bins)]
ax.plot(bin_x, bin_means, 'b-o', linewidth=2, markersize=4, label='Reward moyen / 100 ep')
ax.fill_between(bin_x,
                [m - s for m, s in zip(bin_means, bin_stds)],
                [m + s for m, s in zip(bin_means, bin_stds)],
                alpha=0.2, color='blue', label='± 1 std')
ax.set_xlabel("Épisode"); ax.set_ylabel("Reward moyen")
ax.set_title("Progression par blocs de 100 épisodes")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2]
valid_pct = [valid_mask.values[i*bin_size:(i+1)*bin_size].mean()*100 for i in range(n_bins)]
ax.plot(bin_x, valid_pct, 'g-^', linewidth=2, markersize=4)
ax.fill_between(bin_x, valid_pct, alpha=0.2, color='green')
ax.set_xlabel("Épisode"); ax.set_ylabel("% Molécules valides")
ax.set_title("Taux de validité SMILES"); ax.set_ylim(0, 105)
ax.axhline(valid_mask.mean()*100, color='red', linestyle='--', linewidth=1.5,
           label=f"Moy. globale = {valid_mask.mean()*100:.1f}%")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("figures/03_dqn_reward.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ figures/03_dqn_reward.png")

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 4 : Distribution QED / Lipinski des candidats générés
# ═══════════════════════════════════════════════════════════════════════
print("\n[4/5] Distribution QED / Lipinski...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Distribution QED & Propriétés Lipinski — Candidats GraphGA", fontsize=13, fontweight='bold')

qeds  = df_props["qed"].values
mws   = df_props["mw"].values
logps = df_props["logp"].values
hbds  = df_props["hbd"].values
hbas  = df_props["hba"].values
labels_bar = [f"#{i+1}" for i in range(len(df_cands))]

# QED barres
ax = axes[0, 0]
colors = ['gold' if q >= 0.9 else 'steelblue' if q >= 0.8 else 'lightcoral' for q in qeds]
bars = ax.bar(labels_bar, qeds, color=colors, edgecolor='black', alpha=0.85)
ax.axhline(0.7, color='red',  linestyle='--', linewidth=1.5, label='Drug-like ≥ 0.7')
ax.axhline(0.9, color='gold', linestyle=':',  linewidth=1.5, label='Excellent ≥ 0.9')
ax.set_ylabel("QED Score"); ax.set_title("Drug-Likeness (QED)")
ax.legend(fontsize=8); ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, qeds):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
            ha='center', va='bottom', fontsize=7)

# MW
ax = axes[0, 1]
ax.bar(labels_bar, mws, color='mediumseagreen', edgecolor='black', alpha=0.8)
ax.axhline(500, color='red', linestyle='--', linewidth=1.5, label='Lipinski MW≤500 Da')
ax.set_ylabel("Masse Moléculaire (Da)"); ax.set_title("Masse Moléculaire")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(mws):
    ax.text(i, v + 3, f"{v:.0f}", ha='center', va='bottom', fontsize=7)

# LogP
ax = axes[0, 2]
colors_logp = ['steelblue' if l <= 5 else 'salmon' for l in logps]
ax.bar(labels_bar, logps, color=colors_logp, edgecolor='black', alpha=0.8)
ax.axhline(5, color='red',   linestyle='--', linewidth=1.5, label='Lipinski LogP≤5')
ax.axhline(0, color='green', linestyle=':',  linewidth=1.0)
ax.set_ylabel("LogP"); ax.set_title("LogP (lipophilicité)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

# HBD & HBA
ax = axes[1, 0]
x = np.arange(len(labels_bar)); w = 0.35
ax.bar(x - w/2, hbds, w, label='HBD', color='cornflowerblue', edgecolor='black', alpha=0.8)
ax.bar(x + w/2, hbas, w, label='HBA', color='coral',          edgecolor='black', alpha=0.8)
ax.axhline(5,  color='blue', linestyle='--', linewidth=1.2, label='Lipinski HBD≤5')
ax.axhline(10, color='red',  linestyle=':',  linewidth=1.2, label='Lipinski HBA≤10')
ax.set_xticks(x); ax.set_xticklabels(labels_bar)
ax.set_ylabel("Nombre"); ax.set_title("Donneurs / Accepteurs H")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

# Conformité Lipinski (heatmap)
ax = axes[1, 1]
criteres = ['MW≤500', 'LogP≤5', 'HBD≤5', 'HBA≤10', 'QED≥0.7']
lip_matrix = np.array([
    [int(mws[i] <= 500), int(logps[i] <= 5), int(hbds[i] <= 5),
     int(hbas[i] <= 10), int(qeds[i] >= 0.7)]
    for i in range(len(df_props))
])
ax.imshow(lip_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(5)); ax.set_xticklabels(criteres, fontsize=8)
ax.set_yticks(range(len(labels_bar))); ax.set_yticklabels(labels_bar, fontsize=8)
ax.set_title("Conformité Règles de Lipinski\n(Vert=OK, Rouge=Violation)")
for i in range(len(df_props)):
    for j in range(5):
        ax.text(j, i, "✓" if lip_matrix[i, j] else "✗",
                ha='center', va='center', fontsize=10,
                color='darkgreen' if lip_matrix[i, j] else 'darkred')

# QED vs Score composite scatter
ax = axes[1, 2]
sc = ax.scatter(qeds, df_cands["composite"].values,
                c=mws, cmap='plasma', s=80, edgecolors='black', linewidth=0.5, zorder=3)
plt.colorbar(sc, ax=ax, label='MW (Da)', shrink=0.85)
for i in range(len(df_cands)):
    ax.annotate(f"#{i+1}", (qeds[i], df_cands['composite'].iloc[i]),
                textcoords="offset points", xytext=(4, 2), fontsize=7)
ax.set_xlabel("QED"); ax.set_ylabel("Score composite")
ax.set_title("QED vs Score composite (coloré par MW)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("figures/04_qed_lipinski.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ figures/04_qed_lipinski.png")

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 5 : Dashboard récapitulatif
# ═══════════════════════════════════════════════════════════════════════
print("\n[5/5] Dashboard récapitulatif...")

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor('#f8f9fa')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)
fig.suptitle("Bi-Int Digital Twin — Dashboard Résultats (données réelles)", fontsize=15,
             fontweight='bold', y=0.98)

# Meilleur candidat (structure 2D)
ax_mol = fig.add_subplot(gs[0:2, 0:2])
best_smi = df_cands.iloc[0]["smiles"]
best_mol = Chem.MolFromSmiles(best_smi)
rdDepictor.Compute2DCoords(best_mol)
drawer = rdMolDraw2D.MolDraw2DCairo(420, 320)
drawer.drawOptions().addAtomIndices = False
drawer.DrawMolecule(best_mol)
drawer.FinishDrawing()
bio = io.BytesIO(drawer.GetDrawingText())
img_mol = Image.open(bio)
ax_mol.imshow(img_mol)
ax_mol.axis('off')
ax_mol.set_title(
    f"Meilleur Candidat #1\n"
    f"QED={df_cands.iloc[0]['qed']:.3f}  Score={df_cands.iloc[0]['composite']:.3f}  "
    f"MW={df_cands.iloc[0]['mw']:.0f} Da",
    fontsize=10, fontweight='bold', color='darkblue'
)
rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                       linewidth=2, edgecolor='gold', facecolor='none',
                       transform=ax_mol.transAxes)
ax_mol.add_patch(rect)

# Tableau métriques réelles
ax_tab = fig.add_subplot(gs[0, 2:4])
ax_tab.axis('off')
valid_rate = (df_dqn["valid"].astype(str).str.lower() == "true").mean() * 100
best_reward = df_dqn["reward"].max()
table_data = [
    ["Métrique", "Valeur"],
    ["Époques QSAR complétées", str(len(df_train))],
    [f"Train RMSE (ep {len(df_train)})", f"{df_train['train_rmse'].iloc[-1]:.4f} log µM"],
    [f"Val RMSE   (ep {len(df_train)})", f"{df_train['val_rmse'].iloc[-1]:.4f} log µM"],
    [f"Pearson r  (ep {len(df_train)})", f"{df_train['pearson_r'].iloc[-1]:.4f}"],
    ["Épisodes DQN",        f"{len(df_dqn):,}"],
    ["Meilleur reward DQN", f"{best_reward:.3f}"],
    ["Validité SMILES",     f"{valid_rate:.1f}%"],
    ["Candidats GraphGA",   f"{len(df_cands)} (tous valides)"],
    ["QED moyen",           f"{qeds.mean():.3f}"],
    ["MW moyen",            f"{mws.mean():.1f} Da"],
]
table = ax_tab.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(8.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#ecf0f1')
    cell.set_edgecolor('#bdc3c7')
ax_tab.set_title("Métriques Réelles du Modèle", fontweight='bold', fontsize=10)

# Mini courbe RMSE
ax_rmse = fig.add_subplot(gs[1, 2])
ax_rmse.plot(epochs, tr_rmse,  'b-o', markersize=4, linewidth=2, label='Train')
ax_rmse.plot(epochs, val_rmse, 'r--s', markersize=4, linewidth=2, label='Val')
ax_rmse.set_title("RMSE QSAR", fontsize=9); ax_rmse.legend(fontsize=7)
ax_rmse.set_xlabel("Époque", fontsize=8); ax_rmse.grid(True, alpha=0.3)
ax_rmse.tick_params(labelsize=7)

# Mini courbe Pearson
ax_pear = fig.add_subplot(gs[1, 3])
ax_pear.plot(epochs, pearson, '-^', markersize=4, linewidth=2, color='darkorange')
ax_pear.set_title("Pearson r (val)", fontsize=9)
ax_pear.set_xlabel("Époque", fontsize=8); ax_pear.set_ylabel("r", fontsize=8)
ax_pear.set_ylim(0, 1); ax_pear.grid(True, alpha=0.3)
ax_pear.tick_params(labelsize=7)

# DQN reward rolling (bas gauche)
ax_dqn = fig.add_subplot(gs[2, 0:2])
rolling_mean_s = pd.Series(rewards_all).rolling(window, min_periods=1).mean().values
ax_dqn.scatter(episodes_all[valid_mask], rewards_all[valid_mask],
               s=1, alpha=0.2, color='steelblue')
ax_dqn.plot(episodes_all, rolling_mean_s, 'k-', linewidth=1.5,
            label=f'Rolling mean (w={window})', zorder=5)
ax_dqn.set_xlabel("Épisode", fontsize=8); ax_dqn.set_ylabel("Reward", fontsize=8)
ax_dqn.set_title("DQN Reward (5 000 épisodes)", fontsize=9)
ax_dqn.legend(fontsize=7); ax_dqn.grid(True, alpha=0.2)
ax_dqn.tick_params(labelsize=7)

# QED barres (bas droite)
ax_qed = fig.add_subplot(gs[2, 2:4])
colors_b = ['gold' if q >= 0.9 else 'steelblue' if q >= 0.8 else 'lightcoral' for q in qeds]
ax_qed.bar(labels_bar, qeds, color=colors_b, edgecolor='black', alpha=0.85)
ax_qed.axhline(0.7, color='red', linestyle='--', linewidth=1.5, label='Drug-like ≥ 0.7')
ax_qed.axhline(0.9, color='gold', linestyle=':', linewidth=1.5, label='Excellent ≥ 0.9')
ax_qed.set_ylabel("QED", fontsize=8); ax_qed.set_title("QED des candidats", fontsize=9)
ax_qed.legend(fontsize=7); ax_qed.set_ylim(0, 1.15)
ax_qed.tick_params(labelsize=7); ax_qed.grid(True, alpha=0.3, axis='y')

plt.savefig("figures/05_dashboard.png", dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()
print("   ✓ figures/05_dashboard.png")

# ═══════════════════════════════════════════════════════════════════════
#  RÉSUMÉ CONSOLE
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  RÉSULTATS NUMÉRIQUES (données réelles)")
print("=" * 65)
print(f"\n{'#':<4} {'QED':>6} {'SA':>6} {'MW':>8} {'LogP':>6} {'Score':>8}")
print("-" * 42)
for _, row in df_cands.iterrows():
    flag = " ← MEILLEUR" if row['rank'] == 1 else ""
    print(f"{int(row['rank']):<4} {row['qed']:>6.3f} {row['sa']:>6.3f} "
          f"{row['mw']:>8.1f} {row['logp']:>6.3f} {row['composite']:>8.3f}{flag}")

print(f"\nQSAR (époque {len(df_train)}):")
print(f"  Train RMSE : {df_train['train_rmse'].iloc[-1]:.4f} log µM")
print(f"  Val RMSE   : {df_train['val_rmse'].iloc[-1]:.4f} log µM")
print(f"  Pearson r  : {df_train['pearson_r'].iloc[-1]:.4f}")

print(f"\nDQN BRICS (5 000 épisodes):")
valid_rate = (df_dqn["valid"].astype(str).str.lower() == "true").mean()
print(f"  Validité SMILES : {valid_rate*100:.1f}%")
print(f"  Meilleur reward : {df_dqn['reward'].max():.3f}")
print(f"  Reward moyen    : {df_dqn['reward'].mean():.3f}")

print("\n" + "=" * 65)
print("  FICHIERS GÉNÉRÉS")
print("=" * 65)
for f in sorted(os.listdir("figures")):
    path = os.path.join("figures", f)
    size = os.path.getsize(path) // 1024
    print(f"  figures/{f:<42} {size:>5} Ko")

print("\n[OK] Toutes les figures sauvegardées dans figures/")
print("[→]  Pour les ouvrir depuis Windows :")
print("     explorer.exe $(wslpath -w ~/Twin/figures)")
