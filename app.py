"""
Twin — Bi-Int Drug Response Predictor
Streamlit demo app for pitch / jury presentation.

Run:
    cd ~/Twin && source venv_tf/bin/activate
    streamlit run app.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Twin — AI Drug Response Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title  { font-size:2.4rem; font-weight:700; color:#1a1a2e; margin-bottom:0; }
  .sub-title   { font-size:1.1rem; color:#555; margin-top:0; margin-bottom:1.5rem; }
  .metric-card { background:#f8f9fa; border-radius:12px; padding:1rem 1.4rem;
                 border-left:4px solid #4f8ef7; margin-bottom:0.5rem; }
  .alert-green { background:#e8f5e9; border-radius:8px; padding:0.6rem 1rem;
                 color:#2e7d32; font-weight:600; }
  .alert-orange{ background:#fff3e0; border-radius:8px; padding:0.6rem 1rem;
                 color:#e65100; font-weight:600; }
  .alert-red   { background:#ffebee; border-radius:8px; padding:0.6rem 1rem;
                 color:#c62828; font-weight:600; }
  .disclaimer  { font-size:0.8rem; color:#888; font-style:italic; }
</style>
""", unsafe_allow_html=True)

# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du modèle Bi-Int…")
def load_model():
    import tensorflow as tf
    from fullPipeline import BiIntDigitalTwin, HP, generate_synthetic_ccle_batch
    hp_path = os.path.join(ROOT, "logs/ldo_checkpoint/hp_snapshot.json")
    w_path  = os.path.join(ROOT, "logs/ldo_checkpoint/biint_ic50_model.weights.h5")
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            HP.update(json.load(f))
    model = BiIntDigitalTwin(HP)
    dummy = generate_synthetic_ccle_batch(batch_size=2)
    model(dummy[:-1], training=False)
    if os.path.exists(w_path):
        model.load_weights(w_path)
    return model, HP

@st.cache_resource(show_spinner="Chargement des données CCLE…")
def load_ccle_data():
    from _ccle_loader import load_ccle_cached
    return load_ccle_cached()

@st.cache_data(show_spinner=False)
def load_candidates():
    p = os.path.join(ROOT, "graphga_top_candidates.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_validation():
    p = os.path.join(ROOT, "Dataset/molecular_validation_report.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_baselines():
    p = os.path.join(ROOT, "Dataset/baseline_results_with_CI.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_smiles_map():
    p = os.path.join(ROOT, "Dataset/ccle_drug_smiles.csv")
    df = pd.read_csv(p)
    return {r["drug_name"]: r["smiles"] for _, r in df.iterrows()
            if pd.notna(r.get("smiles")) and r["smiles"]}

# ── Prediction helpers ─────────────────────────────────────────────────────────
def tanimoto_alert(smiles: str, smiles_map: dict) -> tuple:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0, "N/A", "🔴 HORS DOMAINE"
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        best_sim, best_drug = 0.0, "N/A"
        for drug, smi in smiles_map.items():
            ref = Chem.MolFromSmiles(smi)
            if ref is None:
                continue
            fp2 = AllChem.GetMorganFingerprintAsBitVect(ref, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim > best_sim:
                best_sim, best_drug = sim, drug
        if best_sim >= 0.6:
            alert = "🟢 FIABLE"
        elif best_sim >= 0.4:
            alert = "🟡 PRUDENCE"
        else:
            alert = "🔴 HORS DOMAINE"
        return best_sim, best_drug, alert
    except Exception:
        return 0.0, "N/A", "🔴 HORS DOMAINE"


def predict_ic50(model, HP, smiles: str, gex: np.ndarray,
                 mut: np.ndarray, cnv: np.ndarray, n_mc: int = 20):
    import tensorflow as tf
    from fullPipeline import BRICSMolecularFeaturizer
    featurizer = BRICSMolecularFeaturizer()
    feats = featurizer.featurize(smiles)
    if feats is None:
        return None, None, None
    atoms, adj = feats
    drug_t = tf.constant(atoms[np.newaxis], dtype=tf.float32)
    adj_t  = tf.constant(adj[np.newaxis],   dtype=tf.float32)
    gex_t  = tf.constant(gex[np.newaxis],   dtype=tf.float32)
    mut_t  = tf.constant(mut[np.newaxis],   dtype=tf.float32)
    cnv_t  = tf.constant(cnv[np.newaxis],   dtype=tf.float32)
    inp = (drug_t, adj_t, gex_t, mut_t, cnv_t)
    # Deterministic prediction
    pred_det, _ = model(inp, training=False)
    ic50_mean = float(pred_det.numpy().squeeze())
    # MC Dropout uncertainty
    preds_mc = [float(model(inp, training=True)[0].numpy().squeeze()) for _ in range(n_mc)]
    ic50_std = float(np.std(preds_mc))
    ci_lo = float(np.percentile(preds_mc, 2.5))
    ci_hi = float(np.percentile(preds_mc, 97.5))
    return ic50_mean, ic50_std, (ci_lo, ci_hi)


def mol_image_html(smiles: str, size: int = 200) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit import RDLogger
        import base64, io
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        img = Draw.MolToImage(mol, size=(size, size))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/png;base64,{b64}" width="{size}"/>'
    except Exception:
        return ""

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.markdown("## 🧬 Twin")
st.sidebar.markdown("*AI Drug Response Predictor*")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "🔬 Prédiction IC50", "💊 Molécules générées",
     "📊 Performances", "⚠️ Fiabilité", "ℹ️ À propos"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='disclaimer'>⚠️ Prototype de recherche.<br>"
    "Validation expérimentale requise.</div>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.markdown("<div class='main-title'>🧬 Twin</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Prédicteur multimodal de réponse aux drogues "
        "sur lignées cancéreuses (CCLE) + générateur de molécules de novo</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modèle", "Bi-Int", "GNN + VAE + Attention")
    c2.metric("Dataset", "103 477 triplets", "647 lignées · 201 drogues")
    c3.metric("Pearson r (random)", "0.811", "split aléatoire")
    c4.metric("Candidats générés", "60 molécules", "38/60 MedChem-clean")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Comment ça marche")
        st.markdown("""
```
Structure chimique (SMILES)
        │
        ▼ GNN pré-entraîné ChEMBL
   Drug embeddings
        │
        ├──────────────────────────┐
        │                          ▼
        │         Profil omique de la lignée
        │         GEx (978) + CNA (426) + Mut (735)
        │                    │
        │                    ▼ VAE quaternionique
        │              z latent (128-d)
        │                    │
        └──────────┬──────────┘
                   ▼
         Bi-Int Blocks (×4)
         attention croisée
                   │
                   ▼
           IC50 prédit (log µM)
           + alerte fiabilité
```
        """)

    with col2:
        st.markdown("### Résultats clés")
        df_res = pd.DataFrame({
            "Split": ["Random", "Leave-Drug-Out", "Leave-Drug-Out", "Leave-Cell-Out"],
            "Modèle": ["Bi-Int", "XGBoost", "Bi-Int", "Bi-Int"],
            "Pearson r": [0.811, 0.367, 0.316, 0.766],
            "IC 95%": ["[0.736, 0.886]", "[0.338, 0.393]", "[0.287, 0.344]", "partiel"],
        })
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        st.info(
            "⚠️ Le split **Leave-Drug-Out** (r = 0.316) est la métrique honnête : "
            "les drogues de validation ne sont jamais vues à l'entraînement. "
            "XGBoost surpasse le modèle profond sur cette métrique."
        )

    st.markdown("---")
    st.markdown("### Pipeline en 3 phases")
    p1, p2, p3 = st.columns(3)
    p1.success("**Phase 1** — Entraînement & génération\nModèle Bi-Int + GraphGA + BRICS-DQN")
    p2.warning("**Phase 2** — Validation chimique\nSA score · PAINS · Brenk · Tanimoto · diversité 0.90")
    p3.info("**Phase 3** — Fiabilité\nGradient×Input · MC Dropout · Domaine applicabilité")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRÉDICTION IC50
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Prédiction IC50":
    st.markdown("## 🔬 Prédiction IC50")
    st.markdown("Entrez un SMILES et sélectionnez une lignée cellulaire pour prédire l'IC50.")

    # Load resources
    with st.spinner("Chargement du modèle…"):
        try:
            model, HP = load_model()
            gex_mat, cna_mat, mut_mat, common_cells, top_genes, _, smiles_map, ic50_df, drugs_w_smi = load_ccle_data()
            model_loaded = True
        except Exception as e:
            st.error(f"Erreur chargement modèle : {e}")
            model_loaded = False

    if not model_loaded:
        st.stop()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Entrées")

        input_mode = st.radio("Mode d'entrée", ["SMILES libre", "Drogue CCLE connue"], horizontal=True)

        if input_mode == "SMILES libre":
            smiles_input = st.text_input(
                "SMILES de la drogue",
                value="CN1CCCN(C2CC2)CC(c2ccccc2NCCO)C1",
                help="SMILES canonique RDKit",
            )
        else:
            drug_choice = st.selectbox("Drogue CCLE", sorted(drugs_w_smi))
            smiles_input = smiles_map.get(drug_choice, "")
            st.code(smiles_input, language=None)

        cell_choice = st.selectbox("Lignée cellulaire", sorted(common_cells), index=0)

        st.markdown("---")
        run_btn = st.button("🚀 Prédire IC50", type="primary", use_container_width=True)

    with col_right:
        st.markdown("### Structure moléculaire")
        if smiles_input:
            img_html = mol_image_html(smiles_input, size=220)
            if img_html:
                st.markdown(img_html, unsafe_allow_html=True)
            else:
                st.warning("SMILES invalide — structure non affichée.")

    if run_btn and smiles_input:
        st.markdown("---")
        st.markdown("### Résultats")

        with st.spinner("Calcul en cours…"):
            # Get omics for selected cell
            c2i = {c: i for i, c in enumerate(common_cells)}
            ci = c2i[cell_choice]
            gex_v = gex_mat[ci]
            mut_v = mut_mat[ci]
            cnv_v = cna_mat[ci]

            ic50_val, ic50_std, ci_bounds = predict_ic50(
                model, HP, smiles_input, gex_v, mut_v, cnv_v, n_mc=20
            )
            tan_sim, closest_drug, tan_alert = tanimoto_alert(smiles_input, smiles_map)

        if ic50_val is None:
            st.error("❌ SMILES invalide ou featurization échouée. Vérifiez la structure.")
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("IC50 prédit", f"{ic50_val:.3f}", "log µM (z-score)")
            r2.metric("Incertitude MC", f"σ = {ic50_std:.3f}", f"IC95% [{ci_bounds[0]:.3f}, {ci_bounds[1]:.3f}]")
            r3.metric("Tanimoto max", f"{tan_sim:.3f}", f"vs {closest_drug[:20]}")

            # Applicability alert
            if "🟢" in tan_alert:
                st.markdown(f"<div class='alert-green'>{tan_alert} — Drogue dans le domaine d'applicabilité du modèle</div>", unsafe_allow_html=True)
            elif "🟡" in tan_alert:
                st.markdown(f"<div class='alert-orange'>{tan_alert} — Prédiction à interpréter avec prudence</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='alert-red'>{tan_alert} — Structure hors domaine d'applicabilité, prédiction peu fiable</div>", unsafe_allow_html=True)

            # Uncertainty alert
            UNCERTAINTY_THRESHOLD = 0.1975
            if ic50_std > UNCERTAINTY_THRESHOLD:
                st.markdown(
                    f"<div class='alert-orange'>⚠️ HAUTE INCERTITUDE MC Dropout (σ={ic50_std:.3f} > seuil {UNCERTAINTY_THRESHOLD})</div>",
                    unsafe_allow_html=True,
                )

            # IC50 gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ic50_val,
                title={"text": "IC50 prédit (log µM z-score)"},
                gauge={
                    "axis": {"range": [-3, 5]},
                    "bar": {"color": "#4f8ef7"},
                    "steps": [
                        {"range": [-3, 0], "color": "#c8e6c9"},
                        {"range": [0, 2],  "color": "#fff9c4"},
                        {"range": [2, 5],  "color": "#ffcdd2"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75,
                        "value": ic50_val,
                    },
                },
            ))
            fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                "<div class='disclaimer'>Les valeurs sont en z-score normalisé (mean=0, std=1 sur les IC50 CCLE log1p). "
                "Valeur négative = sensible, positive = résistant.</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MOLÉCULES GÉNÉRÉES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💊 Molécules générées":
    st.markdown("## 💊 Molécules générées de novo")
    st.markdown("Candidats produits par GraphGA et BRICS-DQN, classés par score qualité (IC50-agnostique).")

    df_cand = load_candidates()
    df_val  = load_validation()

    if df_cand.empty:
        st.warning("Fichier graphga_top_candidates.csv non trouvé.")
        st.stop()

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidats GraphGA", len(df_cand))
    if not df_val.empty:
        m2.metric("MedChem-clean", f"{df_val['medchem_clean'].sum()}/{len(df_val)}")
        m3.metric("QED moyen", f"{df_val['qed_computed'].mean():.3f}")
        m4.metric("Diversité interne", "0.90")
    else:
        m2.metric("Filtres PAINS", "0 alertes")
        m3.metric("QED moyen", f"{df_cand['qed'].mean():.3f}")
        m4.metric("Diversité", "0.90")

    st.markdown("---")

    # Top candidates table with structures
    st.markdown("### Top-10 candidats GraphGA")
    cols_show = ["rank", "qed", "mw", "logp", "composite"]
    display_df = df_cand[cols_show].rename(columns={
        "rank": "Rang", "qed": "QED", "mw": "MW (Da)",
        "logp": "LogP", "composite": "Score"
    })
    display_df["QED"]   = display_df["QED"].round(3)
    display_df["MW (Da)"] = display_df["MW (Da)"].round(1)
    display_df["LogP"]  = display_df["LogP"].round(2)
    display_df["Score"] = display_df["Score"].round(3)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Molecular structures grid
    st.markdown("### Structures 2D — Top 6")
    cols = st.columns(3)
    for i, (_, row) in enumerate(df_cand.head(6).iterrows()):
        with cols[i % 3]:
            img = mol_image_html(row["smiles"], size=180)
            if img:
                st.markdown(img, unsafe_allow_html=True)
            st.caption(f"**#{int(row['rank'])}** QED={row['qed']:.3f} | MW={row['mw']:.0f} Da | LogP={row['logp']:.2f}")

    # QED vs SA scatter
    if not df_val.empty and "qed_computed" in df_val.columns and "sa_score" in df_val.columns:
        st.markdown("---")
        st.markdown("### QED vs SA score — 60 candidats")
        fig = px.scatter(
            df_val,
            x="sa_score", y="qed_computed",
            color="medchem_clean",
            color_discrete_map={True: "#4caf50", False: "#f44336"},
            hover_data=["id", "source"],
            labels={"sa_score": "SA Score (1=facile, 10=difficile)",
                    "qed_computed": "QED (0-1)",
                    "medchem_clean": "MedChem OK"},
            title="Espace QED × SA — candidats générés (vert = MedChem-clean)",
        )
        fig.add_hline(y=0.7, line_dash="dash", line_color="grey",
                      annotation_text="QED = 0.7 (drug-like threshold)")
        fig.add_vline(x=5.0, line_dash="dash", line_color="grey",
                      annotation_text="SA = 5 (difficile)")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PERFORMANCES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Performances":
    st.markdown("## 📊 Performances du modèle")

    df_bl = load_baselines()

    if df_bl.empty:
        st.warning("Fichier baseline_results_with_CI.csv non trouvé.")
        st.stop()

    # Training curves
    st.markdown("### Courbes d'entraînement — split random")
    train_log = os.path.join(ROOT, "logs/run_gpu_main/training_log.csv")
    if os.path.exists(train_log):
        df_tr = pd.read_csv(train_log)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_tr["epoch"], y=df_tr["train_rmse"],
                                  mode="lines+markers", name="Train RMSE",
                                  line=dict(color="#2196F3", width=2)))
        fig.add_trace(go.Scatter(x=df_tr["epoch"], y=df_tr["val_rmse"],
                                  mode="lines+markers", name="Val RMSE",
                                  line=dict(color="#F44336", width=2)))
        fig.add_trace(go.Scatter(x=df_tr["epoch"], y=df_tr["pearson_r"],
                                  mode="lines+markers", name="Pearson r",
                                  line=dict(color="#4CAF50", width=2),
                                  yaxis="y2"))
        fig.update_layout(
            yaxis=dict(title="RMSE"),
            yaxis2=dict(title="Pearson r", overlaying="y", side="right", range=[0, 1]),
            legend=dict(x=0.02, y=0.98),
            height=350,
            title="Bi-Int — Random split (4 epochs)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Comparison bar chart per split
    st.markdown("### Pearson r par split et modèle")
    splits = df_bl["Split"].unique()
    split_choice = st.selectbox("Split", splits)
    sub = df_bl[df_bl["Split"] == split_choice].copy()
    sub = sub.sort_values("Pearson_r", ascending=True)

    fig2 = go.Figure()
    colors = ["#4f8ef7" if "Bi-Int" in m else "#90caf9" for m in sub["Model"]]
    fig2.add_trace(go.Bar(
        x=sub["Pearson_r"], y=sub["Model"],
        orientation="h",
        marker_color=colors,
        error_x=dict(
            type="data",
            symmetric=False,
            array=(sub["CI_high"] - sub["Pearson_r"]).clip(lower=0),
            arrayminus=(sub["Pearson_r"] - sub["CI_low"]).clip(lower=0),
        ),
    ))
    fig2.update_layout(
        xaxis=dict(title="Pearson r", range=[0, 1]),
        height=350,
        title=f"Pearson r — {split_choice} (barres = IC 95%)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Full table
    st.markdown("### Tableau complet")
    show_df = df_bl[["Model","Split","Pearson_r","CI_low","CI_high","RMSE"]].copy()
    show_df.columns = ["Modèle","Split","Pearson r","IC bas","IC haut","RMSE"]
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.info(
        "**Note :** Random r=0.811 est optimiste (même drogues train/val). "
        "Leave-Drug-Out est la métrique honnête de généralisation. "
        "XGBoost (r=0.367) surpasse Bi-Int (r=0.316) sur ce split."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FIABILITÉ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Fiabilité":
    st.markdown("## ⚠️ Alertes de fiabilité")
    st.markdown(
        "Deux mécanismes complémentaires signalent quand une prédiction est hors du domaine "
        "d'applicabilité ou incertaine."
    )

    tab1, tab2, tab3 = st.tabs(["🗺️ Domaine d'applicabilité", "🎲 Incertitude MC Dropout", "🧬 Biomarqueurs"])

    # ── Tab 1 : Tanimoto domain ──────────────────────────────────────────────
    with tab1:
        st.markdown("### Domaine d'applicabilité — Tanimoto (Morgan FP r=2, 2048 bits)")
        st.markdown(
            "Pour chaque drogue, on calcule la similarité Tanimoto maximale avec les drogues "
            "d'entraînement. En dessous de 0.4 = hors domaine."
        )

        ad_path = os.path.join(ROOT, "Dataset/applicability_domain.csv")
        if os.path.exists(ad_path):
            df_ad = pd.read_csv(ad_path)
            val_df = df_ad[df_ad["split"] == "val"]

            a1, a2, a3 = st.columns(3)
            n_rel = (val_df["alert"] == "RELIABLE").sum()
            n_cau = (val_df["alert"] == "CAUTION").sum()
            n_unr = (val_df["alert"] == "UNRELIABLE").sum()
            n_tot = len(val_df)

            a1.metric("🟢 FIABLE (≥0.6)",   f"{n_rel} / {n_tot}", f"{100*n_rel/n_tot:.0f}%")
            a2.metric("🟡 PRUDENCE (0.4–0.6)", f"{n_cau} / {n_tot}", f"{100*n_cau/n_tot:.0f}%")
            a3.metric("🔴 HORS DOMAINE (<0.4)", f"{n_unr} / {n_tot}", f"{100*n_unr/n_tot:.0f}%")

            fig_ad = px.histogram(
                df_ad, x="max_tanimoto", color="split",
                nbins=30, barmode="overlay",
                color_discrete_map={"train": "#90caf9", "val": "#ef9a9a"},
                labels={"max_tanimoto": "Tanimoto max vs drogues d'entraînement"},
                title="Distribution Tanimoto — train (bleu) vs validation (rouge)",
            )
            fig_ad.add_vline(x=0.6, line_dash="dash", line_color="green",
                             annotation_text="FIABLE ≥0.6")
            fig_ad.add_vline(x=0.4, line_dash="dash", line_color="orange",
                             annotation_text="PRUDENCE ≥0.4")
            fig_ad.update_layout(height=360)
            st.plotly_chart(fig_ad, use_container_width=True)
        else:
            st.warning("Dataset/applicability_domain.csv non trouvé.")

    # ── Tab 2 : MC Dropout ───────────────────────────────────────────────────
    with tab2:
        st.markdown("### Incertitude MC Dropout (N=30 passes stochastiques)")
        mc_path = os.path.join(ROOT, "Dataset/uncertainty_mc_dropout.csv")
        if os.path.exists(mc_path):
            df_mc = pd.read_csv(mc_path)
            THRESHOLD = 0.1975

            b1, b2, b3 = st.columns(3)
            n_high = (df_mc["alert"] == "HIGH_UNCERTAINTY").sum()
            b1.metric("Paires analysées", len(df_mc))
            b2.metric("Haute incertitude", f"{n_high} ({100*n_high/len(df_mc):.1f}%)")
            b3.metric("Seuil σ", f"{THRESHOLD:.4f}")

            fig_mc = px.histogram(
                df_mc, x="ic50_std",
                nbins=30, color="alert",
                color_discrete_map={"OK": "#4caf50", "HIGH_UNCERTAINTY": "#f44336"},
                labels={"ic50_std": "Écart-type MC Dropout (σ)", "alert": "Alerte"},
                title="Distribution de l'incertitude — σ par paire (drogue, lignée)",
            )
            fig_mc.add_vline(x=THRESHOLD, line_dash="dash", line_color="red",
                             annotation_text=f"Seuil = {THRESHOLD:.3f}")
            fig_mc.update_layout(height=360)
            st.plotly_chart(fig_mc, use_container_width=True)

            # Scatter IC50 true vs predicted
            fig_sc = px.scatter(
                df_mc, x="ic50_true", y="ic50_mean",
                color="alert",
                color_discrete_map={"OK": "#4caf50", "HIGH_UNCERTAINTY": "#f44336"},
                error_y=df_mc["ic50_std"],
                labels={"ic50_true": "IC50 réel (log µM z-score)",
                        "ic50_mean": "IC50 prédit",
                        "alert": "Alerte"},
                title="IC50 prédit vs réel avec barres d'incertitude",
            )
            fig_sc.add_shape(type="line", x0=-3, y0=-3, x1=5, y1=5,
                             line=dict(dash="dot", color="grey"))
            fig_sc.update_layout(height=400)
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.warning("Dataset/uncertainty_mc_dropout.csv non trouvé.")

    # ── Tab 3 : Biomarqueurs ─────────────────────────────────────────────────
    with tab3:
        st.markdown("### Biomarqueurs génomiques — Importance Gradient×Input")
        st.info(
            "⚠️ Calculés sur le checkpoint LDO (r=0.210). "
            "Recalcul sur le checkpoint random (r=0.811) en cours."
        )

        nc_path = os.path.join(ROOT, "Dataset/ncrna_biomarker_importance.csv")
        co_path = os.path.join(ROOT, "Dataset/coding_biomarker_importance.csv")

        col_nc, col_co = st.columns(2)

        with col_nc:
            st.markdown("#### Top-15 ncRNA")
            if os.path.exists(nc_path):
                df_nc = pd.read_csv(nc_path).head(15)
                colors_nc = ["#d62728" if r == "oncogene" else
                             "#1f77b4" if r == "suppressor" else "#aaaaaa"
                             for r in df_nc["role"]]
                fig_nc = go.Figure(go.Bar(
                    x=df_nc["importance"], y=df_nc["name"],
                    orientation="h", marker_color=colors_nc,
                ))
                fig_nc.update_layout(
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Importance |gradient×input|",
                    height=420,
                    margin=dict(l=10),
                )
                st.plotly_chart(fig_nc, use_container_width=True)
                st.caption("🔴 Oncogène · 🔵 Suppresseur · ⚫ Non documenté")
            else:
                st.warning("ncrna_biomarker_importance.csv non trouvé.")

        with col_co:
            st.markdown("#### Top-15 gènes codants")
            if os.path.exists(co_path):
                df_co = pd.read_csv(co_path).head(15)
                colors_co = ["#d62728" if km else "#aec7e8"
                             for km in df_co["is_known_marker"]]
                fig_co = go.Figure(go.Bar(
                    x=df_co["importance"], y=df_co["name"],
                    orientation="h", marker_color=colors_co,
                ))
                fig_co.update_layout(
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Importance |gradient×input|",
                    height=420,
                    margin=dict(l=10),
                )
                st.plotly_chart(fig_co, use_container_width=True)
                st.caption("🔴 Biomarqueur oncologique connu · 🔵 Gène codant non annoté")
            else:
                st.warning("coding_biomarker_importance.csv non trouvé.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — À PROPOS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ À propos":
    st.markdown("## ℹ️ À propos du projet Twin")

    st.markdown("""
### Architecture Bi-Int

| Composant | Détail |
|-----------|--------|
| Encodeur drogue | GNN (3 couches), pré-entraîné sur ChEMBL |
| Encodeur omique | VAE quaternionique, latent z ∈ ℝ¹²⁸ |
| Fusion | 4 blocs Bi-Int (row-cross + col-cross attention + triangular updates) |
| Output | MLP → IC50 log µM (z-score) |
| Paramètres | 9 255 070 |
| Inférence | Déterministe (µ du VAE, sans reparamétrisation) |

### Dataset

| Source | Taille | Preprocessing |
|--------|--------|---------------|
| CCLE Broad 2019 | 647 lignées · 266 drogues · 103 477 IC50 valides | log1p + z-score |
| Drogues avec SMILES | 201 / 266 (75.6%) | PubChem API lookup |
| Features GEx | 978 gènes top-variance | StandardScaler |
| Features CNA | 426 régions top-variance | StandardScaler |
| Features Mutations | 735 gènes top-mutés | Binaire (0/1) |

### Limitations

- **LDO r = 0.316 < XGBoost r = 0.367** : deep learning pas encore supérieur à l'échelle actuelle
- **80 % des nouvelles drogues hors domaine d'applicabilité** en LDO
- **65 / 266 drogues sans SMILES** : exclues de l'entraînement
- **MC Dropout trop confiant** (10% dropout insuffisant)
- **Biomarqueurs calculés sur LDO r=0.210** : à refaire sur checkpoint random

### Références

- Barretina et al., *Nature* 2012 — CCLE dataset
- Bickerton et al., *Nature Chemistry* 2012 — QED
- Gal & Ghahramani, *ICML* 2016 — MC Dropout
- Rogers & Hahn, *JCIM* 2010 — Morgan fingerprints
    """)

    st.warning(
        "**Prototype de recherche — pas un outil médical.**  "
        "Toute prédiction doit être validée expérimentalement avant usage clinique."
    )
