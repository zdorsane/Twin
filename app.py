"""
Twin — Demo Streamlit pour pitch startup
========================================
Lance avec : streamlit run app.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Twin — AI Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Linear/Vercel style ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── Global background ── */
.stApp { background: #080b12 !important; }
[data-testid="stAppViewContainer"] { background: #080b12 !important; }
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid rgba(0, 220, 255, 0.08) !important;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(0,220,255,0.12) !important; }

/* ── Radio buttons in sidebar ── */
[data-testid="stSidebar"] [data-baseweb="radio"] label {
    color: #8b949e !important;
    font-size: 0.9rem !important;
    padding: 4px 0 !important;
    transition: color 0.2s;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label:hover { color: #00dcff !important; }

/* ── Metric cards — glassmorphism ── */
.metric-card {
    background: linear-gradient(135deg, rgba(0,220,255,0.04) 0%, rgba(13,17,23,0.9) 100%);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 22px 24px;
    border: 1px solid rgba(0,220,255,0.15);
    margin-bottom: 14px;
    box-shadow: 0 0 24px rgba(0,220,255,0.04), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.3s, box-shadow 0.3s;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,220,255,0.4), transparent);
}
.metric-card:hover {
    border-color: rgba(0,220,255,0.35);
    box-shadow: 0 0 32px rgba(0,220,255,0.10);
}
.metric-card.warn {
    border-color: rgba(255,170,0,0.2);
    background: linear-gradient(135deg, rgba(255,170,0,0.04) 0%, rgba(13,17,23,0.9) 100%);
}
.metric-card.warn::before {
    background: linear-gradient(90deg, transparent, rgba(255,170,0,0.4), transparent);
}
.metric-card.danger {
    border-color: rgba(255,70,70,0.2);
    background: linear-gradient(135deg, rgba(255,70,70,0.04) 0%, rgba(13,17,23,0.9) 100%);
}
.metric-card.danger::before {
    background: linear-gradient(90deg, transparent, rgba(255,70,70,0.4), transparent);
}
.metric-card.blue {
    border-color: rgba(100,160,255,0.2);
    background: linear-gradient(135deg, rgba(100,160,255,0.04) 0%, rgba(13,17,23,0.9) 100%);
}
.metric-val {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #00dcff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-card.warn  .metric-val { background: linear-gradient(135deg,#fff 0%,#ffaa00 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.metric-card.danger .metric-val { background: linear-gradient(135deg,#fff 0%,#ff4646 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.metric-lbl {
    font-size: 0.8rem;
    color: #6e7681;
    margin-top: 6px;
    font-weight: 500;
    letter-spacing: 0.01em;
}

/* ── Alert badges ── */
.alert-reliable {
    background: linear-gradient(135deg, rgba(0,220,100,0.08), rgba(0,220,100,0.03));
    border: 1px solid rgba(0,220,100,0.3);
    border-radius: 10px; padding: 12px 16px;
    color: #3ddc84; font-weight: 600; font-size: 0.92rem;
    box-shadow: 0 0 16px rgba(0,220,100,0.08);
}
.alert-caution {
    background: linear-gradient(135deg, rgba(255,170,0,0.08), rgba(255,170,0,0.03));
    border: 1px solid rgba(255,170,0,0.3);
    border-radius: 10px; padding: 12px 16px;
    color: #ffaa00; font-weight: 600; font-size: 0.92rem;
    box-shadow: 0 0 16px rgba(255,170,0,0.08);
}
.alert-unreliable {
    background: linear-gradient(135deg, rgba(255,70,70,0.08), rgba(255,70,70,0.03));
    border: 1px solid rgba(255,70,70,0.3);
    border-radius: 10px; padding: 12px 16px;
    color: #ff6b6b; font-weight: 600; font-size: 0.92rem;
    box-shadow: 0 0 16px rgba(255,70,70,0.08);
}

/* ── Titles ── */
h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #ffffff 30%, #00dcff 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
h2, h3 {
    color: #e6edf3 !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00bcd4, #0077ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 20px !important;
    box-shadow: 0 0 20px rgba(0,188,212,0.3) !important;
    transition: all 0.2s !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    box-shadow: 0 0 32px rgba(0,188,212,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input, .stSelectbox > div > div {
    background: #161b22 !important;
    border: 1px solid rgba(0,220,255,0.15) !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(0,220,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,220,255,0.08) !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(0,220,255,0.1) !important;
    gap: 4px !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    color: #6e7681 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 8px 16px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #00dcff !important;
    border-bottom: 2px solid #00dcff !important;
    background: rgba(0,220,255,0.04) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,220,255,0.1) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(0,220,255,0.2), transparent) !important;
    margin: 20px 0 !important;
}

/* ── Info / warning boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
    background: rgba(13,17,23,0.8) !important;
    backdrop-filter: blur(8px) !important;
}

/* ── Caption ── */
.stCaption { color: #484f58 !important; font-size: 0.78rem !important; }

/* ── Metric widget ── */
[data-testid="stMetric"] {
    background: rgba(0,220,255,0.03);
    border: 1px solid rgba(0,220,255,0.1);
    border-radius: 12px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: rgba(0,220,255,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,220,255,0.4); }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Twin")
    st.markdown("*AI-powered drug response prediction*")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠  Accueil",
        "🔬  Prédiction IC50",
        "💊  Bibliothèque moléculaire",
        "📊  Dashboard performance",
        "⚠️  Fiabilité & alertes",
    ])
    st.markdown("---")
    st.markdown("**Modèle :** Bi-Int (GNN + VAE)")
    st.markdown("**Dataset :** CCLE 647 lignées · 201 drogues")
    st.markdown("**Candidats générés :** 60 (38 MedChem-clean)")
    st.markdown("---")
    st.caption("Prototype de recherche — validation expérimentale requise")


# ═══════════════════════════════════════════
# Helpers / loaders
# ═══════════════════════════════════════════

@st.cache_resource(show_spinner="Chargement du modèle Bi-Int…")
def load_model():
    try:
        import tensorflow as tf
        from fullPipeline import BiIntDigitalTwin, HP, generate_synthetic_ccle_batch
        hp_path = os.path.join(ROOT, "logs/ldo_checkpoint/hp_snapshot.json")
        weights = os.path.join(ROOT, "logs/ldo_checkpoint/biint_ic50_model.weights.h5")
        if os.path.exists(hp_path):
            with open(hp_path) as f:
                HP.update(json.load(f))
        model = BiIntDigitalTwin(HP)
        dummy = generate_synthetic_ccle_batch(batch_size=2)
        model(dummy[:-1], training=False)
        if os.path.exists(weights):
            model.load_weights(weights)
            return model, HP, True
        return model, HP, False
    except Exception:
        return None, {}, False


@st.cache_resource(show_spinner="Chargement données CCLE…")
def load_ccle_data():
    try:
        from _ccle_loader import load_ccle_cached
        gex, cna, mut, cells, genes, HP, smiles_map, ic50_df, drugs = \
            load_ccle_cached(os.path.join(ROOT, "logs/ldo_checkpoint/hp_snapshot.json"))
        return gex, cna, mut, cells, genes, smiles_map, ic50_df, drugs
    except Exception:
        return None, None, None, [], [], {}, None, []


@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def mol_image_b64(smiles):
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Draw, AllChem
        from io import BytesIO
        import base64
        RDLogger.DisableLog("rdApp.*")
        # Try with sanitize first, fallback without stereochemistry
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None
            Chem.SanitizeMol(mol, catchErrors=True)
        img = Draw.MolToImage(mol, size=(300, 220))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def tanimoto_alert(smiles, smiles_map):
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, DataStructs
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None
        fp_q = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        best_sim, best_drug = 0.0, "N/A"
        for drug, smi in smiles_map.items():
            if not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            sim = DataStructs.TanimotoSimilarity(
                fp_q, AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
            if sim > best_sim:
                best_sim, best_drug = sim, drug
        level = "RELIABLE" if best_sim >= 0.6 else ("CAUTION" if best_sim >= 0.4 else "UNRELIABLE")
        return round(best_sim, 4), best_drug, level
    except Exception:
        return None, None, None


def predict(model, HP, smiles, gex_v, mut_v, cnv_v):
    try:
        import tensorflow as tf
        from fullPipeline import BRICSMolecularFeaturizer
        feats = BRICSMolecularFeaturizer().featurize(smiles)
        if feats is None:
            return None, None
        atoms, adj = feats
        inp = (tf.constant(atoms[np.newaxis], dtype=tf.float32),
               tf.constant(adj[np.newaxis],   dtype=tf.float32),
               tf.constant(gex_v[np.newaxis],  dtype=tf.float32),
               tf.constant(mut_v[np.newaxis],  dtype=tf.float32),
               tf.constant(cnv_v[np.newaxis],  dtype=tf.float32))
        ic50_z = float(model(inp, training=False)[0].numpy().squeeze())
        mc_std = float(np.std([model(inp, training=True)[0].numpy().squeeze()
                                for _ in range(10)]))
        return ic50_z, mc_std
    except Exception:
        return None, None


# ═══════════════════════════════════════════
# PAGE 1 — Accueil
# ═══════════════════════════════════════════
if page == "🏠  Accueil":
    st.markdown("# 🧬 Twin — AI Drug Discovery Platform")
    st.markdown("### Prédiction de réponse aux drogues + génération de molécules de novo")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("647", "Lignées cellulaires (CCLE)", "blue"),
        ("201", "Drogues avec SMILES", "blue"),
        ("103 477", "Triplets IC50 valides", ""),
        ("60", "Candidats générés (38 MedChem-clean)", ""),
    ]
    for col, (val, lbl, cls) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f'<div class="metric-card {cls}"><div class="metric-val">{val}</div>'
                     f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Architecture Bi-Int")
        st.code("""
Drug SMILES ──► GNN (ChEMBL pré-entraîné)
                        │
                ┌───────┴────────┐
                │  Bi-Int (×4)   │──► MLP ──► IC50
                └───────┬────────┘
                        │
Omics ──► Quaternion VAE
  (978 GEx + 426 CNA + 735 Mutations)
  → vecteur latent z ∈ ℝ¹²⁸
        """, language="text")
        st.markdown("**9 255 070** paramètres · checkpoint sauvegardé · inférence déterministe")

    with col_r:
        st.markdown("### Performances QSAR")
        perf = pd.DataFrame({
            "Split": ["Random ⚠️", "Leave-Drug-Out ✓", "Leave-Drug-Out ✓", "Leave-Cell-Out"],
            "Modèle": ["Bi-Int", "XGBoost", "Bi-Int", "Bi-Int"],
            "Pearson r": [0.811, 0.367, 0.316, 0.766],
            "IC 95%": ["[0.736, 0.886]", "[0.338, 0.393]", "[0.287, 0.344]", "—"],
        })
        st.dataframe(perf, use_container_width=True, hide_index=True)
        st.caption("⚠️ Random = métrique optimiste (mêmes drogues train/val). "
                   "LDO = métrique honnête.")

    st.markdown("---")
    st.markdown("### Génération moléculaire")
    c1, c2, c3, c4 = st.columns(4)
    gen_metrics = [
        ("63%", "MedChem-clean (38/60)", ""),
        ("0.90", "Diversité interne (max=1)", ""),
        ("0.833", "QED moyen GraphGA top-10", ""),
        ("< 0.30", "Tanimoto max vs CCLE (nouveauté)", "warn"),
    ]
    for col, (val, lbl, cls) in zip([c1, c2, c3, c4], gen_metrics):
        col.markdown(f'<div class="metric-card {cls}"><div class="metric-val">{val}</div>'
                     f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.warning("**Disclaimer :** prototype de recherche. Les prédictions IC50 pour de nouvelles "
               "molécules sont hors distribution — validation expérimentale in vitro requise.")


# ═══════════════════════════════════════════
# PAGE 2 — Prédiction IC50
# ═══════════════════════════════════════════
elif page == "🔬  Prédiction IC50":
    st.markdown("# 🔬 Prédiction IC50")

    model, HP, model_ok = load_model()
    gex_mat, cna_mat, mut_mat, common_cells, top_genes, smiles_map, ic50_df, drugs_w_smi = load_ccle_data()

    col_in, col_out = st.columns([1, 1])

    EXAMPLES = {
        "Afatinib (EGFR inhibitor)": "CN(C)CCCOc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OC",
        "BRI-46 (généré — top candidat)": "O=S(=O)(c1ccc2ccccc2c1)N1CCNCC1",
        "BRI-12 (généré — SA=1.68)": "NS(=O)(=O)c1ccc(-c2cccc(O)c2)cc1",
        "Gra-1 (GraphGA #1, QED=0.872)": "CN1CCCN(C2CC2)CC(c2ccccc2NCCO)C1",
        "Imatinib (Gleevec — référence)": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    }

    with col_in:
        st.markdown("### Entrées")
        choice = st.selectbox("Choisir un exemple ou entrer un SMILES :",
                              ["— SMILES personnalisé —"] + list(EXAMPLES.keys()))
        default_smi = EXAMPLES.get(choice, "")
        smiles_input = st.text_input("SMILES :", value=default_smi)

        cell_options = sorted(common_cells) if common_cells else [
            "MCF7_BREAST", "A549_LUNG", "HCT116_LARGE_INTESTINE",
            "HELA_CERVIX", "PC3_PROSTATE", "HL60_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE"]
        cell_line = st.selectbox("Lignée cellulaire :", cell_options)
        run = st.button("🚀 Prédire IC50", type="primary", use_container_width=True)

    with col_out:
        st.markdown("### Résultats")
        if run and smiles_input:
            with st.spinner("Calcul…"):

                # Structure 2D
                img = mol_image_b64(smiles_input)
                if img:
                    st.markdown(
                        f'<img src="data:image/png;base64,{img}" '
                        f'style="border-radius:8px;width:100%;max-width:280px;"/>',
                        unsafe_allow_html=True)
                else:
                    st.error("SMILES invalide.")

                # Alerte Tanimoto
                st.markdown("**Domaine d'applicabilité**")
                if smiles_map:
                    sim, drug_ref, level = tanimoto_alert(smiles_input, smiles_map)
                    if level == "RELIABLE":
                        st.markdown(f'<div class="alert-reliable">🟢 FIABLE — '
                                    f'Tanimoto = {sim:.3f} vs {drug_ref}</div>',
                                    unsafe_allow_html=True)
                    elif level == "CAUTION":
                        st.markdown(f'<div class="alert-caution">🟡 PRUDENCE — '
                                    f'Tanimoto = {sim:.3f} vs {drug_ref}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-unreliable">🔴 HORS DOMAINE — '
                                    f'Tanimoto = {sim:.3f} vs {drug_ref} '
                                    f'— prédiction non fiable</div>',
                                    unsafe_allow_html=True)

                # Prédiction
                st.markdown("**IC50 prédit**")
                if model_ok and gex_mat is not None:
                    c2i = {c: i for i, c in enumerate(common_cells)}
                    ci = c2i.get(cell_line, 0)
                    ic50_z, mc_std = predict(model, HP, smiles_input,
                                             gex_mat[ci], mut_mat[ci], cna_mat[ci])
                    if ic50_z is not None:
                        ic50_um = float(np.expm1(max(ic50_z * 1.844 + 2.654, 0)))
                        ca, cb = st.columns(2)
                        ca.metric("IC50 (z-score)", f"{ic50_z:.3f}")
                        cb.metric("IC50 approx.", f"{ic50_um:.1f} µM")
                        st.markdown(f"**Incertitude MC Dropout** σ = {mc_std:.4f}")
                        st.progress(float(min(mc_std / 0.4, 1.0)))
                        if mc_std > 0.198:
                            st.warning("⚠️ Haute incertitude.")
                    else:
                        st.error("SMILES non featurisable par le GNN.")
                else:
                    # Fallback demo
                    st.info("Modèle non chargé — valeurs de démonstration.")
                    z = round(np.random.uniform(-1.5, 2.0), 3)
                    ca, cb = st.columns(2)
                    ca.metric("IC50 (z-score, DEMO)", f"{z:.3f}")
                    cb.metric("IC50 approx. (DEMO)", f"{np.expm1(max(z*1.844+2.654,0)):.1f} µM")

                st.caption("Modèle LDO r=0.316 — validation expérimentale obligatoire.")
        elif not run:
            st.info("Remplissez les entrées puis cliquez sur **Prédire IC50**.")


# ═══════════════════════════════════════════
# PAGE 3 — Bibliothèque moléculaire
# ═══════════════════════════════════════════
elif page == "💊  Bibliothèque moléculaire":
    st.markdown("# 💊 Bibliothèque moléculaire générée")

    df_cand = load_csv(os.path.join(ROOT, "graphga_top_candidates.csv"))
    df_val  = load_csv(os.path.join(ROOT, "Dataset/molecular_validation_report.csv"))

    tab1, tab2, tab3 = st.tabs(["Top candidats GraphGA", "Validation MedChem complète", "Visualisation QED vs SA"])

    with tab1:
        if not df_cand.empty:
            st.markdown(f"**{len(df_cand)} candidats** classés par score composite")
            disp = df_cand[["rank","smiles","qed","sa","mw","logp","composite"]].copy()
            disp.columns = ["Rang","SMILES","QED","SA","MW (Da)","LogP","Score"]
            for col in ["QED","SA","MW (Da)","LogP","Score"]:
                disp[col] = disp[col].round(3)
            st.dataframe(disp, use_container_width=True, hide_index=True)

            st.markdown("### Top-3 structures 2D")
            cols = st.columns(3)
            for i, (_, row) in enumerate(df_cand.head(3).iterrows()):
                with cols[i]:
                    img = mol_image_b64(row["smiles"])
                    if img:
                        st.markdown(f'<img src="data:image/png;base64,{img}" '
                                    f'style="width:100%;border-radius:8px;"/>',
                                    unsafe_allow_html=True)
                    st.markdown(f"**#{int(row['rank'])}** QED={row['qed']:.3f} · SA={row['sa']:.2f} · MW={row['mw']:.0f} Da")
                    st.caption(row["smiles"][:48] + "…")
        else:
            st.warning("graphga_top_candidates.csv non trouvé.")

    with tab2:
        if not df_val.empty:
            n_clean = int(df_val["medchem_clean"].sum()) if "medchem_clean" in df_val.columns else "—"
            avg_qed = df_val["qed_computed"].mean() if "qed_computed" in df_val.columns else 0
            avg_sa  = df_val["sa_score"].mean() if "sa_score" in df_val.columns else 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MedChem-clean", f"{n_clean}/{len(df_val)}")
            c2.metric("PAINS alerts", str(int(df_val["pains_flag"].sum())) if "pains_flag" in df_val.columns else "—")
            c3.metric("QED moyen", f"{avg_qed:.3f}")
            c4.metric("SA moyen", f"{avg_sa:.2f}")
            cols_show = [c for c in ["id","source","qed_computed","sa_score","pains_flag",
                                      "lipinski_pass","medchem_clean","quality_score",
                                      "max_tanimoto_ccle"] if c in df_val.columns]
            sort_col = "quality_score" if "quality_score" in df_val.columns else cols_show[0]
            st.dataframe(df_val[cols_show].sort_values(sort_col, ascending=False),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Dataset/molecular_validation_report.csv non trouvé.")

    with tab3:
        if not df_val.empty and "qed_computed" in df_val.columns:
            fig = px.scatter(
                df_val, x="sa_score", y="qed_computed",
                color="medchem_clean" if "medchem_clean" in df_val.columns else None,
                hover_data=["id"] if "id" in df_val.columns else None,
                color_discrete_map={True: "#4CAF50", False: "#F44336"},
                title="QED vs SA Score — vert = MedChem-clean",
                labels={"sa_score": "SA Score (1=facile à synthétiser)", "qed_computed": "QED"},
                template="plotly_dark"
            )
            fig.add_vline(x=3, line_dash="dash", line_color="#FF9800", annotation_text="SA=3")
            fig.add_hline(y=0.7, line_dash="dash", line_color="#2196F3", annotation_text="QED=0.7")
            st.plotly_chart(fig, use_container_width=True)
        elif not df_cand.empty:
            fig = px.bar(df_cand.head(10), x="rank", y="qed",
                         title="QED — top-10 GraphGA", template="plotly_dark",
                         color="qed", color_continuous_scale="Greens")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# PAGE 4 — Dashboard performance
# ═══════════════════════════════════════════
elif page == "📊  Dashboard performance":
    st.markdown("# 📊 Dashboard performance")

    df_bl  = load_csv(os.path.join(ROOT, "Dataset/baseline_results_with_CI.csv"))
    df_nc  = load_csv(os.path.join(ROOT, "Dataset/ncrna_biomarker_importance.csv"))
    df_cod = load_csv(os.path.join(ROOT, "Dataset/coding_biomarker_importance.csv"))

    tab1, tab2, tab3 = st.tabs(["Comparaison modèles", "Courbes entraînement", "Biomarqueurs"])

    with tab1:
        if not df_bl.empty:
            split_sel = st.selectbox("Split :", df_bl["Split"].unique().tolist())
            sub = df_bl[df_bl["Split"] == split_sel].sort_values("Pearson_r")
            colors = ["#4CAF50" if "Bi-Int" in m else "#2196F3" for m in sub["Model"]]
            fig = go.Figure(go.Bar(
                y=sub["Model"], x=sub["Pearson_r"], orientation="h",
                marker_color=colors,
                error_x=dict(type="data",
                             array=(sub["CI_high"] - sub["Pearson_r"]).clip(0),
                             arrayminus=(sub["Pearson_r"] - sub["CI_low"]).clip(0),
                             visible=True),
                text=sub["Pearson_r"].round(3), textposition="outside"
            ))
            fig.update_layout(title=f"Pearson r — {split_sel} (IC 95% bootstrap n=1000)",
                              xaxis_title="Pearson r", xaxis_range=[0, 1],
                              template="plotly_dark", height=360)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Vert = Bi-Int · Bleu = baselines classiques")
        else:
            st.warning("Dataset/baseline_results_with_CI.csv non trouvé.")

        # Pearson r par epoch
        log_r = os.path.join(ROOT, "logs/run_gpu_main/training_log.csv")
        log_l = os.path.join(ROOT, "logs/run_ldo/training_log.csv")
        if os.path.exists(log_r):
            dr = pd.read_csv(log_r)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=dr["epoch"], y=dr["pearson_r"],
                                      name="Random", mode="lines+markers",
                                      line=dict(color="#4CAF50", width=2)))
            if os.path.exists(log_l):
                dl = pd.read_csv(log_l)
                fig2.add_trace(go.Scatter(x=dl["epoch"], y=dl["pearson_r"],
                                          name="LDO", mode="lines+markers",
                                          line=dict(color="#FF9800", width=2)))
            fig2.update_layout(title="Pearson r par epoch", template="plotly_dark",
                               yaxis_range=[0, 1], xaxis_title="Epoch")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        log_r = os.path.join(ROOT, "logs/run_gpu_main/training_log.csv")
        if os.path.exists(log_r):
            dr = pd.read_csv(log_r)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dr["epoch"], y=dr["train_rmse"],
                                     name="Train RMSE", line=dict(color="#2196F3")))
            fig.add_trace(go.Scatter(x=dr["epoch"], y=dr["val_rmse"],
                                     name="Val RMSE", line=dict(color="#F44336")))
            fig.add_trace(go.Scatter(x=dr["epoch"], y=dr["pearson_r"],
                                     name="Pearson r", line=dict(color="#4CAF50"),
                                     yaxis="y2"))
            fig.update_layout(
                title="Courbes entraînement — split random",
                template="plotly_dark", xaxis_title="Epoch",
                yaxis=dict(title="RMSE"),
                yaxis2=dict(title="Pearson r", overlaying="y",
                            side="right", range=[0, 1]),
                legend=dict(orientation="h")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dr.round(4), use_container_width=True, hide_index=True)
        else:
            st.info("Logs non trouvés — lancer fullPipeline.py.")

    with tab3:
        c_nc, c_cod = st.columns(2)
        with c_nc:
            st.markdown("#### Top-15 ncRNA")
            if not df_nc.empty:
                top15 = df_nc.nsmallest(15, "rank_ncrna")
                role_col = {"oncogene": "#F44336", "suppressor": "#2196F3",
                            "unknown": "#546E7A"}
                fig = go.Figure(go.Bar(
                    y=top15["name"], x=top15["importance"], orientation="h",
                    marker_color=[role_col.get(r, "#546E7A") for r in top15["role"]],
                ))
                fig.update_layout(template="plotly_dark",
                                  yaxis=dict(autorange="reversed"), height=420,
                                  xaxis_title="|Gradient×Input|",
                                  title="Rouge=oncogène · Bleu=suppresseur")
                st.plotly_chart(fig, use_container_width=True)
                for tgt in ["H19", "GAS5"]:
                    row = df_nc[df_nc["name"] == tgt]
                    if not row.empty:
                        st.metric(f"{tgt} rang ncRNA",
                                  f"{int(row.iloc[0]['rank_ncrna'])}/76")
            else:
                st.info("Lancer scripts/ncrna_biomarker_analysis.py")

        with c_cod:
            st.markdown("#### Top-15 gènes codants")
            if not df_cod.empty:
                top15c = df_cod.head(15)
                fig = go.Figure(go.Bar(
                    y=top15c["name"], x=top15c["importance"], orientation="h",
                    marker_color=["#F44336" if km else "#546E7A"
                                  for km in top15c["is_known_marker"]],
                ))
                fig.update_layout(template="plotly_dark",
                                  yaxis=dict(autorange="reversed"), height=420,
                                  xaxis_title="|Gradient×Input|",
                                  title="Rouge = biomarqueur oncologique connu")
                st.plotly_chart(fig, use_container_width=True)
                n_known = int(top15c["is_known_marker"].sum())
                st.metric("Biomarqueurs connus top-15", f"{n_known}/15")
            else:
                st.info("Lancer scripts/coding_biomarker_analysis.py")


# ═══════════════════════════════════════════
# PAGE 5 — Fiabilité & alertes
# ═══════════════════════════════════════════
elif page == "⚠️  Fiabilité & alertes":
    st.markdown("# ⚠️ Fiabilité & alertes")
    st.markdown("Deux mécanismes complémentaires — à utiliser ensemble.")

    df_ad = load_csv(os.path.join(ROOT, "Dataset/applicability_domain.csv"))

    tab1, tab2 = st.tabs(["Domaine d'applicabilité (Tanimoto)", "Incertitude MC Dropout"])

    with tab1:
        st.markdown("""
| Seuil Tanimoto | Niveau | Signification |
|----------------|--------|---------------|
| ≥ 0.6 | 🟢 FIABLE | Analogue structurel d'une drogue d'entraînement |
| 0.4 – 0.6 | 🟡 PRUDENCE | Proximité partielle |
| < 0.4 | 🔴 HORS DOMAINE | Structure nouvelle — prédiction non fiable |
        """)

        if not df_ad.empty:
            val_df = df_ad[df_ad["split"] == "val"] if "split" in df_ad.columns else df_ad
            n_rel = int((val_df["alert"] == "RELIABLE").sum())
            n_cau = int((val_df["alert"] == "CAUTION").sum())
            n_unr = int((val_df["alert"] == "UNRELIABLE").sum())
            n_tot = len(val_df)
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card"><div class="metric-val">{n_rel}</div>'
                        f'<div class="metric-lbl">🟢 FIABLE ({100*n_rel/n_tot:.0f}%)</div></div>',
                        unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card warn"><div class="metric-val">{n_cau}</div>'
                        f'<div class="metric-lbl">🟡 PRUDENCE ({100*n_cau/n_tot:.0f}%)</div></div>',
                        unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card danger"><div class="metric-val">{n_unr}</div>'
                        f'<div class="metric-lbl">🔴 HORS DOMAINE ({100*n_unr/n_tot:.0f}%)</div></div>',
                        unsafe_allow_html=True)

            fig = go.Figure(go.Histogram(
                x=val_df["max_tanimoto"], nbinsx=25,
                marker_color="#2196F3", opacity=0.8))
            fig.add_vrect(x0=0, x1=0.4, fillcolor="#F44336", opacity=0.1,
                          annotation_text="HORS DOMAINE")
            fig.add_vrect(x0=0.4, x1=0.6, fillcolor="#FF9800", opacity=0.1,
                          annotation_text="PRUDENCE")
            fig.add_vrect(x0=0.6, x1=1.0, fillcolor="#4CAF50", opacity=0.1,
                          annotation_text="FIABLE")
            fig.add_vline(x=0.4, line_dash="dash", line_color="#FF9800")
            fig.add_vline(x=0.6, line_dash="dash", line_color="#4CAF50")
            fig.update_layout(
                title="Distribution Tanimoto — drogues de validation LDO",
                xaxis_title="Tanimoto max vs drogues d'entraînement",
                yaxis_title="Nombre de drogues", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("80 % hors domaine — résultat attendu en LDO (drogues nouvelles par construction).")
        else:
            st.info("Lancer scripts/applicability_domain.py")

    with tab2:
        st.markdown("""
**Méthode :** N=30 passes forward avec dropout actif.
σ élevé = modèle incertain sur cette prédiction.

| σ | Interprétation |
|---|----------------|
| < 0.15 | Confiant |
| 0.15 – 0.20 | Modéré |
| ≥ 0.20 | ⚠️ Haute incertitude |
        """)

        mc_path = os.path.join(ROOT, "Dataset/uncertainty_mc_dropout.csv")
        if os.path.exists(mc_path):
            df_mc = pd.read_csv(mc_path)
            threshold = 0.1975
            n_high = int((df_mc["alert"] == "HIGH_UNCERTAINTY").sum())
            n_tot  = len(df_mc)
            c1, c2 = st.columns(2)
            c1.metric("Seuil d'alerte (σ)", f"{threshold:.4f}")
            c2.metric("Haute incertitude", f"{n_high}/{n_tot} ({100*n_high/n_tot:.1f}%)")

            fig = go.Figure(go.Histogram(
                x=df_mc["ic50_std"], nbinsx=30,
                marker_color="#2196F3", opacity=0.8))
            fig.add_vline(x=threshold, line_dash="dash", line_color="#F44336",
                          annotation_text=f"Seuil = {threshold:.4f}")
            fig.update_layout(
                title=f"Distribution σ MC Dropout — {n_high}/{n_tot} au-dessus du seuil",
                xaxis_title="σ (N=30 passes)", yaxis_title="Paires",
                template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Combiner obligatoirement avec l'alerte Tanimoto — "
                       "un modèle peut être confiant ET hors domaine.")
        else:
            st.info("Lancer scripts/uncertainty_mc_dropout.py")
