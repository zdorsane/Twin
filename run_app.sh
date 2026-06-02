#!/bin/bash
# Lancer l'interface Twin — Bi-Int Drug Response Predictor
cd "$(dirname "$0")"
source venv_tf/bin/activate
echo "============================================"
echo "  Twin — AI Drug Response Predictor"
echo "  http://localhost:8501"
echo "============================================"
streamlit run app.py --server.port 8501
