"""
FastAPI deployment for Digital Twin IC50 prediction
Run: uvicorn api_server:app --reload
"""

from fastapi import FastAPI, BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from fullPipeline import BiIntDigitalTwin, BRICSMolecularFeaturizer, DigitalTwinInference, HP

app = FastAPI(title="Bi-Int Digital Twin API")

# Load model
model = BiIntDigitalTwin(HP)
featurizer = BRICSMolecularFeaturizer()
inference = DigitalTwinInference(model, featurizer)

class PredictRequest(BaseModel):
    smiles: str
    gene_expression: List[float]
    mutations: List[float]
    cnv: List[float]

class DrugScreenRequest(BaseModel):
    smiles_list: List[str]
    gene_expression: List[float]
    mutations: List[float]
    cnv: List[float]

class KORequest(BaseModel):
    smiles: str
    gene_expression: List[float]
    mutations: List[float]
    cnv: List[float]
    gene_indices: List[int]

@app.post("/predict")
async def predict_ic50(req: PredictRequest):
    """Predict IC50 for single drug-cell line pair"""
    try:
        gex = np.array(req.gene_expression[:HP['gex_dim']], dtype=np.float32)
        mut = np.array(req.mutations[:HP['mut_dim']], dtype=np.float32)
        cnv = np.array(req.cnv[:HP['cnv_dim']], dtype=np.float32)
        
        ic50 = inference.predict_ic50(req.smiles, gex, mut, cnv)
        return {"smiles": req.smiles, "ic50_log_um": float(ic50)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/screen")
async def screen_library(req: DrugScreenRequest):
    """Screen compound library"""
    try:
        gex = np.array(req.gene_expression[:HP['gex_dim']], dtype=np.float32)
        mut = np.array(req.mutations[:HP['mut_dim']], dtype=np.float32)
        cnv = np.array(req.cnv[:HP['cnv_dim']], dtype=np.float32)
        
        results = inference.screen_drug_library(req.smiles_list, gex, mut, cnv)
        return {
            "library_size": len(results),
            "predictions": [{"smiles": s, "ic50": ic50} for s, ic50 in results.items()]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/virtual_ko")
async def virtual_knockout(req: KORequest):
    """Simulate gene knockout effect on IC50"""
    try:
        gex = np.array(req.gene_expression[:HP['gex_dim']], dtype=np.float32)
        mut = np.array(req.mutations[:HP['mut_dim']], dtype=np.float32)
        cnv = np.array(req.cnv[:HP['cnv_dim']], dtype=np.float32)
        
        result = inference.virtual_gene_ko(req.smiles, gex, mut, cnv, req.gene_indices)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_params": model.count_params()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
