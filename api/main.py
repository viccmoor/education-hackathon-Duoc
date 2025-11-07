from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib, numpy as np, pandas as pd

app = FastAPI(title="API Riesgo Aprobación", version="1.0.0")

model = joblib.load("models/model_edu_lr.pkl")
feature_names = joblib.load("models/feature_names_edu.pkl")

class EduRequest(BaseModel):
    promedio: float = Field(..., ge=1.0, le=7.0)
    asistencia: float = Field(..., ge=0.0, le=100.0)  # porcentaje 0-100
    edad: Optional[int] = None
    sexo: Optional[str] = None
    asignatura: Optional[str] = None
    establecimiento: Optional[str] = None

class EduResponse(BaseModel):
    score: float
    riesgo: str
    threshold: float = 0.5

@app.get("/")
def root():
    return {"message": "API Educación /predict", "features": feature_names}

@app.post("/predict", response_model=EduResponse)
def predict(req: EduRequest):
    try:
        feats = {
            "prom_gral": float(req.promedio),
            "asistencia_pct": float(req.asistencia)/100.0 if req.asistencia > 1 else float(req.asistencia),
        }
        X = pd.DataFrame([feats])[feature_names]
        p = float(model.predict_proba(X)[0, 1])
        riesgo = "Alto" if p >= 0.5 else "Bajo"
        return EduResponse(score=p, riesgo=riesgo)
    except Exception as e:
        raise HTTPException(status_code=500, detail = str(e)) from e
