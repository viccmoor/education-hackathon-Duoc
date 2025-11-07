from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import pandas as pd
from pathlib import Path
import os
from contextlib import asynccontextmanager
import sys
from dotenv import load_dotenv
from openai import OpenAI
import json

# Configurar path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DesercionPredictor
from src.features import create_features

# Cargar variables de entorno
load_dotenv()

# Base absoluta del repo
BASE_DIR = Path(__file__).parent.parent.resolve()

# Variable global para el modelo
MODEL_PATH = BASE_DIR / "models/desercion_predictor.joblib"
model: Optional[DesercionPredictor] = None

THRESHOLD_PATH = BASE_DIR / "models/threshold.txt"
METRICS_PATH = BASE_DIR / "models/metrics.json"

_threshold_cache = {"value": 0.5, "mtime": None}

def current_threshold() -> float:
    try:
        st = THRESHOLD_PATH.stat()
        if _threshold_cache["mtime"] != st.st_mtime:
            _threshold_cache["value"] = float(THRESHOLD_PATH.read_text().strip())
            _threshold_cache["mtime"] = st.st_mtime
        return float(_threshold_cache["value"])
    except Exception:
        return 0.5

# Modelos Pydantic (ANTES de usar @app)
class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)

openai_client: Optional[OpenAI] = None
openai_key_cache: Optional[str] = None

def get_openai_client(force: bool = False) -> Optional[OpenAI]:
    global openai_client, openai_key_cache
    if force:
        openai_client = None
        openai_key_cache = None
    if openai_client is not None:
        return openai_client
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    if openai_key_cache == key and openai_client is not None:
        return openai_client
    try:
        openai_client = OpenAI(api_key=key)
        openai_key_cache = key
        print("✅ OpenAI client inicializado")
        return openai_client
    except Exception as e:
        print(f"⚠️ Error inicializando OpenAI client: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación (startup/shutdown)."""
    global model
    
    # Startup
    print("\n" + "="*60)
    print("🚀 INICIANDO API DE PREDICCIÓN DE DESERCIÓN")
    print("="*60)
    
    try:
        if MODEL_PATH.exists():
            model = DesercionPredictor.load(str(MODEL_PATH))
            print(f"✅ Modelo cargado desde {MODEL_PATH}")
            print(f"   Features: {len(model.feature_names)}")
        else:
            print(f"⚠️  Modelo no encontrado en {MODEL_PATH}")
            print("   Ejecuta 'python src/train.py' para entrenar el modelo")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        model = None
    
    # Verificar OpenAI API Key
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI API Key configurada")
    else:
        print("⚠️  OpenAI API Key no configurada - endpoint /coach no disponible")
    
    print("="*60 + "\n")
    
    yield  # Aplicación en ejecución
    
    # Shutdown
    print("\n🛑 Cerrando API...")

# Inicializar FastAPI con lifespan
app = FastAPI(
    title="API Predicción Deserción Estudiantil - Duoc UC",
    description="Sistema de predicción de riesgo de deserción y coaching académico con LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === ENDPOINTS DE THRESHOLD Y MÉTRICAS (DESPUÉS DE DEFINIR app) ===

@app.get("/threshold")
async def get_threshold():
    return {"threshold": current_threshold()}

@app.get("/metrics")
async def get_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {"detail": "metrics.json no disponible"}

@app.post("/threshold")
async def set_threshold(req: ThresholdUpdate):
    val = float(req.threshold)
    THRESHOLD_PATH.write_text(str(val))
    _threshold_cache["mtime"] = None
    return {"status": "updated", "threshold": current_threshold()}

@app.get("/openai/status")
async def openai_status():
    return {
        "has_key": bool(os.getenv("OPENAI_API_KEY")),
        "cached": openai_client is not None,
        "threshold": current_threshold()
    }

@app.post("/openai/reload")
async def openai_reload():
    c = get_openai_client(force=True)
    if c is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY no disponible tras recarga")
    return {"status": "reloaded"}

# === MODELOS DE DATOS ===

class StudentData(BaseModel):
    """Datos del estudiante para predicción."""
    # Features que usa el modelo entrenado
    AGNO: Optional[int] = Field(None, ge=2020, le=2030, description="Año académico")
    prom_gral: Optional[float] = Field(None, ge=1.0, le=7.0, description="Promedio general (1-7)")
    asistencia_pct: Optional[float] = Field(None, ge=0.0, le=1.0, description="Asistencia (0-1, ej: 0.85 = 85%)")
    
    # Alias para compatibilidad con nombres alternativos
    promedio: Optional[float] = Field(None, ge=1.0, le=7.0, description="Alias de prom_gral")
    asistencia: Optional[float] = Field(None, ge=0.0, le=100.0, description="Asistencia en % (0-100)")
    año: Optional[int] = Field(None, ge=2020, le=2030, description="Alias de AGNO")
    
    class Config:
        json_schema_extra = {
            "example": {
                "AGNO": 2024,
                "prom_gral": 5.5,
                "asistencia_pct": 0.85
            }
        }

class PredictionRequest(BaseModel):
    """Solicitud de predicción."""
    payload: StudentData

class PredictionResponse(BaseModel):
    """Respuesta de predicción."""
    riesgo_desercion: float = Field(..., ge=0, le=1, description="Probabilidad de deserción (0-1)")
    nivel_riesgo: str = Field(..., description="BAJO, MEDIO o ALTO")
    recomendacion: str = Field(..., description="Recomendación textual")
    confianza: str = Field(..., description="Nivel de confianza de la predicción")

class CoachRequest(BaseModel):
    """Solicitud de coaching con LLM."""
    student_data: Dict[str, Any]
    question: str = Field(..., description="Pregunta del estudiante o docente")
    context: Optional[str] = Field(None, description="Contexto adicional")

class CoachResponse(BaseModel):
    """Respuesta de coaching."""
    answer: str = Field(..., description="Respuesta del coach virtual")
    riesgo_detectado: Optional[float] = Field(None, description="Riesgo detectado si aplica")

# === ENDPOINTS PRINCIPALES ===

@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "API de Predicción de Deserción Estudiantil - Duoc UC",
        "version": "1.0.0",
        "status": "online",
        "model_loaded": model is not None,
        "endpoints": {
            "GET /": "Información de la API",
            "GET /health": "Estado de la API y modelo",
            "POST /predict": "Predicción de riesgo de deserción",
            "POST /coach": "Coach virtual con LLM (requiere OpenAI API key)",
            "GET /stats": "Estadísticas del modelo",
            "GET /threshold": "Ver umbral actual",
            "POST /threshold": "Actualizar umbral",
            "GET /metrics": "Métricas del modelo",
            "GET /docs": "Documentación interactiva Swagger",
            "GET /redoc": "Documentación ReDoc"
        }
    }

@app.get("/health")
async def health():
    """Verifica el estado de la API y el modelo."""
    openai_configured = bool(os.getenv("OPENAI_API_KEY"))
    
    return {
        "status": "healthy",
        "model": {
            "loaded": model is not None,
            "path": str(MODEL_PATH),
            "exists": MODEL_PATH.exists(),
            "features_count": len(model.feature_names) if model else 0
        },
        "services": {
            "openai_coach": "available" if openai_configured else "not_configured"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predice el riesgo de deserción de un estudiante.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta 'python src/train.py' para entrenar el modelo."
        )
    
    try:
        # Convertir a dict y normalizar campos
        data_dict = request.payload.dict(exclude_none=True)
        
        # Normalizar alias (promedio -> prom_gral, asistencia -> asistencia_pct, año -> AGNO)
        if "promedio" in data_dict and "prom_gral" not in data_dict:
            data_dict["prom_gral"] = data_dict.pop("promedio")
        if "asistencia" in data_dict and "asistencia_pct" not in data_dict:
            # Convertir de 0-100 a 0-1
            data_dict["asistencia_pct"] = data_dict.pop("asistencia") / 100.0
        if "año" in data_dict and "AGNO" not in data_dict:
            data_dict["AGNO"] = data_dict.pop("año")
        
        df = pd.DataFrame([data_dict])
        
        # create_features normaliza y crea las columnas que el modelo espera
        X = create_features(df)
        
        # Asegurar que todas las features del modelo existan
        for feat in model.feature_names:
            if feat not in X.columns:
                X[feat] = 0.0
        
        # Calcular confianza basada en features NO nulas
        campos_disponibles = (~X[model.feature_names].isna().all()).sum()
        campos_totales = len(model.feature_names)
        confianza_pct = campos_disponibles / campos_totales if campos_totales > 0 else 0
        
        if confianza_pct > 0.7:
            confianza = "ALTA"
        elif confianza_pct > 0.4:
            confianza = "MEDIA"
        else:
            confianza = "BAJA"
        
        # Predecir
        prob = float(model.predict_proba(X)[0])
        thr = current_threshold()
        
        if prob >= thr:
            nivel = "ALTO"
        elif prob >= 0.5:
            nivel = "MEDIO"
        else:
            nivel = "BAJO"

        # Recomendaciones según nivel
        if nivel == "BAJO":
            recomendacion = (
                "✅ El estudiante presenta bajo riesgo de deserción. "
                "Mantener seguimiento regular y reforzar hábitos positivos de estudio."
            )
        elif nivel == "MEDIO":
            recomendacion = (
                "⚠️ Riesgo medio de deserción. "
                "Recomendaciones:\n"
                "• Reunión con tutor académico\n"
                "• Plan de mejora en asistencia y/o notas\n"
                "• Apoyo psicopedagógico si es necesario\n"
                "• Seguimiento quincenal"
            )
        else:  # ALTO
            recomendacion = (
                "🚨 ALERTA: Riesgo alto - Acción inmediata\n\n"
                "Plan de intervención:\n"
                "1. Entrevista individual (esta semana)\n"
                "2. Evaluar situación personal/familiar\n"
                "3. Plan personalizado con metas claras\n"
                "4. Seguimiento semanal obligatorio\n"
                "5. Coordinación con Bienestar Estudiantil\n"
                "6. Apoyo financiero/becas si aplica\n"
                "7. Tutorías especiales"
            )
        
        return PredictionResponse(
            riesgo_desercion=prob,
            nivel_riesgo=nivel,
            recomendacion=recomendacion,
            confianza=confianza
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error en predicción: {str(e)}\nFeatures del modelo: {model.feature_names if model else 'N/A'}\nDatos recibidos: {list(request.payload.dict(exclude_none=True).keys())}"
        )

@app.post("/coach", response_model=CoachResponse)
async def coach(request: CoachRequest):
    """
    Coach virtual con LLM para estudiantes y docentes.
    
    Requiere OPENAI_API_KEY en .env
    """
    client = get_openai_client()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key no configurada. Añade OPENAI_API_KEY al archivo .env"
        )

    try:
        # Predecir riesgo si hay datos suficientes y modelo disponible
        riesgo = None
        if model and request.student_data:
            try:
                df = pd.DataFrame([request.student_data])
                X = create_features(df)
                riesgo = float(model.predict_proba(X)[0])
            except Exception as e:
                print(f"No se pudo calcular riesgo en /coach: {e}")

        # Construir prompt
        context_str = f"\nContexto adicional: {request.context}" if request.context else ""
        riesgo_str = ""
        if riesgo is not None:
            thr = current_threshold()
            nivel = "ALTO" if riesgo >= thr else "MEDIO" if riesgo >= 0.5 else "BAJO"
            riesgo_str = f"\n\n🎯 Riesgo de deserción detectado: {riesgo:.1%} ({nivel})"

        system_prompt = """Eres un coach académico experto de Duoc UC especializado en prevención de deserción estudiantil.

Tu rol es:
1. Brindar apoyo emocional y motivacional con empatía
2. Sugerir estrategias de estudio y organización concretas
3. Identificar recursos institucionales disponibles (tutorías, becas, apoyo psicológico)
4. Ofrecer consejos prácticos y accionables
5. Detectar señales de riesgo y sugerir intervenciones tempranas

Principios:
- Ser empático y comprensivo
- Ofrecer soluciones realistas y alcanzables
- Enfocarte en fortalezas del estudiante
- Promover autonomía y autorregulación
- Conectar con recursos institucionales cuando sea necesario

Responde de forma concreta, orientada a la acción y sin tecnicismos innecesarios."""
        
        user_prompt = f"""Pregunta del estudiante/docente: {request.question}

Datos del estudiante: {request.student_data}{context_str}{riesgo_str}

Por favor, proporciona una respuesta útil, personalizada y empática."""
        
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        answer = response.choices[0].message.content
        
        return CoachResponse(
            answer=answer,
            riesgo_detectado=riesgo
        )
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Librería 'openai' no instalada. Ejecuta: pip install openai"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error en coaching: {str(e)}"
        )

@app.get("/stats")
async def stats():
    """Estadísticas del modelo (requiere modelo cargado)."""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible. Ejecuta 'python src/train.py' primero."
        )
    
    try:
        return {
            "model_info": {
                "type": type(model.model).__name__,
                "features_count": len(model.feature_names),
                "features": model.feature_names,
                "has_scaler": model.scaler is not None,
                "has_imputer": model.imputer is not None
            },
            "model_path": str(MODEL_PATH),
            "model_size_bytes": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo stats: {str(e)}"
        )

# Endpoint adicional para demo
@app.get("/demo")
async def demo():
    """Ejemplos de uso de la API."""
    return {
        "message": "Ejemplos de uso de la API",
        "examples": {
            "predict": {
                "url": "/predict",
                "method": "POST",
                "payload": {
                    "payload": {
                        "promedio_asistencia": 70.0,
                        "porcentaje_aprobacion": 0.60,
                        "promedio_notas": 4.5,
                        "tasa_2020": 0.08
                    }
                }
            },
            "coach": {
                "url": "/coach",
                "method": "POST",
                "payload": {
                    "student_data": {
                        "promedio_asistencia": 65.0,
                        "promedio_notas": 4.2
                    },
                    "question": "Me cuesta concentrarme en clases, ¿qué puedo hacer?"
                }
            }
        },
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }
