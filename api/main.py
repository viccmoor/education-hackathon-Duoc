from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
import sys
import os
import json
from dotenv import load_dotenv

# Configurar path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DesercionPredictor
from src.features import create_features
from src.rag import RendimientoRAG

# Cargar variables de entorno
load_dotenv()

# Variables globales
MODEL_PATH = Path("models/desercion_predictor.joblib")
METRICS_PATH = Path("models/metrics.json")
THRESHOLD_PATH = Path("models/threshold.txt")

model: Optional[DesercionPredictor] = None
rag: Optional[RendimientoRAG] = None
THRESHOLD_OVERRIDE: Optional[float] = None

def current_threshold() -> float:
    """Obtiene el threshold actual (override o desde archivo)."""
    if THRESHOLD_OVERRIDE is not None:
        return THRESHOLD_OVERRIDE
    
    if THRESHOLD_PATH.exists():
        try:
            with open(THRESHOLD_PATH, 'r') as f:
                return float(f.read().strip())
        except Exception:
            pass
    
    return 0.5  # Default

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación (startup/shutdown)."""
    global model, rag
    
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
    
    # Cargar RAG
    print("\n🔍 Inicializando sistema RAG...")
    try:
        rag = RendimientoRAG(max_rows=50_000)
    except Exception as e:
        print(f"⚠️ RAG no disponible: {e}")
        rag = None
    
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

# === MODELOS DE DATOS ===

class StudentData(BaseModel):
    """Datos del estudiante para predicción."""
    promedio: Optional[float] = Field(None, ge=1, le=7, description="Promedio general (1-7)")
    asistencia: Optional[float] = Field(None, ge=0, le=100, description="Asistencia (%)")
    edad: Optional[int] = Field(None, ge=15, le=70, description="Edad del estudiante")
    sexo: Optional[str] = Field(None, description="Sexo (M/F/Otro)")
    asignatura: Optional[str] = Field(None, description="Asignatura principal")
    establecimiento: Optional[str] = Field(None, description="Establecimiento educacional")
    año: Optional[int] = Field(None, ge=2020, le=2030, description="Año académico")

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

# === ENDPOINTS ===

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
            "GET /threshold": "Obtener threshold actual",
            "GET /metrics": "Métricas del modelo",
            "GET /docs": "Documentación interactiva Swagger"
        }
    }

@app.get("/health")
async def health():
    """Verifica el estado de la API y el modelo."""
    return {
        "status": "healthy",
        "model": {
            "loaded": model is not None,
            "path": str(MODEL_PATH),
            "exists": MODEL_PATH.exists(),
            "features_count": len(model.feature_names) if model else 0
        },
        "services": {
            "openai_coach": "available" if os.getenv("OPENAI_API_KEY") else "not_configured",
            "rag": "available" if rag and rag.bm25 else "not_available"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predice el riesgo de deserción de un estudiante."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta 'python src/train.py' para entrenar el modelo."
        )
    
    try:
        # Convertir a dict
        data_dict = request.payload.dict(exclude_none=True)
        
        if not data_dict:
            raise HTTPException(
                status_code=400,
                detail="Debe proporcionar al menos un campo de datos del estudiante"
            )
        
        df = pd.DataFrame([data_dict])
        X = create_features(df)
        
        # Calcular confianza
        campos_disponibles = len(data_dict)
        campos_totales = 7  # promedio, asistencia, edad, sexo, asignatura, establecimiento, año
        confianza_pct = campos_disponibles / campos_totales
        
        if confianza_pct > 0.7:
            confianza = "ALTA"
        elif confianza_pct > 0.4:
            confianza = "MEDIA"
        else:
            confianza = "BAJA"
        
        # Predecir
        prob = float(model.predict_proba(X)[0])
        threshold = current_threshold()
        
        # Clasificar nivel
        if prob >= threshold:
            nivel = "ALTO"
            recomendacion = (
                "🚨 ALERTA: Riesgo alto de deserción - Acción inmediata requerida\n\n"
                "Plan de intervención urgente:\n"
                "1. Entrevista individual con el estudiante (esta semana)\n"
                "2. Evaluar situación personal/familiar/económica\n"
                "3. Plan de intervención personalizado con metas claras\n"
                "4. Seguimiento semanal obligatorio\n"
                "5. Coordinación con Bienestar Estudiantil\n"
                "6. Considerar opciones de apoyo financiero/becas\n"
                "7. Vincular con tutorías académicas especializadas"
            )
        elif prob >= 0.5:
            nivel = "MEDIO"
            recomendacion = (
                "⚠️ Riesgo medio de deserción. "
                "Recomendaciones:\n"
                "• Reunión con tutor académico para identificar causas\n"
                "• Plan de mejora en asistencia y/o notas\n"
                "• Apoyo psicopedagógico si es necesario\n"
                "• Seguimiento quincenal del progreso"
            )
        else:
            nivel = "BAJO"
            recomendacion = (
                "✅ Bajo riesgo de deserción. "
                "Mantener seguimiento regular y reforzar hábitos positivos de estudio."
            )
        
        return PredictionResponse(
            riesgo_desercion=prob,
            nivel_riesgo=nivel,
            recomendacion=recomendacion,
            confianza=confianza
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error en predicción: {str(e)}"
        )

def sanitize_for_api(text: str) -> str:
    """Normaliza texto para evitar errores de encoding."""
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

@app.post("/coach", response_model=CoachResponse)
async def coach(request: CoachRequest):
    """Coach virtual con LLM."""
    print(f"🔍 DEBUG /coach - Iniciando request")
    
    # Verificar API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key no configurada. Añade OPENAI_API_KEY al archivo .env"
        )
    
    print(f"🔍 DEBUG - API key length: {len(api_key)}")
    print(f"🔍 DEBUG - API key starts with 'sk-': {api_key.startswith('sk-')}")
    
    # Verificar que es ASCII puro
    try:
        api_key.encode('ascii')
        print(f"✅ DEBUG - API key es ASCII válido")
    except UnicodeEncodeError as e:
        print(f"❌ DEBUG - API key contiene caracteres no-ASCII: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"OpenAI API key tiene caracteres inválidos. Regenerá la key desde platform.openai.com"
        )
    
    try:
        from openai import OpenAI
        
        print(f"🔍 DEBUG - Inicializando cliente OpenAI...")
        client = OpenAI(api_key=api_key)
        print(f"✅ DEBUG - Cliente OpenAI inicializado")
        
        # Predecir riesgo si hay datos
        riesgo = None
        if model and request.student_data:
            try:
                print(f"🔍 DEBUG - Calculando riesgo con datos: {list(request.student_data.keys())}")
                df = pd.DataFrame([request.student_data])
                X = create_features(df)
                riesgo = float(model.predict_proba(X)[0])
                print(f"✅ DEBUG - Riesgo calculado: {riesgo:.3f}")
            except Exception as e:
                print(f"⚠️ No se pudo calcular riesgo en /coach: {e}")
                import traceback
                traceback.print_exc()
        
        # Sanitizar inputs
        print(f"🔍 DEBUG - Sanitizando question: {request.question[:50]}...")
        safe_question = sanitize_for_api(request.question)
        safe_context = sanitize_for_api(request.context or "")
        
        # Buscar contexto RAG
        rag_context = ""
        if rag and rag.bm25 is not None:
            try:
                print(f"🔍 DEBUG - Buscando contexto RAG...")
                query_parts = [safe_question]
                if request.student_data.get("promedio"):
                    query_parts.append(f"promedio {request.student_data['promedio']}")
                if request.student_data.get("asistencia"):
                    query_parts.append(f"asistencia {request.student_data['asistencia']}")
                
                query = " ".join(query_parts)
                results = rag.search(query, top_k=3)
                
                if results:
                    formatted = rag.format_context(results)
                    rag_context = f"\n\nContexto de datos historicos:\n{sanitize_for_api(formatted)}"
                    print(f"✅ DEBUG - RAG context generado ({len(rag_context)} chars)")
            except Exception as e:
                print(f"⚠️ Error en busqueda RAG: {e}")
                import traceback
                traceback.print_exc()
        
        # Construir prompt
        context_str = f"\nContexto adicional: {safe_context}" if safe_context else ""
        riesgo_str = ""
        if riesgo is not None:
            thr = current_threshold()
            nivel = "ALTO" if riesgo >= thr else "MEDIO" if riesgo >= 0.5 else "BAJO"
            riesgo_str = f"\n\nRiesgo de desercion detectado: {riesgo:.1%} ({nivel})"
        
        # Sanitizar student_data
        safe_student_data = {}
        for k, v in request.student_data.items():
            if isinstance(v, str):
                safe_student_data[k] = sanitize_for_api(v)
            else:
                safe_student_data[k] = v
        
        system_prompt = """Eres un coach academico experto de Duoc UC especializado en prevencion de desercion estudiantil.

Tu rol es:
1. Brindar apoyo emocional y motivacional con empatia
2. Sugerir estrategias de estudio y organizacion concretas
3. Identificar recursos institucionales disponibles (tutorias, becas, apoyo psicologico)
4. Usar datos historicos de estudiantes similares para contextualizar tus recomendaciones
5. Ofrecer consejos practicos y accionables

Principios:
- Ser empatico y comprensivo
- Ofrecer soluciones realistas y alcanzables
- Enfocarte en fortalezas del estudiante
- Promover autonomia y autorregulacion
- Conectar con recursos institucionales cuando sea necesario
- Citar datos historicos cuando sea relevante

Responde de forma concreta, orientada a la accion y sin tecnicismos innecesarios."""
        
        user_prompt = f"""Pregunta del estudiante/docente: {safe_question}

Datos del estudiante: {safe_student_data}{context_str}{riesgo_str}{rag_context}

Por favor, proporciona una respuesta util, personalizada y empatica."""
        
        print(f"🔍 DEBUG - User prompt length: {len(user_prompt)}")
        print(f"🔍 DEBUG - Llamando OpenAI API (model: {os.getenv('LLM_MODEL', 'gpt-4o-mini')})...")
        
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        print(f"✅ DEBUG - OpenAI API respondió correctamente")
        answer = response.choices[0].message.content
        
        return CoachResponse(
            answer=answer,
            riesgo_detectado=riesgo
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("\n" + "="*60)
        print("❌ ERROR COMPLETO EN /coach:")
        print("="*60)
        traceback.print_exc()
        print("="*60 + "\n")
        raise HTTPException(
            status_code=500, 
            detail=f"Error en coaching: {str(e)}"
        )

@app.get("/threshold")
async def get_threshold():
    """Obtiene el threshold actual."""
    return {
        "threshold": current_threshold(),
        "description": "Umbral para clasificar riesgo ALTO vs MEDIO"
    }

@app.post("/threshold")
async def update_threshold(threshold: float):
    """Actualiza el threshold dinámicamente."""
    global THRESHOLD_OVERRIDE
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="Threshold debe estar entre 0.0 y 1.0")
    THRESHOLD_OVERRIDE = threshold
    return {
        "threshold": threshold,
        "message": f"Threshold actualizado a {threshold:.3f}"
    }

@app.get("/metrics")
async def get_metrics():
    """Obtiene métricas del modelo."""
    if not METRICS_PATH.exists():
        return {
            "roc_auc": 0.0,
            "f1_opt": 0.0,
            "precision_opt": 0.0,
            "recall_opt": 0.0,
            "threshold_opt": current_threshold()
        }
    
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo métricas: {e}")
