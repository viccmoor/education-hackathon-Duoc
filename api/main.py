from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
import sys
import os
from dotenv import load_dotenv

# Configurar path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DesercionPredictor
from src.features import create_features

# Cargar variables de entorno
load_dotenv()

# Variable global para el modelo
MODEL_PATH = os.path.join("..", "models", "desercion_predictor.joblib")
model: Optional[DesercionPredictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación (startup/shutdown)."""
    global model
    
    # Startup
    print("\n" + "="*60)
    print("🚀 INICIANDO API DE PREDICCIÓN DE DESERCIÓN")
    print("="*60)
    
    try:
        if os.path.exists(MODEL_PATH):
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

# === MODELOS DE DATOS ===

class StudentData(BaseModel):
    """Datos del estudiante para predicción."""
    promedio_asistencia: Optional[float] = Field(None, ge=0, le=100, description="Promedio de asistencia (0-100%)")
    porcentaje_aprobacion: Optional[float] = Field(None, ge=0, le=1, description="Porcentaje de aprobación (0-1)")
    promedio_notas: Optional[float] = Field(None, ge=1, le=7, description="Promedio de notas (1-7)")
    tasa_2020: Optional[float] = Field(None, ge=0, le=1, description="Tasa histórica (0-1)")
    estudiantes_retirados: Optional[int] = Field(None, ge=0, description="Estudiantes retirados en el curso")
    porcentaje_retiro: Optional[float] = Field(None, ge=0, le=1, description="Porcentaje de retiro (0-1)")
    total_estudiantes: Optional[int] = Field(None, ge=1, description="Total de estudiantes en el curso")
    año: Optional[int] = Field(None, ge=2020, le=2030, description="Año académico")
    
    class Config:
        json_schema_extra = {
            "example": {
                "promedio_asistencia": 75.0,
                "porcentaje_aprobacion": 0.65,
                "promedio_notas": 5.0,
                "tasa_2020": 0.05,
                "estudiantes_retirados": 15,
                "porcentaje_retiro": 0.03,
                "total_estudiantes": 500,
                "año": 2024
            }
        }

class StudentPayload(BaseModel):
    """Datos del estudiante para predicción."""
    promedio: float = Field(..., ge=1, le=7, description="Promedio de notas (1-7)")
    asistencia: float = Field(..., ge=0, le=100, description="Porcentaje de asistencia (0-100%)")
    edad: int = Field(..., ge=10, le=100, description="Edad del estudiante")
    sexo: str = Field(..., description="Sexo del estudiante (M/F u otro)")
    asignatura: str = Field(..., description="Nombre de la asignatura")
    establecimiento: str = Field(..., description="Nombre del establecimiento educativo")

    class Config:
        schema_extra = {
            "example": {
                "promedio": 5.5,
                "asistencia": 85,
                "edad": 20,
                "sexo": "M",
                "asignatura": "Programación",
                "establecimiento": "Duoc UC Sede Maipú"
            }
        }

class PredictionRequest(BaseModel):
    """Solicitud de predicción."""
    payload: StudentPayload

class PredictionResponse(BaseModel):
    """Respuesta de predicción."""
    riesgo_desercion: float = Field(..., ge=0, le=1, description="Probabilidad de deserción (0-1)")
    nivel_riesgo: str = Field(..., description="BAJO, MEDIO o ALTO")
    recomendacion: str = Field(..., description="Recomendación textual")
    confianza: str = Field(..., description="Nivel de confianza de la predicción")
    drivers: list[dict] = Field(
        ..., description="Factores que influyen en la predicción: feature, value e importancia"
    )

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
            "GET /stats": "Estadísticas del modelo",
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
    
    Retorna:
    - riesgo_desercion: Probabilidad entre 0 y 1
    - nivel_riesgo: BAJO (<0.5), MEDIO (0.5-0.8), ALTO (>0.8)
    - recomendacion: Texto con recomendaciones según el nivel de riesgo
    - confianza: Nivel de confianza basado en datos disponibles
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta 'python src/train.py' para entrenar el modelo."
        )
    
    try:
        # Convertir a DataFrame
        data_dict = request.payload.dict(exclude_none=True)
        
        if not data_dict:
            raise HTTPException(
                status_code=400,
                detail="Debe proporcionar al menos un campo de datos del estudiante"
            )
        
        df = pd.DataFrame([data_dict])
        
        # Crear features
        X = create_features(df)
        
        # Calcular confianza basada en datos disponibles
        campos_disponibles = len(data_dict)
        campos_totales = len(StudentData.model_fields)
        confianza_pct = campos_disponibles / campos_totales
        
        if confianza_pct > 0.7:
            confianza = "ALTA"
        elif confianza_pct > 0.4:
            confianza = "MEDIA"
        else:
            confianza = "BAJA"
        
        # Predecir
        prob = float(model.predict_proba(X)[0])
        
        # Clasificar nivel de riesgo
        if prob < 0.5:
            nivel = "BAJO"
            recomendacion = (
                "✅ El estudiante presenta bajo riesgo de deserción. "
                "Mantener seguimiento regular y reforzar hábitos positivos de estudio."
            )
        elif prob < 0.8:
            nivel = "MEDIO"
            recomendacion = (
                "⚠️ El estudiante presenta riesgo medio de deserción. "
                "Recomendaciones:\n"
                "• Reunión con tutor académico para identificar causas\n"
                "• Plan de mejora en asistencia y/o notas\n"
                "• Apoyo psicopedagógico si es necesario\n"
                "• Seguimiento quincenal del progreso"
            )
        else:
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

        drivers = []
        for col in X.columns:
            val = X[col].iloc[0]
            importance = abs(val - X[col].mean())
            drivers.append({
                "feature": str(col),
                "value": float(val),
                "importance": float(importance)
            })

        drivers = sorted(drivers, key=lambda x: x["importance"], reverse=True)[:5]

        return PredictionResponse(
            riesgo_desercion=prob,
            nivel_riesgo=nivel,
            recomendacion=recomendacion,
            confianza=confianza,
            drivers=drivers
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error en predicción: {str(e)}\nVerifica que los datos estén en el formato correcto."
        )

@app.post("/coach", response_model=CoachResponse)
async def coach(request: CoachRequest):
    """
    Coach virtual con LLM para estudiantes y docentes.
    
    Requiere OPENAI_API_KEY en .env
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key no configurada. Añade OPENAI_API_KEY al archivo .env"
        )
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=openai_key)
        
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
            nivel = "ALTO" if riesgo > 0.8 else "MEDIO" if riesgo > 0.5 else "BAJO"
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
        
        # Llamar a OpenAI
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
