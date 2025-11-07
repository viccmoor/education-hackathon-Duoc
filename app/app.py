"""
App Streamlit para predicción de riesgo académico y coaching personalizado.
"""
import streamlit as st
import requests
import json

# Configuración
st.set_page_config(
    page_title="Coach Académico Duoc",
    page_icon="📚",
    layout="wide"
)

API_URL = "http://localhost:8000"  # <-- cambiar de 8501 a 8000 (puerto FastAPI)


# ========== Header ==========
st.title("📚 Coach Académico Preventivo")
st.markdown("""
Este sistema estima tu riesgo de reprobación y genera un plan personalizado.

**⚠️ DISCLAIMER:** Este NO es un diagnóstico académico oficial. Consulta con tu tutor.
""")

# ========== Sidebar (Formulario) ==========
with st.sidebar:
    st.header("📋 Tu Perfil Académico")
    
    # Datos académicos
    st.subheader("Rendimiento")
    promedio = st.number_input("Promedio General", min_value=1.0, max_value=7.0, value=5.5, step=0.1)
    asistencia = st.slider("Asistencia (%)", 0, 100, 85)
    
    # Datos demográficos (opcionales)
    st.subheader("Información Adicional (Opcional)")
    edad = st.number_input("Edad", min_value=15, max_value=70, value=20)
    sexo = st.selectbox("Sexo", ["M", "F", "Otro"])
    asignatura = st.text_input("Asignatura principal", "Programación")
    establecimiento = st.text_input("Establecimiento", "Duoc UC Sede Maipú")
    
    # Botón de evaluación
    evaluar_btn = st.button("🔍 Evaluar Riesgo", type="primary")

# ========== Main Area ==========
if evaluar_btn:
    user_data = {
        "promedio": promedio,
        "asistencia": asistencia,
        "edad": edad,
        "sexo": sexo,
        "asignatura": asignatura,
        "establecimiento": establecimiento
    }
    
    with st.spinner("Analizando tu perfil..."):
        try:
            # Obtener threshold actual
            thr_resp = requests.get(f"{API_URL}/threshold", timeout=5)
            threshold = thr_resp.json().get("threshold", 0.5) if thr_resp.status_code == 200 else 0.5

            # Predicción
            response = requests.post(f"{API_URL}/predict", json={"payload": user_data})
            
            if response.status_code == 200:
                result = response.json()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_score = result['riesgo_desercion']
                    st.metric(
                        "Puntaje de Riesgo",
                        f"{risk_score:.1%}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Nivel de Riesgo",
                        result['nivel_riesgo']
                    )
                
                with col3:
                    # Indicador usa threshold dinámico
                    if risk_score >= threshold:
                        color = "🔴"
                    elif risk_score >= 0.5:
                        color = "🟡"
                    else:
                        color = "🟢"
                    st.metric("Indicador", color)
                
                # Mostrar threshold y métricas
                st.caption(f"Umbral alto: {threshold:.3f}")
                
                # Métricas del modelo
                try:
                    m = requests.get(f"{API_URL}/metrics", timeout=5)
                    if m.status_code == 200:
                        mets = m.json()
                        st.caption(f"ROC-AUC: {mets.get('roc_auc', 0):.3f} | F1(opt): {mets.get('f1_opt', 0):.3f} | Precision: {mets.get('precision_opt', 0):.3f}")
                except Exception:
                    pass

                if result['nivel_riesgo'] == "ALTO":
                    st.error("⚠️ Riesgo alto detectado. Se recomienda derivación a tutor académico.")
                elif result['nivel_riesgo'] == "MEDIO":
                    st.warning("⚠️ Riesgo medio. Considera apoyo preventivo.")
                else:
                    st.success("✅ Riesgo bajo. Mantén tus hábitos actuales.")
                
                # Drivers (opcional si /predict los devuelve)
                if 'drivers' in result:
                    st.subheader("🎯 Principales Factores de Riesgo")
                    for driver in result.get('drivers', []):
                        st.write(f"- **{driver['feature']}**: {driver['value']:.2f} (importancia: {driver['importance']:.2f})")
                
                # Coach (si quieres agregar)
                if st.button("📝 Generar Plan Personalizado"):
                    with st.spinner("Creando tu plan..."):
                        coach_request = {
                            "user_profile": user_data,
                            "risk_score": risk_score,
                            "top_drivers": [d['feature'] for d in drivers_data]
                        }
                        
                        coach_response = requests.post(f"{API_URL}/coach", json=coach_request)
                        
                        if coach_response.status_code == 200:
                            plan_data = coach_response.json()
                            
                            st.subheader("📋 Tu Plan de Éxito Académico")
                            st.markdown(plan_data['plan'])
                            
                            if plan_data['sources']:
                                st.caption(f"📚 Fuentes: {', '.join(plan_data['sources'])}")
                        else:
                            st.error(f"Error generando plan: {coach_response.status_code}")
            else:
                st.error(f"Error en predicción: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error conectando con la API: {e}")
            st.info("Asegúrate de que la API esté corriendo en http://localhost:8000")

st.markdown("---")
st.caption("""
Desarrollado para Hackathon IA Duoc UC 2025 | 
Basado en datos de rendimiento académico | 
⚠️ No sustituye orientación académica profesional
""")