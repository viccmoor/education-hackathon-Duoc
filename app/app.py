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

API_URL = "http://localhost:8000"

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
            response = requests.post(f"{API_URL}/predict", json=user_data)
            
            if response.status_code == 200:
                result = response.json()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_score = result['score']
                    st.metric(
                        "Puntaje de Riesgo",
                        f"{risk_score:.1%}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Nivel de Riesgo",
                        result['riesgo']
                    )
                
                with col3:
                    if risk_score < 0.3:
                        color = "🟢"
                    elif risk_score < 0.6:
                        color = "🟡"
                    else:
                        color = "🔴"
                    st.metric("Indicador", color)
                
                if result['riesgo'] == "Alto":
                    st.error("⚠️ Riesgo alto detectado. Se recomienda derivación a tutor académico.")
                else:
                    st.success("✅ Riesgo bajo. Mantén tus hábitos actuales.")
                
                st.subheader("🎯 Principales Factores de Riesgo")
                drivers_data = result['drivers']
                for driver in drivers_data:
                    st.write(f"- **{driver['feature']}**: {driver['value']:.2f} (importancia: {driver['importance']:.2f})")
                
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