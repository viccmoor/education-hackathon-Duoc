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

# ========== Sidebar (Formulario) - UNA SOLA VEZ ==========
with st.sidebar:
    st.header("📋 Tu Perfil Académico")
    
    # Datos académicos
    st.subheader("Rendimiento")
    promedio = st.number_input("Promedio General", min_value=1.0, max_value=7.0, value=5.5, step=0.1, key="promedio_global")
    asistencia = st.slider("Asistencia (%)", 0, 100, 85, key="asistencia_global")
    
    # Datos demográficos (opcionales)
    st.subheader("Información Adicional (Opcional)")
    edad = st.number_input("Edad", min_value=15, max_value=70, value=20, key="edad_global")
    sexo = st.selectbox("Sexo", ["M", "F", "Otro"], key="sexo_global")
    asignatura = st.text_input("Asignatura principal", "Programación", key="asignatura_global")
    establecimiento = st.text_input("Establecimiento", "Duoc UC Sede Maipú", key="establecimiento_global")
    
    # Botón de evaluación
    evaluar_btn = st.button("🔍 Evaluar Riesgo", type="primary")

# ========== PESTAÑAS PRINCIPALES ==========
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Evaluación de Riesgo",
    "📊 Mi Panel",
    "💬 Coach Virtual",
    "📈 Estadísticas del Modelo"
])

# === PESTAÑA 1: EVALUACIÓN ===
with tab1:
    st.header("Evaluación de Riesgo Académico")
    
    if evaluar_btn:
        user_data = {
            "promedio": promedio,
            "asistencia": asistencia,
            "edad": edad,
            "sexo": sexo,
            "asignatura": asignatura,
            "establecimiento": establecimiento
        }
        
        # Guardar en session_state para usar en chatbot
        st.session_state.last_prediction = user_data
        
        with st.spinner("Analizando tu perfil..."):
            try:
                # Obtener threshold actual
                thr_resp = requests.get(f"{API_URL}/threshold", timeout=5)
                threshold = thr_resp.json().get("threshold", 0.5) if thr_resp.status_code == 200 else 0.5

                # Predicción
                response = requests.post(f"{API_URL}/predict", json={"payload": user_data}, timeout=10)
                
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
                    
                    # Mostrar recomendación
                    if 'recomendacion' in result:
                        st.info(result['recomendacion'])
                    
                else:
                    st.error(f"Error en predicción: {response.status_code}")
                    st.code(response.text)
                    
            except Exception as e:
                st.error(f"Error conectando con la API: {e}")
                st.info("Asegúrate de que la API esté corriendo en http://localhost:8000")

# === PESTAÑA 2: PANEL ===
with tab2:
    st.header("📊 Mi Panel Académico")
    st.info("Funcionalidad en desarrollo: histórico de predicciones, evolución de riesgo, etc.")

# === PESTAÑA 3: CHATBOT ===
with tab3:
    st.header("💬 Coach Virtual")
    st.markdown("""
    Pregúntame sobre:
    - Estrategias para mejorar tu rendimiento académico
    - Cómo manejar la ansiedad o falta de motivación
    - Recursos disponibles en Duoc UC (becas, tutorías, apoyo psicológico)
    - Experiencias de estudiantes en situaciones similares
    """)
    
    # Historial de chat (usar session_state para persistencia)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Contenedor para el chat
    chat_container = st.container()
    
    # Mostrar historial
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Input del usuario
    user_input = st.chat_input("Escribe tu pregunta aquí...")
    
    if user_input:
        # Agregar mensaje del usuario al historial
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Preparar datos del estudiante (si están disponibles en session_state)
        student_data = {}
        if 'last_prediction' in st.session_state:
            pred = st.session_state.last_prediction
            student_data = {
                "promedio": pred.get("promedio"),
                "asistencia": pred.get("asistencia"),
                "edad": pred.get("edad"),
                "sexo": pred.get("sexo")
            }
        
        # Llamar al endpoint /coach
        with st.spinner("Pensando..."):
            try:
                coach_payload = {
                    "student_data": student_data,
                    "question": user_input,
                    "context": None
                }
                
                response = requests.post(
                    f"{API_URL}/coach",
                    json=coach_payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "Lo siento, no pude generar una respuesta.")
                    riesgo = result.get("riesgo_detectado")
                    
                    # Agregar respuesta del asistente al historial
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    # Mostrar respuesta
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                            
                            # Mostrar riesgo si está disponible
                            if riesgo is not None:
                                st.caption(f"🎯 Riesgo de deserción detectado: {riesgo:.1%}")
                    
                    st.rerun()
                
                elif response.status_code == 503:
                    st.error("⚠️ El servicio de coach no está disponible. Verifica que OPENAI_API_KEY esté configurada.")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            
            except requests.exceptions.Timeout:
                st.error("⏱️ La consulta tardó demasiado. Intenta de nuevo con una pregunta más específica.")
            except Exception as e:
                st.error(f"Error al contactar el coach: {e}")
    
    # Botón para limpiar historial
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.chat_history = []
        st.rerun()
    
    # === PALABRAS CLAVE PARA TEMPLATES ===
    st.markdown("---")
    st.markdown("**💡 Palabras Clave para Consultas Rápidas:**")
    st.caption("Haz clic en una palabra clave para activar una pregunta predeterminada")
    
    # Diccionario de palabras clave -> prompts
    KEYWORDS = {
        # Organización de Estudio
        "📚 Técnicas de Estudio": f"Soy estudiante de {asignatura} en {establecimiento}. Mi promedio es {promedio} y asisto {asistencia}% de las clases. ¿Qué técnicas de estudio me recomiendas para mejorar mi rendimiento académico?",
        "📅 Plan de Estudio": f"Necesito organizar mi tiempo de estudio. Tengo promedio {promedio} y asistencia {asistencia}%. ¿Puedes ayudarme a crear un plan de estudio semanal realista?",
        "📝 Tomar Apuntes": "¿Cuáles son las mejores técnicas para tomar apuntes en clase que me ayuden a retener información?",
        
        # Rendimiento Académico
        "📊 Mejorar Promedio": f"Mi promedio actual es {promedio}. ¿Qué estrategias concretas puedo implementar para mejorarlo en las próximas semanas?",
        "🔄 Recuperar Ramos": f"Estoy atrasado en {asignatura}. ¿Cómo puedo recuperar el ritmo sin afectar mis otras asignaturas?",
        "✨ Casos de Éxito": f"¿Qué hicieron otros estudiantes con promedio {promedio} y asistencia {asistencia}% que lograron mejorar significativamente?", 

        # Asistencia y Compromiso
        "📈 Mejorar Asistencia": f"Mi asistencia es {asistencia}%. ¿Qué estrategias puedo usar para asistir más regularmente a clases?",
        "📄 Justificar Inasistencias": "¿Qué debo hacer cuando tengo inasistencias justificadas? ¿Cómo afecta mi situación académica?",
        "💪 Motivación Asistencia": "Me cuesta motivarme para ir a clases. ¿Qué consejos me das para mantener el compromiso con mi asistencia?",
        
        # Bienestar y Motivación
        "😌 Manejo de Estrés": "Me siento abrumado con la carga académica. ¿Qué técnicas puedo usar para manejar el estrés?",
        "🔥 Falta de Motivación": f"Me siento desmotivado con mis estudios en {asignatura}. ¿Cómo puedo recuperar la motivación y el interés?",
        "🧠 Apoyo Psicológico": "¿Qué recursos de apoyo psicológico hay disponibles en Duoc UC para estudiantes?",
        "⚖️ Balance Vida-Estudio": "¿Cómo puedo equilibrar mis estudios con mi vida personal y responsabilidades familiares?",
        
        # Recursos Institucionales
        "👨‍🏫 Tutorías": f"¿Qué tutorías académicas están disponibles para {asignatura}? ¿Cómo puedo acceder a ellas?",
        "💰 Becas": "¿Qué becas o ayudas financieras hay disponibles en Duoc UC? ¿Cómo puedo postular?",
        "🏥 Bienestar Estudiantil": "¿Qué servicios ofrece Bienestar Estudiantil en Duoc UC y cómo puedo acceder a ellos?",
        "🎯 Orientación Vocacional": "Tengo dudas sobre si esta carrera es para mí. ¿Hay servicios de orientación vocacional disponibles?",
        
        # Estrategias Específicas
        "📖 Preparación Exámenes": "Tengo un examen importante próximamente. ¿Qué estrategias de preparación me recomiendas?",
        "👥 Trabajos en Grupo": "¿Cómo puedo organizar trabajos en grupo de manera más efectiva y productiva?",
        "⏱️ Gestión de Tiempo": "Me cuesta administrar el tiempo durante pruebas y exámenes. ¿Qué técnicas puedo usar?",
        "📚 Comprensión Lectora": f"En {asignatura} tengo que leer mucho material técnico. ¿Cómo puedo mejorar mi comprensión lectora?"
    }
    
    # Mostrar botones de palabras clave en grid de 3 columnas
    keywords_list = list(KEYWORDS.items())
    for i in range(0, len(keywords_list), 3):
        cols = st.columns(3)
        for j in range(3):
            idx = i + j
            if idx < len(keywords_list):
                keyword, prompt = keywords_list[idx]
                with cols[j]:
                    if st.button(
                        keyword,
                        key=f"keyword_{idx}",
                        use_container_width=True,
                        help=f"Clic para preguntar: {prompt[:60]}..."
                    ):
                        # Agregar al historial y recargar
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": prompt
                        })
                        st.rerun()

# === PESTAÑA 4: ESTADÍSTICAS ===
with tab4:
    st.header("📈 Estadísticas del Modelo")
    st.markdown("""
    Visualiza el rendimiento y precisión del modelo predictivo.
    """)
    
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            
            st.subheader("Métricas Globales")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
            with col2:
                st.metric("Precisión", f"{metrics.get('precision_opt', 0):.3f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall_opt', 0):.3f}")
            with col4:
                st.metric("F1 Score", f"{metrics.get('f1_opt', 0):.3f}")
            
            st.subheader("Distribución de Riesgo")
            riesgo_data = metrics.get("riesgo_distribution", {})
            if riesgo_data:
                st.bar_chart(riesgo_data)
            else:
                st.write("No hay datos de distribución de riesgo disponibles.")
        else:
            st.error(f"Error al obtener métricas: {response.status_code}")
    except Exception as e:
        st.error(f"Error conectando con la API: {e}")

st.markdown("---")
st.caption("""
Desarrollado para Hackathon IA Duoc UC 2025 | 
Basado en datos de rendimiento académico | 
⚠️ No sustituye orientación académica profesional
""")