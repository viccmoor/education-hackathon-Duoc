"""
Módulo en que se deployea la aplicación de Gradio.
"""
import requests
import gradio as gr


API_URL = "https://tu-api.onrender.com"
RISK_THRESHOLD = 0.7


def predict(inputs):
    """Llama al endpoint /predict de la API."""
    try:
        response = requests.post(f"{API_URL}/predict", json=inputs)
        data = response.json()
        score = data["score"]
        drivers = data.get("drivers", [])
        return score, drivers
    except Exception as e:
        return None, [f"Error al conectar con la API: {e}"]


def coach(inputs):
    """Llama al endpoint /coach de la API."""
    try:
        response = requests.post(f"{API_URL}/coach", json=inputs)
        data = response.json()
        return data["plan"]
    except Exception as e:
        return f"Error al conectar con la API: {e}"


def interface_fn(
    edad: int,
    sexo: str,
    asignatura: str,
    promedio: str,
    asistencia,
    establecimiento
):
    inputs = {
        "edad": edad,
        "sexo": sexo,
        "asignatura": asignatura,
        "promedio": promedio,
        "asistencia": asistencia,
        "establecimiento": establecimiento,
    }

    score, drivers = predict(inputs)
    if score is None:
        return "❌ Error en la predicción.", "", ""

    riesgo = f"{score:.2f} ({'ALTO' if score >= RISK_THRESHOLD else 'BAJO'})"
    explicacion = "\n".join([f"- {d}" for d in drivers])

    if score >= RISK_THRESHOLD:
        derivacion = (
            "⚠️ Riesgo alto detectado. Se recomienda derivar al tutor académico."
        )
    else:
        derivacion = "✅ Riesgo bajo. Mantén los hábitos actuales."

    plan = coach(inputs)
    return riesgo, explicacion, f"{derivacion}\n\n{plan}"


with gr.Blocks(
    title="Education Hackathon Duoc",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.Markdown(
        """
        # Tutor Virtual Adaptativo
        _Estimación de riesgo de deserción y plan personalizado._

        **Aviso:** Este sistema no reemplaza la orientación profesional.
        Los resultados son estimaciones generadas por un modelo de IA
        educativo.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            edad = gr.Slider(5, 20, value=18, step=1, label="Edad")
            sexo = gr.Dropdown(["Masculino", "Femenino", "Otro"], label="Sexo")
            asignatura = gr.Textbox(label="Asignatura principal")
            promedio = gr.Number(
                label="Promedio general (0-7)",
                value=5.5, minimum=1, maximum=7,
                step=0.01
            )
            asistencia = gr.Slider(0, 100, value=85, step=1, label="Asistencia (%)")
            establecimiento = gr.Textbox(label="Establecimiento")

            btn = gr.Button("📊 Estimar riesgo y generar plan")

        with gr.Column(scale=2):
            riesgo = gr.Textbox(label="Nivel de Riesgo", interactive=False)
            explicacion = gr.Textbox(label="Variables más influyentes", lines=4, interactive=False)
            plan = gr.Textbox(label="Plan de acción personalizado", lines=6, interactive=False)

    btn.click(
        interface_fn,
        inputs=[edad, sexo, asignatura, promedio, asistencia, establecimiento],
        outputs=[riesgo, explicacion, plan],
    )

demo.launch(share=True)
