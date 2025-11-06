"""
Módulo en que se deployea la aplicación de Gradio.
"""
import gradio as gr


def default_response() -> str:
    """Testing response."""
    return "Respuesta predeterminada"


gr.ChatInterface(
    fn=default_response,
    type="messages"
).launch()
