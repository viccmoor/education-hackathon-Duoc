"""
Módulo en que se deployea la aplicación de Gradio.
"""
from typing import List, Dict
import gradio as gr


def default_response(message: str, history: List[Dict[any, any]]) -> str:
    """Testing response."""
    return "Respuesta predeterminada"


gr.ChatInterface(
    fn=default_response,
    type="messages"
).launch(share=True)
