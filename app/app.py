"""
Módulo en que se deployea la aplicación de Gradio.
"""
from typing import List, Dict, Any
import gradio as gr


def default_response(message: str, history: List[Dict[Any, Any]]) -> str:
    """Testing response."""
    return "Respuesta predeterminada"


with gr.Blocks(
    title="Education Hackathon Duoc",
    theme=gr.themes.Soft,
) as demo:
    chatbot = gr.Chatbot(
        placeholder="¡Hola! Soy tu tutor virtual. Pregúntame lo que quieras.",
        type="messages"
    )
    gr.ChatInterface(
        fn=default_response,
        type="messages",
        chatbot=chatbot
    )

demo.launch(share=True)
