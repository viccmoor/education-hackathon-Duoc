"""
Módulo en que se deployea la aplicación de Gradio.
"""
from typing import List, Dict
import gradio as gr


def default_response(message: str, history: List[Dict[any, any]]) -> str:
    """Testing response."""
    return "Respuesta predeterminada"


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        placeholder="¡Hola! Soy tu tutor virtual. Pregúntame lo que quieras.",
    )
    gr.ChatInterface(
        fn=default_response,
        type="messages",
        chatbot=chatbot
    )

demo.launch(share=True)
