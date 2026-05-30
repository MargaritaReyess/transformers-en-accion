import gradio as gr
import whisper
from transformers import pipeline

asr_model = whisper.load_model("base")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis"
)

def transcribe_and_classify(audio_path):
    if audio_path is None:
        return "No se recibió audio.", ""

    result = asr_model.transcribe(audio_path, language="es")
    text = result["text"].strip()

    if not text:
        return "No se pudo transcribir el audio.", ""

    sentiment = sentiment_model(text)[0]

    label_map = {
        "POS": "Positivo",
        "NEG": "Negativo",
        "NEU": "Neutral"
    }

    label = label_map.get(sentiment["label"], sentiment["label"])
    score = round(sentiment["score"] * 100, 2)

    classification = f"{label} ({score}% de confianza)"

    return text, classification


demo = gr.Interface(
    fn=transcribe_and_classify,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="Graba tu frase"),
    outputs=[
        gr.Textbox(label="Transcripción"),
        gr.Textbox(label="Clasificación de sentimiento")
    ],
    title="De la voz al significado",
    description="Graba una frase en español. El sistema la transcribe automáticamente y después clasifica su sentimiento."
)

demo.launch()
