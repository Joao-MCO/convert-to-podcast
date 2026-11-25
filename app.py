"""Interface web local do gerador de podcasts.

Uso:
    python app.py        # abre http://127.0.0.1:7860 no navegador
"""

import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

# O podcastfy grava roteiros com a codificação padrão do sistema; roda em UTF-8
# (ver converter.py).
if __name__ == "__main__" and not sys.flags.utf8_mode:
    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

os.chdir(Path(__file__).parent)  # data/ e .env são relativos à pasta do projeto

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from converter import AUDIO_DIR, AUDIOS_TO_KEEP, ConversionError, QwenTTS, convert

LLMS = {
    "Groq — nuvem, mais rápido (gpt-oss-120b)": "groq/openai/gpt-oss-120b",
    "Ollama — no seu computador (qwen3:8b)": "ollama_chat/qwen3:8b",
}
FILE_TYPES = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".epub", ".txt", ".md", ".csv"]


def recent_audios() -> list:
    audios = sorted(AUDIO_DIR.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)
    updates = []
    for i in range(AUDIOS_TO_KEEP):
        if i < len(audios):
            when = datetime.fromtimestamp(audios[i].stat().st_mtime).strftime("%d/%m %H:%M")
            updates.append(gr.update(value=str(audios[i]), label=f"Gerado em {when}", visible=True))
        else:
            updates.append(gr.update(value=None, visible=False))
    return updates


def generate(links: str, files, llm_label: str, progress=gr.Progress()):
    inputs = [line.strip() for line in (links or "").splitlines() if line.strip()]
    inputs += [str(f) for f in files or []]
    if not inputs:
        raise gr.Error("Cole pelo menos um link ou envie um arquivo.")

    progress(0, desc="Lendo o conteúdo e escrevendo o roteiro…")
    QwenTTS.on_progress = lambda done, total: progress(done / total, desc=f"Narrando fala {done} de {total}…")
    try:
        audio = convert(inputs, LLMS[llm_label])
    except ConversionError as error:
        raise gr.Error(str(error))
    except Exception as error:  # erros do podcastfy, do LLM ou da extração de páginas
        raise gr.Error(f"Não foi possível gerar o podcast: {error}")
    finally:
        QwenTTS.on_progress = None
    return [str(audio), *recent_audios()]


with gr.Blocks(title="Gerador de Podcasts") as demo:
    gr.Markdown(
        "# 🎙️ Gerador de Podcasts\n"
        "Transforme notícias, artigos e documentos num bate-papo em áudio, "
        "com o João Marcos apresentando e o Vitor de convidado."
    )
    with gr.Row():
        with gr.Column():
            links = gr.Textbox(
                label="Links",
                placeholder="Cole um link por linha (notícias, artigos, vídeos do YouTube…)",
                lines=4,
            )
            files = gr.File(
                label="Arquivos (PDF, Word, PowerPoint, Excel, EPUB, texto…)",
                file_count="multiple",
                file_types=FILE_TYPES,
                type="filepath",
            )
            default_llm = next(iter(LLMS)) if os.environ.get("GROQ_API_KEY") else list(LLMS)[1]
            llm = gr.Radio(list(LLMS), value=default_llm, label="Quem escreve o roteiro")
            button = gr.Button("Gerar podcast", variant="primary", size="lg")
            gr.Markdown("A narração roda na sua placa de vídeo e leva alguns minutos.")
        with gr.Column():
            result = gr.Audio(label="Seu podcast", type="filepath", interactive=False)
            gr.Markdown(f"### Últimos {AUDIOS_TO_KEEP} podcasts")
            history = [
                gr.Audio(type="filepath", interactive=False, visible=False) for _ in range(AUDIOS_TO_KEEP)
            ]

    button.click(generate, [links, files, llm], [result, *history], concurrency_limit=1)
    demo.load(recent_audios, outputs=history)


if __name__ == "__main__":
    # Carrega o modelo de voz em segundo plano para o primeiro podcast sair mais rápido.
    threading.Thread(target=lambda: QwenTTS().tts, daemon=True).start()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    demo.launch(
        server_name="127.0.0.1",
        inbrowser=True,
        allowed_paths=[str(AUDIO_DIR.resolve())],
        theme=gr.themes.Soft(),
    )
