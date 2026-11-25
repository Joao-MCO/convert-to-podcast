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

from converter import AUDIO_DIR, AUDIOS_TO_KEEP, ConversionError, QwenTTS, add_voice, convert, load_voices

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


READ_ALOUD = (
    "Oi, tudo bem? Hoje eu vou contar uma novidade que me deixou bem animado. "
    "A gente vai conversar com calma, explicar cada detalhe e, no final, "
    "eu quero saber o que você achou. Bora começar?"
)


def voice_choices():
    """Atualiza as listas de vozes: apresentador = 1ª voz, convidado = 2ª."""
    voices = list(load_voices())
    host = voices[0] if voices else None
    guest = voices[1] if len(voices) > 1 else None
    return gr.update(choices=voices, value=host), gr.update(choices=voices, value=guest)


def save_voice(name: str, audio: str, transcript: str, consent: bool):
    if not consent:
        raise gr.Error("Confirme que a pessoa autorizou o uso da voz dela.")
    if not audio:
        raise gr.Error("Envie ou grave um áudio.")
    try:
        text = add_voice(name, audio, transcript)
    except ConversionError as error:
        raise gr.Error(str(error))
    voices = list(load_voices())
    gr.Info(f"Voz “{name.strip()}” cadastrada.")
    return (
        f"Transcrição usada: _{text}_\n\nSe alguma palavra estiver errada, corrija no campo acima e salve de novo.",
        gr.update(choices=voices),
        gr.update(choices=voices),
    )


def generate(topic: str, links: str, files, llm_label: str, host: str, guest: str, progress=gr.Progress()):
    topic = (topic or "").strip() or None
    inputs = [line.strip() for line in (links or "").splitlines() if line.strip()]
    inputs += [str(f) for f in files or []]
    if not inputs and not topic:
        raise gr.Error("Informe um tema, cole um link ou envie um arquivo.")

    step = "Pesquisando na web, lendo as fontes" if topic else "Lendo o conteúdo"
    progress(0, desc=f"{step} e escrevendo o roteiro…")
    QwenTTS.on_progress = lambda done, total: progress(done / total, desc=f"Narrando fala {done} de {total}…")
    try:
        podcast = convert(inputs, LLMS[llm_label], topic, host, guest)
    except ConversionError as error:
        raise gr.Error(str(error))
    except Exception as error:  # erros do podcastfy, do LLM ou da extração de páginas
        raise gr.Error(f"Não foi possível gerar o podcast: {error}")
    finally:
        QwenTTS.on_progress = None
    sources = "\n".join(f"- [{s.title}]({s.url})" for s in podcast.sources)
    sources_update = gr.update(value=f"**Fontes pesquisadas**\n\n{sources}", visible=bool(sources))
    return [str(podcast.audio), sources_update, *recent_audios()]


with gr.Blocks(title="Gerador de Podcasts") as demo:
    gr.Markdown(
        "# 🎙️ Gerador de Podcasts\n"
        "Transforme um tema pesquisado na web, notícias, artigos e documentos num bate-papo em áudio, "
        "com as vozes que você escolher."
    )
    with gr.Row():
        with gr.Column():
            topic = gr.Textbox(
                label="Tema para pesquisar (opcional)",
                placeholder="Ex.: queda de rendimento do Atlético-MG em 2026",
            )
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
            with gr.Row():
                host = gr.Dropdown(label="Voz do apresentador")
                guest = gr.Dropdown(label="Voz do convidado")
            button = gr.Button("Gerar podcast", variant="primary", size="lg")
            gr.Markdown("A narração roda na sua placa de vídeo e leva alguns minutos.")

            with gr.Accordion("Cadastrar nova voz", open=False):
                gr.Markdown(
                    "Envie ou grave **10 a 20 segundos** de fala clara, sem música nem barulho. "
                    f"Uma sugestão de texto para ler:\n\n> {READ_ALOUD}"
                )
                voice_name = gr.Textbox(label="Nome da voz", placeholder="Ex.: Maria")
                voice_audio = gr.Audio(label="Gravação", sources=["upload", "microphone"], type="filepath")
                voice_text = gr.Textbox(
                    label="Transcrição (opcional)",
                    placeholder="Deixe em branco para transcrever automaticamente",
                    lines=2,
                )
                voice_consent = gr.Checkbox(label="A pessoa desta voz autorizou o uso dela para gerar podcasts.")
                voice_button = gr.Button("Salvar voz")
                voice_status = gr.Markdown()
        with gr.Column():
            result = gr.Audio(label="Seu podcast", type="filepath", interactive=False)
            sources = gr.Markdown(visible=False)
            gr.Markdown(f"### Últimos {AUDIOS_TO_KEEP} podcasts")
            history = [
                gr.Audio(type="filepath", interactive=False, visible=False) for _ in range(AUDIOS_TO_KEEP)
            ]

    button.click(
        generate,
        [topic, links, files, llm, host, guest],
        [result, sources, *history],
        concurrency_limit=1,
        show_progress_on=[result],  # senão a barra aparece em cada player do histórico
    )
    voice_button.click(
        save_voice, [voice_name, voice_audio, voice_text, voice_consent], [voice_status, host, guest]
    )
    demo.load(recent_audios, outputs=history)
    demo.load(voice_choices, outputs=[host, guest])


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
