"""Converte URLs, PDFs ou textos em podcast usando apenas software aberto.

- Roteiro: podcastfy + LLM local via Ollama (LiteLLM).
- Voz: Kokoro-82M (Apache-2.0), registrado como provedor TTS do podcastfy.
- Áudio: pydub + ffmpeg.

Uso:
    python converter.py https://exemplo.com/artigo
    python converter.py documento.pdf notas.txt
    python converter.py --transcript data/transcripts/transcript_x.txt
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# O podcastfy lê e grava roteiros com a codificação padrão do sistema (cp1252 no
# Windows); relança em modo UTF-8, antes dos imports pesados, para aceitar
# qualquer caractere que o LLM gere.
if __name__ == "__main__" and not sys.flags.utf8_mode:
    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import numpy as np
from langchain_core.load import load as load_langchain
from langchain_core.prompts import ChatPromptTemplate
from pydub import AudioSegment

import podcastfy.content_generator
from podcastfy.client import generate_podcast
from podcastfy.content_parser.website_extractor import WebsiteExtractor
from podcastfy.tts.base import TTSProvider
from podcastfy.tts.factory import TTSProviderFactory
from podcastfy.utils.config import load_config

PROMPTS_DIR = Path(__file__).parent / "prompts"
KOKORO_REPO ="hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000
DEFAULT_LLM = "ollama_chat/qwen3:8b"

CONVERSATION_CONFIG = {
    "output_language": "Brazilian Portuguese",
    "podcast_name": "Convert to Podcast",
    "podcast_tagline": "Seu podcast gerado por IA aberta",
    "conversation_style": ["envolvente", "didático", "descontraído"],
    "roles_person1": "apresentador que resume o conteúdo",
    "roles_person2": "convidado que faz perguntas e esclarece pontos",
    "user_instructions": "Escreva todo o diálogo em português do Brasil.",
    "text_to_speech": {
        "default_tts_model": "kokoro",
        "audio_format": "mp3",
        "ending_message": "Até a próxima!",
        "output_directories": {
            "transcripts": "./data/transcripts",
            "audio": "./data/audio",
        },
        # Vozes pt-BR do Kokoro: pf_dora (feminina), pm_alex e pm_santa (masculinas).
        "kokoro": {
            "default_voices": {"question": "pm_alex", "answer": "pf_dora"},
            "model": "kokoro",
        },
    },
}


class KokoroTTS(TTSProvider):
    """Provedor TTS local baseado no Kokoro-82M."""

    def __init__(self, api_key: Optional[str] = None, model: str = "kokoro"):
        self.model = model
        self._pipeline = None

    @property
    def pipeline(self):
        # Carrega o modelo só quando o áudio for gerado de fato.
        if self._pipeline is None:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code="p", repo_id=KOKORO_REPO)
        return self._pipeline

    def get_supported_tags(self) -> List[str]:
        return []  # Kokoro não interpreta SSML: todas as tags são removidas.

    def split_qa(self, input_text: str, ending_message: str, supported_tags: List[str] = None):
        # O podcastfy só pareia falas alternadas; LLMs locais às vezes repetem o
        # mesmo locutor, e as tags iriam parar no texto falado. Junta essas falas.
        merged = re.sub(r"</(Person[12])>\s*<\1>", " ", input_text)
        return super().split_qa(merged, ending_message, supported_tags)

    def generate_audio(self, text: str, voice: str, model: str, voice2: str = None) -> bytes:
        self.validate_parameters(text, voice, model)
        chunks = [
            audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
            for _, _, audio in self.pipeline(text, voice=voice)
            if audio is not None
        ]
        if not chunks:
            raise RuntimeError(f"Kokoro não gerou áudio para: {text[:80]!r}")

        samples = np.clip(np.concatenate(chunks), -1.0, 1.0)
        segment = AudioSegment(
            (samples * 32767).astype(np.int16).tobytes(),
            frame_rate=KOKORO_SAMPLE_RATE,
            sample_width=2,
            channels=1,
        )
        buffer = io.BytesIO()
        segment.export(buffer, format="mp3", bitrate="192k")
        return buffer.getvalue()


TTSProviderFactory.register_provider("kokoro", KokoroTTS)


def pull_local_prompt(ref: str) -> ChatPromptTemplate:
    """Substitui `hub.pull` do podcastfy pelas cópias em ./prompts.

    O podcastfy baixa seus prompts do LangChain Hub (serviço fechado) a cada
    execução. As versões fixadas por commit em seu config.yaml foram salvas em
    ./prompts como `<nome>@<commit>.json`.
    """
    path = PROMPTS_DIR / (ref.split("/")[-1].replace(":", "@") + ".json")
    if not path.is_file():
        raise FileNotFoundError(f"Prompt {ref!r} não encontrado em {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return load_langchain(data, allowed_objects="core")


podcastfy.content_generator.hub.pull = pull_local_prompt


def fetch_rendered_html(self: WebsiteExtractor, url: str) -> str:
    """Substitui `WebsiteExtractor.fetch_with_playwright` do podcastfy.

    O original espera `networkidle` por 10 s, o que nunca acontece em sites com
    anúncios e rastreadores (ex.: ge.globo.com). Aqui basta o HTML carregar; a
    ociosidade da rede é só uma espera curta e opcional para conteúdo via JS.
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=self.user_agent, ignore_https_errors=True)
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            try:
                page.wait_for_load_state("networkidle", timeout=5_000)
            except PlaywrightTimeout:
                pass
            return page.content()
        finally:
            browser.close()


WebsiteExtractor.fetch_with_playwright = fetch_rendered_html



def build_config():
    # O podcastfy busca `<TTS>_API_KEY` no config; o Kokoro é local e não usa chave.
    config = load_config()
    config.KOKORO_API_KEY = ""
    return config


def api_key_label_for(llm_model: str) -> str:
    """Variável de ambiente com a chave do provedor LiteLLM (ex.: groq/... -> GROQ_API_KEY).

    O podcastfy exige que ela exista; o Ollama local não usa chave.
    """
    provider = llm_model.split("/", 1)[0]
    if provider.startswith("ollama"):
        os.environ.setdefault("OLLAMA_API_KEY", "ollama")
        return "OLLAMA_API_KEY"
    label = f"{provider.upper()}_API_KEY"
    if not os.environ.get(label):
        raise SystemExit(f"Defina {label} no .env para usar o modelo {llm_model!r}.")
    return label


def convert(
    inputs: List[str],
    llm_model: str = DEFAULT_LLM,
    transcript_file: Optional[str] = None,
    transcript_only: bool = False,
) -> Optional[str]:
    urls: List[str] = []
    texts: List[str] = []
    for item in inputs:
        path = Path(item)
        if path.suffix.lower() in {".txt", ".md"} and path.is_file():
            texts.append(path.read_text(encoding="utf-8"))
        else:
            urls.append(item)  # URLs, PDFs e vídeos do YouTube o podcastfy extrai sozinho.

    config = build_config()  # também carrega o .env
    api_key_label = api_key_label_for(llm_model)

    return generate_podcast(
        urls=urls or None,
        text="\n\n".join(texts) or None,
        transcript_file=transcript_file,
        transcript_only=transcript_only,
        tts_model="kokoro",
        config=config,
        conversation_config=CONVERSATION_CONFIG,
        llm_model_name=llm_model,
        api_key_label=api_key_label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte conteúdo em podcast (Ollama + Kokoro).")
    parser.add_argument("inputs", nargs="*", help="URLs, PDFs ou arquivos .txt/.md")
    parser.add_argument("--llm", default=DEFAULT_LLM, help=f"modelo LiteLLM (padrão: {DEFAULT_LLM})")
    parser.add_argument("--transcript", help="gera o áudio a partir de um roteiro já existente")
    parser.add_argument("--transcript-only", action="store_true", help="gera só o roteiro, sem áudio")
    args = parser.parse_args()

    if not args.inputs and not args.transcript:
        parser.error("informe ao menos uma entrada ou --transcript")

    result = convert(args.inputs, args.llm, args.transcript, args.transcript_only)
    print(result or "Roteiro gerado em ./data/transcripts")


if __name__ == "__main__":
    main()
