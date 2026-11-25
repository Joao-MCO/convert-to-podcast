"""Converte links e arquivos em podcast usando apenas software aberto.

- Roteiro: podcastfy + LLM via LiteLLM (Ollama local ou API, ex.: Groq).
- Voz: Qwen3-TTS (Apache-2.0), registrado como provedor TTS do podcastfy.
- Arquivos: markitdown (PDF, Word, PowerPoint, Excel, HTML, EPUB, texto...).

Uso:
    python converter.py https://exemplo.com/artigo
    python converter.py relatorio.pdf apresentacao.pptx notas.md
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
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
AUDIO_DIR = Path("data/audio")
AUDIOS_TO_KEEP = 5
DEFAULT_LLM = "ollama_chat/qwen3:8b"

QWEN_TTS_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN_TTS_STYLE = "Fale em português do Brasil, de forma natural e descontraída, como num podcast."

CONVERSATION_CONFIG = {
    "output_language": "Brazilian Portuguese",
    "conversation_style": ["envolvente", "didático", "descontraído"],
    "roles_person1": "apresentador que resume o conteúdo",
    "roles_person2": "convidado que faz perguntas e esclarece pontos",
    "user_instructions": (
        "Escreva todo o diálogo em português do Brasil, em linguagem falada e natural. "
        "Não use palavras em inglês, siglas soltas, emojis nem marcações."
    ),
    "text_to_speech": {
        "default_tts_model": "qwen",
        "audio_format": "mp3",
        "ending_message": "Até a próxima!",
        "output_directories": {"audio": str(AUDIO_DIR)},
        "qwen": {
            "default_voices": {"question": "ryan", "answer": "serena"},
            "model": QWEN_TTS_REPO,
        },
    },
}


def to_mp3(samples: np.ndarray, sample_rate: int) -> bytes:
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    segment = AudioSegment(
        (samples * 32767).astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3", bitrate="192k")
    return buffer.getvalue()


class QwenTTS(TTSProvider):
    """Provedor TTS local baseado no Qwen3-TTS (vozes prontas + instrução de estilo)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # O podcastfy lê `self.model`, mas passa nele o nome do provedor ("qwen");
        # o modelo carregado é sempre QWEN_TTS_REPO.
        self.model = model
        self._tts = None

    @property
    def tts(self):
        # Carrega o modelo só quando o áudio for gerado de fato.
        if self._tts is None:
            import torch
            from qwen_tts import Qwen3TTSModel

            cuda = torch.cuda.is_available()
            self._tts = Qwen3TTSModel.from_pretrained(
                QWEN_TTS_REPO,
                device_map="cuda:0" if cuda else "cpu",
                dtype=torch.bfloat16 if cuda else torch.float32,
            )
        return self._tts

    def get_supported_tags(self) -> List[str]:
        return []  # O Qwen3-TTS não interpreta SSML: todas as tags são removidas.

    def split_qa(self, input_text: str, ending_message: str, supported_tags: List[str] = None):
        # O podcastfy só pareia falas alternadas; LLMs às vezes repetem o mesmo
        # locutor, e as tags iriam parar no texto falado. Junta essas falas.
        merged = re.sub(r"</(Person[12])>\s*<\1>", " ", input_text)
        return super().split_qa(merged, ending_message, supported_tags)

    def generate_audio(self, text: str, voice: str, model: str, voice2: str = None) -> bytes:
        self.validate_parameters(text, voice, model)
        wavs, sample_rate = self.tts.generate_custom_voice(
            text=text, speaker=voice, language="Portuguese", instruct=QWEN_TTS_STYLE
        )
        return to_mp3(wavs[0], sample_rate)


TTSProviderFactory.register_provider("qwen", QwenTTS)


def pull_local_prompt(ref: str) -> ChatPromptTemplate:
    """Substitui `hub.pull` do podcastfy pelas cópias em ./prompts.

    O podcastfy baixa seus prompts do LangChain Hub (serviço fechado) a cada
    execução. As versões fixadas por commit em seu config.yaml foram salvas em
    ./prompts como `<nome>@<commit>.json` (a abertura foi ajustada para não
    citar nome de podcast).
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
    # O podcastfy busca `<TTS>_API_KEY` no config; o Qwen3-TTS é local e não usa chave.
    config = load_config()  # também carrega o .env
    config.QWEN_API_KEY = ""
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


def read_file(path: Path) -> str:
    """Extrai o texto de um arquivo local (PDF, Word, PowerPoint, Excel, HTML, texto...)."""
    from markitdown import MarkItDown

    text = MarkItDown().convert(str(path)).text_content.strip()
    if not text:
        raise SystemExit(f"Nenhum texto encontrado em {path}.")
    return text


def prune_audios(keep: int = AUDIOS_TO_KEEP) -> None:
    audios = sorted(AUDIO_DIR.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in audios[keep:]:
        old.unlink()


def convert(inputs: List[str], llm_model: str = DEFAULT_LLM) -> str:
    """Gera o podcast e devolve o caminho do MP3. O roteiro não é guardado."""
    urls: List[str] = []
    texts: List[str] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            texts.append(read_file(path))
        else:
            urls.append(item)  # páginas e vídeos do YouTube o podcastfy extrai sozinho.

    config = build_config()
    api_key_label = api_key_label_for(llm_model)

    with tempfile.TemporaryDirectory() as transcripts_dir:
        conversation_config = deepcopy(CONVERSATION_CONFIG)
        conversation_config["text_to_speech"]["output_directories"]["transcripts"] = transcripts_dir
        audio = generate_podcast(
            urls=urls or None,
            text="\n\n".join(texts) or None,
            tts_model="qwen",
            config=config,
            conversation_config=conversation_config,
            llm_model_name=llm_model,
            api_key_label=api_key_label,
        )

    if not audio:
        raise SystemExit("Falha ao gerar o podcast; veja o log acima.")
    prune_audios()
    return audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte links e arquivos em podcast.")
    parser.add_argument("inputs", nargs="+", help="URLs, vídeos do YouTube ou arquivos (PDF, DOCX, PPTX, TXT...)")
    parser.add_argument("--llm", default=DEFAULT_LLM, help=f"modelo LiteLLM (padrão: {DEFAULT_LLM})")
    args = parser.parse_args()
    print(convert(args.inputs, args.llm))


if __name__ == "__main__":
    main()
