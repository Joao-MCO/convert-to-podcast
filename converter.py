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
import threading
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional

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

# As vozes prontas do Qwen3-TTS têm sotaque estrangeiro em português. Por isso
# clonamos (modelo Base) gravações de vozes brasileiras: ./voices/<voz>.wav, com a
# transcrição exata de cada uma em ./voices/vozes.json.
QWEN_TTS_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICES_DIR = Path(__file__).parent / "voices"

CONVERSATION_CONFIG = {
    "output_language": "Brazilian Portuguese",
    "conversation_style": ["envolvente", "didático", "descontraído"],
    "roles_person1": "apresentador que resume o conteúdo",
    "roles_person2": "convidado que faz perguntas e esclarece pontos",
    "user_instructions": (
        "Escreva todo o diálogo em português do Brasil, em linguagem falada e natural. "
        "Não use palavras em inglês, emojis nem marcações. Escreva números, placares e datas "
        "por extenso e troque siglas pelo nome falado (ex.: 'Atlético Mineiro' em vez de 'Atlético-MG')."
    ),
    "text_to_speech": {
        "default_tts_model": "qwen",
        "audio_format": "mp3",
        "ending_message": "Até a próxima!",
        "output_directories": {"audio": str(AUDIO_DIR)},
        "qwen": {
            "default_voices": {"question": "apresentador", "answer": "convidado"},
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
    """Provedor TTS local: Qwen3-TTS clonando as vozes brasileiras de ./voices.

    O podcastfy cria um provedor novo a cada podcast; o modelo e as vozes ficam
    em atributos de classe para não serem recarregados (leva minutos) quando
    vários podcasts são gerados no mesmo processo, como na interface web.
    """

    _tts = None
    _tts_lock = threading.Lock()
    _voice_prompts = {}
    # Chamado como on_progress(falas_prontas, total_de_falas) durante a narração.
    on_progress: Optional[Callable[[int, int], None]] = None

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # O podcastfy lê `self.model`, mas passa nele o nome do provedor ("qwen");
        # o modelo carregado é sempre QWEN_TTS_REPO.
        self.model = model
        self._done = 0
        self._total = 0

    @property
    def tts(self):
        # Carrega o modelo só quando for usado; a trava evita carregar duas vezes
        # quando a interface pré-carrega em segundo plano.
        with QwenTTS._tts_lock:
            if QwenTTS._tts is None:
                import torch
                from qwen_tts import Qwen3TTSModel

                cuda = torch.cuda.is_available()
                QwenTTS._tts = Qwen3TTSModel.from_pretrained(
                    QWEN_TTS_REPO,
                    device_map="cuda:0" if cuda else "cpu",
                    dtype=torch.bfloat16 if cuda else torch.float32,
                )
        return QwenTTS._tts

    def get_supported_tags(self) -> List[str]:
        return []  # O Qwen3-TTS não interpreta SSML: todas as tags são removidas.

    def split_qa(self, input_text: str, ending_message: str, supported_tags: List[str] = None):
        # O podcastfy só pareia falas alternadas; LLMs às vezes repetem o mesmo
        # locutor, e as tags iriam parar no texto falado. Junta essas falas.
        merged = re.sub(r"</(Person[12])>\s*<\1>", " ", input_text)
        pairs = super().split_qa(merged, ending_message, supported_tags)
        self._done, self._total = 0, 2 * len(pairs)
        return pairs

    def generate_audio(self, text: str, voice: str, model: str, voice2: str = None) -> bytes:
        self.validate_parameters(text, voice, model)
        wavs, sample_rate = self.tts.generate_voice_clone(
            text=text, language="Portuguese", voice_clone_prompt=self.voice_prompt(voice)
        )
        self._done += 1
        if QwenTTS.on_progress:
            QwenTTS.on_progress(self._done, max(self._total, self._done))
        return to_mp3(wavs[0], sample_rate)

    def voice_prompt(self, voice: str):
        # Extrai uma vez as características de cada voz de referência; refaz se a
        # gravação for trocada.
        wav = VOICES_DIR / f"{voice}.wav"
        key = (voice, wav.stat().st_mtime)
        if key not in QwenTTS._voice_prompts:
            transcripts = json.loads((VOICES_DIR / "vozes.json").read_text(encoding="utf-8"))
            QwenTTS._voice_prompts[key] = self.tts.create_voice_clone_prompt(
                ref_audio=str(wav), ref_text=transcripts[voice]
            )
        return QwenTTS._voice_prompts[key]


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


class ConversionError(Exception):
    """Erro esperado na conversão, com mensagem pronta para o usuário."""


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
        raise ConversionError(f"Defina {label} no .env para usar o modelo {llm_model!r}.")
    return label


def read_file(path: Path) -> str:
    """Extrai o texto de um arquivo local (PDF, Word, PowerPoint, Excel, HTML, texto...)."""
    from markitdown import MarkItDown

    text = MarkItDown().convert(str(path)).text_content.strip()
    if not text:
        raise ConversionError(f"Nenhum texto encontrado em {path.name}.")
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
        raise ConversionError("Falha ao gerar o podcast; veja o log no terminal.")
    prune_audios()
    return audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte links e arquivos em podcast.")
    parser.add_argument("inputs", nargs="+", help="URLs, vídeos do YouTube ou arquivos (PDF, DOCX, PPTX, TXT...)")
    parser.add_argument("--llm", default=DEFAULT_LLM, help=f"modelo LiteLLM (padrão: {DEFAULT_LLM})")
    args = parser.parse_args()
    try:
        print(convert(args.inputs, args.llm))
    except ConversionError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
