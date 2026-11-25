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
import shutil
import subprocess
import sys
import tempfile
import threading
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

# O podcastfy lê e grava roteiros com a codificação padrão do sistema (cp1252 no
# Windows); relança em modo UTF-8, antes dos imports pesados, para aceitar
# qualquer caractere que o LLM gere.
if __name__ == "__main__" and not sys.flags.utf8_mode:
    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))


def find_ffmpeg() -> bool:
    """Garante o ffmpeg no PATH deste processo; devolve se ele foi encontrado.

    No Windows, um terminal (ou o VS Code) aberto antes de instalar o ffmpeg não
    enxerga o PATH novo; nesse caso lê o PATH atual do usuário e do sistema.
    """
    if shutil.which("ffmpeg"):
        return True
    if os.name == "nt":
        import winreg

        keys = [
            (winreg.HKEY_CURRENT_USER, r"Environment"),
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        ]
        for root, subkey in keys:
            try:
                with winreg.OpenKey(root, subkey) as key:
                    path, _ = winreg.QueryValueEx(key, "Path")
            except OSError:
                continue
            os.environ["PATH"] += os.pathsep + os.path.expandvars(path)
    return shutil.which("ffmpeg") is not None


find_ffmpeg()  # antes do pydub, que procura o ffmpeg ao ser importado

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
RESEARCH_PAGES = 4
RESEARCH_PAGE_CHARS = 3500

# As vozes prontas do Qwen3-TTS têm sotaque estrangeiro em português. Por isso
# clonamos (modelo Base) gravações de vozes brasileiras: ./voices/<Nome>.wav, com a
# transcrição exata de cada uma em ./voices/vozes.json ({"Nome": "transcrição"}).
QWEN_TTS_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICES_DIR = Path(__file__).parent / "voices"
VOICES_FILE = VOICES_DIR / "vozes.json"
VOICE_SAMPLE_RATE = 24000
VOICE_SECONDS = (5, 30)  # duração aceita para a gravação de referência
WHISPER_MODEL = "openai/whisper-large-v3-turbo"

CONVERSATION_CONFIG = {
    "output_language": "Brazilian Portuguese",
    "conversation_style": ["envolvente", "didático", "descontraído"],
    # roles_person1/2 e as vozes são definidos em convert() com os nomes escolhidos.
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
        "qwen": {"model": QWEN_TTS_REPO},
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
            QwenTTS._voice_prompts[key] = self.tts.create_voice_clone_prompt(
                ref_audio=str(wav), ref_text=load_voices()[voice]
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


@dataclass
class Source:
    title: str
    url: str
    text: str


@dataclass
class Podcast:
    audio: str
    sources: List[Source]


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


def load_voices() -> dict:
    """Vozes cadastradas ({nome: transcrição}) cuja gravação existe em ./voices."""
    if not VOICES_FILE.is_file():
        return {}
    voices = json.loads(VOICES_FILE.read_text(encoding="utf-8"))
    return {name: text for name, text in voices.items() if (VOICES_DIR / f"{name}.wav").is_file()}


_whisper = None


def transcribe(samples: np.ndarray) -> str:
    """Transcreve áudio de 24 kHz com o Whisper, na CPU para não disputar a GPU com a voz."""
    global _whisper
    import librosa
    from transformers import pipeline

    if _whisper is None:
        _whisper = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, device="cpu")
    audio16 = librosa.resample(samples, orig_sr=VOICE_SAMPLE_RATE, target_sr=16000)
    result = _whisper({"raw": audio16, "sampling_rate": 16000}, generate_kwargs={"language": "portuguese"})
    return result["text"].strip()


def add_voice(name: str, audio_path: str, transcript: Optional[str] = None) -> str:
    """Cadastra uma voz a partir de uma gravação e devolve a transcrição usada.

    Sem `transcript`, a gravação é transcrita com o Whisper. A clonagem fica mais
    fiel quando a transcrição bate exatamente com o que foi falado.
    """
    import librosa
    import soundfile as sf

    name = " ".join((name or "").split())
    if not re.fullmatch(r"[\w\- ]{1,40}", name):
        raise ConversionError("Use um nome de até 40 letras, números, espaços ou hífens.")
    samples, _ = librosa.load(audio_path, sr=VOICE_SAMPLE_RATE, mono=True)
    seconds = len(samples) / VOICE_SAMPLE_RATE
    if not VOICE_SECONDS[0] <= seconds <= VOICE_SECONDS[1]:
        raise ConversionError(
            f"A gravação tem {seconds:.0f} s; use entre {VOICE_SECONDS[0]} e {VOICE_SECONDS[1]} s de fala."
        )
    transcript = (transcript or "").strip() or transcribe(samples)

    VOICES_DIR.mkdir(exist_ok=True)
    sf.write(VOICES_DIR / f"{name}.wav", samples, VOICE_SAMPLE_RATE)
    voices = json.loads(VOICES_FILE.read_text(encoding="utf-8")) if VOICES_FILE.is_file() else {}
    voices[name] = transcript
    VOICES_FILE.write_text(json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8")
    return transcript


def research(topic: str, max_pages: int = RESEARCH_PAGES) -> List[Source]:
    """Pesquisa o tema no DuckDuckGo e lê as primeiras páginas que abrirem.

    O texto de cada página é cortado para caber no limite de tokens por minuto
    do plano gratuito da Groq.
    """
    from ddgs import DDGS

    # O DuckDuckGo limita buscas seguidas e às vezes responde vazio, assim como
    # os outros mecanismos que a ddgs consulta; tenta cada um até algum responder.
    results, error = [], None
    for backend in ("duckduckgo", "auto", "yahoo", "brave", "google", "bing", "startpage"):
        try:
            results = DDGS().text(topic, region="br-pt", max_results=max_pages * 2, backend=backend)
            break
        except Exception as e:
            error = e
    if not results:
        raise ConversionError(f"A pesquisa na web falhou: {error}")

    extractor = WebsiteExtractor()
    sources: List[Source] = []
    for result in results:
        try:
            text = extractor.extract_content(result["href"])
        except Exception:
            continue  # página fora do ar, bloqueada ou lenta: tenta a próxima
        if len(text) < 300:
            continue
        sources.append(Source(result["title"], result["href"], text[:RESEARCH_PAGE_CHARS]))
        if len(sources) == max_pages:
            break
    if not sources:
        raise ConversionError(f"Nenhuma página encontrada sobre {topic!r}.")
    return sources


def prune_audios(keep: int = AUDIOS_TO_KEEP) -> None:
    audios = sorted(AUDIO_DIR.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in audios[keep:]:
        old.unlink()


def convert(
    inputs: List[str],
    llm_model: str = DEFAULT_LLM,
    topic: Optional[str] = None,
    host: Optional[str] = None,
    guest: Optional[str] = None,
) -> Podcast:
    """Gera o podcast a partir de links, arquivos e/ou um tema pesquisado na web.

    `host` e `guest` são vozes cadastradas (padrão: as duas primeiras). O roteiro
    não é guardado; só os últimos AUDIOS_TO_KEEP MP3s ficam em data/audio.
    """
    if not find_ffmpeg():
        raise ConversionError("ffmpeg não encontrado. Instale com: winget install Gyan.FFmpeg")
    voices = list(load_voices())
    if len(voices) < 2:
        raise ConversionError("Cadastre pelo menos duas vozes (apresentador e convidado).")
    host = host or voices[0]
    guest = guest or next(v for v in voices if v != host)
    for voice in (host, guest):
        if voice not in voices:
            raise ConversionError(f"Voz {voice!r} não cadastrada. Disponíveis: {', '.join(voices)}.")
    if host == guest:
        raise ConversionError("Escolha vozes diferentes para o apresentador e o convidado.")

    urls: List[str] = []
    texts: List[str] = []
    sources = research(topic) if topic else []
    texts += [f"Fonte: {s.title} ({s.url})\n{s.text}" for s in sources]
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
        conversation_config["roles_person1"] = f"{host}, apresentador que resume o conteúdo"
        conversation_config["roles_person2"] = f"{guest}, convidado que faz perguntas e esclarece pontos"
        tts_config = conversation_config["text_to_speech"]
        tts_config["output_directories"]["transcripts"] = transcripts_dir
        tts_config["qwen"]["default_voices"] = {"question": host, "answer": guest}
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
    return Podcast(audio, sources)


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte links, arquivos e temas em podcast.")
    parser.add_argument("inputs", nargs="*", help="URLs, vídeos do YouTube ou arquivos (PDF, DOCX, PPTX, TXT...)")
    parser.add_argument("--tema", help="tema a pesquisar na web (DuckDuckGo)")
    parser.add_argument("--llm", default=DEFAULT_LLM, help=f"modelo LiteLLM (padrão: {DEFAULT_LLM})")
    parser.add_argument("--apresentador", help="voz do apresentador (padrão: a primeira cadastrada)")
    parser.add_argument("--convidado", help="voz do convidado (padrão: a segunda cadastrada)")
    args = parser.parse_args()
    if not args.inputs and not args.tema:
        parser.error("informe links, arquivos ou --tema")
    try:
        podcast = convert(args.inputs, args.llm, args.tema, args.apresentador, args.convidado)
    except ConversionError as error:
        sys.exit(str(error))
    for source in podcast.sources:
        print(f"Fonte: {source.title} — {source.url}")
    print(podcast.audio)


if __name__ == "__main__":
    main()
