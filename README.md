# convert-to-podcast

Converte links e arquivos em um podcast de dois locutores em português, usando só software e modelos abertos:

- **Roteiro:** [podcastfy](https://github.com/souzatharsis/podcastfy) + LLM via [LiteLLM](https://docs.litellm.ai/docs/providers): [Ollama](https://ollama.com) local (padrão: `qwen3:8b`) ou uma API como a Groq.
- **Voz:** [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0), com vozes prontas e instrução de estilo.
- **Arquivos:** [markitdown](https://github.com/microsoft/markitdown) (PDF, Word, PowerPoint, Excel, HTML, EPUB, texto...).
- **Áudio:** ffmpeg.

Os prompts do podcastfy ficam em `prompts/`, em vez de serem baixados do LangChain Hub a cada execução.

## Instalação

Requer Python 3.11 ou 3.12, [ffmpeg](https://ffmpeg.org) no PATH e, para o roteiro local, o [Ollama](https://ollama.com).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
ollama pull qwen3:8b
```

**GPU NVIDIA (recomendado):** o `pip install` traz o torch só para CPU, onde o Qwen3-TTS fica lento. Para gerar a voz na GPU, troque pelo build CUDA:

```powershell
pip install --force-reinstall --no-deps torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Uso

```powershell
python converter.py https://exemplo.com/artigo                  # página ou vídeo do YouTube
python converter.py relatorio.pdf slides.pptx notas.md          # arquivos (vários de uma vez)
python converter.py --llm groq/openai/gpt-oss-120b artigo.docx  # roteiro via API da Groq
```

O comando imprime o caminho do MP3, salvo em `data/audio/`. Só os 5 áudios mais recentes são mantidos, e o roteiro é apagado ao fim de cada geração. Na primeira execução o Qwen3-TTS baixa o modelo (~4 GB) do HuggingFace.

### LLM por API

Qualquer provedor do LiteLLM funciona com `--llm <provedor>/<modelo>`. A chave vem da variável `<PROVEDOR>_API_KEY`, definida no ambiente ou num arquivo `.env` (ex.: `GROQ_API_KEY=...`).

No plano gratuito da Groq, use `groq/openai/gpt-oss-120b`. O `qwen/qwen3.8-27b` limita a saída a 1.000 tokens por minuto, menos do que o podcastfy pede por chamada.
