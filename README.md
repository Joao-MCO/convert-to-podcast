# convert-to-podcast

Converte URLs, PDFs e textos em um podcast de dois locutores em português, usando só software e modelos abertos:

- **Roteiro:** [podcastfy](https://github.com/souzatharsis/podcastfy) + LLM local via [Ollama](https://ollama.com) (padrão: `qwen3:8b`).
- **Voz:** [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0), vozes `pm_alex` e `pf_dora`.
- **Áudio:** ffmpeg.

Os prompts do podcastfy ficam em `prompts/`, em vez de serem baixados do LangChain Hub a cada execução.

## Instalação

Requer Python 3.11 ou 3.12 (o Kokoro ainda não suporta 3.13+; o podcastfy exige 3.11+), [Ollama](https://ollama.com) e [ffmpeg](https://ffmpeg.org) no PATH.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
ollama pull qwen3:8b
```

**GPU NVIDIA (opcional):** o `pip install` traz o torch só para CPU. Para gerar a voz na GPU, troque pelo build CUDA:

```powershell
pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu128
```

## Uso

```powershell
python converter.py https://exemplo.com/artigo        # URL, PDF, YouTube ou .txt/.md
python converter.py --transcript-only artigo.pdf      # só o roteiro
python converter.py --transcript data\transcripts\transcript_x.txt  # áudio a partir de um roteiro editado
python converter.py --llm ollama_chat/qwen3.5:4b nota.txt           # outro modelo do Ollama
python converter.py --llm groq/openai/gpt-oss-120b artigo.pdf       # roteiro via API da Groq
```

### LLM por API

Qualquer provedor do [LiteLLM](https://docs.litellm.ai/docs/providers) funciona com `--llm <provedor>/<modelo>`. A chave vem da variável `<PROVEDOR>_API_KEY`, definida no ambiente ou num arquivo `.env` (ex.: `GROQ_API_KEY=...`).

No plano gratuito da Groq, use `groq/openai/gpt-oss-120b`. O `qwen/qwen3.8-27b` limita a saída a 1.000 tokens por minuto, menos do que o podcastfy pede por chamada.

Roteiros são salvos em `data/transcripts/` e áudios em `data/audio/`. Na primeira execução o Kokoro baixa o modelo (~330 MB) do HuggingFace.
