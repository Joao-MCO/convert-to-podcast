# convert-to-podcast

Converte links e arquivos em um podcast de dois locutores em português, usando só software e modelos abertos:

- **Roteiro:** [podcastfy](https://github.com/souzatharsis/podcastfy) + LLM via [LiteLLM](https://docs.litellm.ai/docs/providers): [Ollama](https://ollama.com) local (padrão: `qwen3:8b`) ou uma API como a Groq.
- **Voz:** [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0), clonando (modelo Base) gravações de vozes brasileiras. As vozes prontas dele têm sotaque estrangeiro.
- **Pesquisa:** [ddgs](https://github.com/deedy5/ddgs) (MIT) busca o tema no DuckDuckGo (com Bing de reserva quando o DuckDuckGo limita as buscas) e lê as 4 primeiras páginas que abrirem.
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

### Vozes

As vozes do podcast são clonadas de gravações em `voices/`, que fica fora do git por serem gravações pessoais. Cadastre quantas quiser pela interface web (**Cadastrar nova voz**: envie ou grave 10 a 20 s de fala limpa em português do Brasil, de quem autorizou o uso da própria voz) e escolha o apresentador e o convidado em cada podcast.

Cada voz é um `voices/<Nome>.wav` mais a transcrição exata do que foi falado em `voices/vozes.json` (`{"Nome": "transcrição"}`). A transcrição é feita automaticamente com o Whisper; se ela errar alguma palavra, corrija-a, porque a clonagem fica mais fiel quando o texto bate com o áudio.

## Uso

### Interface web

```powershell
python app.py
```

Abre `http://127.0.0.1:7860` no navegador. Escreva um tema para pesquisar, cole links e/ou envie arquivos, escolha quem escreve o roteiro (Groq ou Ollama) e clique em **Gerar podcast**. A barra mostra qual fala está sendo narrada, e os 5 últimos podcasts ficam listados ao lado. A interface só aceita acesso do próprio computador.

### Linha de comando

```powershell
python converter.py https://exemplo.com/artigo                  # página ou vídeo do YouTube
python converter.py relatorio.pdf slides.pptx notas.md          # arquivos (vários de uma vez)
python converter.py --tema "energia solar no Brasil"            # pesquisa o tema na web
python converter.py --apresentador Naruto --convidado "Tony Stark" artigo.pdf  # escolhe as vozes
python converter.py --llm groq/openai/gpt-oss-120b artigo.docx  # roteiro via API da Groq
```

O comando imprime o caminho do MP3, salvo em `data/audio/`. Só os 5 áudios mais recentes são mantidos, e o roteiro é apagado ao fim de cada geração. Na primeira execução o Qwen3-TTS baixa o modelo (~4 GB) do HuggingFace.

### LLM por API

Qualquer provedor do LiteLLM funciona com `--llm <provedor>/<modelo>`. A chave vem da variável `<PROVEDOR>_API_KEY`, definida no ambiente ou num arquivo `.env` (ex.: `GROQ_API_KEY=...`).

No plano gratuito da Groq, use `groq/openai/gpt-oss-120b`. O `qwen/qwen3.8-27b` limita a saída a 1.000 tokens por minuto, menos do que o podcastfy pede por chamada.
