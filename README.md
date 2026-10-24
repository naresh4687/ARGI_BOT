# AgriBot AI — Intelligent Bilingual Agriculture Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers farming questions in
**English and Tamil**, grounded in a curated set of documents from the Tamil Nadu
Agricultural University (TNAU) Agritech Portal, with source citations under every
answer.

---

## 1. Abstract

Farmers in Tamil Nadu need quick, understandable and trustworthy answers about crops,
soil, irrigation, nutrients, pests and diseases, preferably in Tamil. General-purpose
chatbots can produce fluent but unsupported or invented advice, which is risky when it
involves pesticides, fertilizers and money. AgriBot AI combines multilingual semantic
search over verified agricultural documents with a large language model (LLM) that is
instructed, and checked in code, to answer only from the retrieved passages. When the
knowledge base has no sufficiently relevant passage, the system says so instead of
guessing. The system works fully offline in a clearly labelled demonstration mode, and
uses a hosted LLM API or a local Ollama model for real answer generation.

## 2. Problem statement

- Agricultural extension information is spread across many web pages and PDFs, mostly
  in English, and is hard for farmers to search.
- Generic AI chatbots can *hallucinate*: invent dosages, product approvals, sources or
  diagnoses.
- Tamil-language support in AI tools is weaker than English support.

## 3. Objectives

1. Answer agricultural questions in English and Tamil in simple language.
2. Ground every answer in verified passages and show the real source (title, publisher,
   page, link).
3. Refuse or ask for clarification when verified information is not available.
4. Never invent pesticide/fertilizer quantities; remind users to follow labels and
   local expert advice.
5. Run on a student laptop (CPU only) and deploy on a free hosting tier.

## 4. Limitations of existing systems

| Existing approach | Limitation |
|---|---|
| Static portals (e.g. TNAU Agritech) | Excellent content but navigation-based; no question answering; Tamil and English pages are separate |
| Generic chatbots (LLM only) | No guarantee answers come from agricultural authorities; can invent citations and quantities |
| Keyword search | Misses synonyms and Tamil word forms (Tamil is agglutinative: பாசனம் / பாசனத்தின்) |
| Helplines / KVK visits | Authoritative but not instant |

## 5. Proposed system

- A **manifest-based ingestion pipeline** downloads a small, verified starter corpus
  (42 documents: 24 English, 17 Tamil, 1 PDF) and records source metadata.
- **Hybrid retrieval**: multilingual sentence embeddings (with per-language centring)
  and character n-gram TF-IDF, combined with language-specific weights, then a
  **relevance threshold enforced in code**. Passages from other documents that score
  well below the best match are dropped, so the LLM is not shown neighbouring topics.
- A **grounded prompt** passes only above-threshold passages, wrapped as untrusted
  evidence, to the LLM.
- **Response validation in code**: citations to non-existent sources are removed, URLs
  not in the source metadata are stripped, quantities that do not appear in the sources
  are flagged, and a missing Tamil answer is detected.
- A **Streamlit** interface with English/Tamil selection, examples, a sources panel,
  safety notes and clear status badges.

## 6. Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| UI | Streamlit 1.66 |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d, CPU) |
| Vector search | FAISS (`IndexFlatIP`, exact cosine) |
| Lexical search | Character 4-gram TF-IDF (standard library only) |
| PDF processing | PyMuPDF |
| Numerics | NumPy |
| LLM | Any OpenAI-compatible hosted API (Groq, Gemini, OpenRouter, …) **or** local Ollama **or** offline demo |
| Config | Environment variables / `.env` / Streamlit secrets |
| Tests | pytest + Streamlit `AppTest` |

## 7. Architecture and RAG workflow

```
            ┌──────────────── offline, once ────────────────┐
manifest ─► fetch_sources ─► data/raw (*.txt, *.pdf + .meta.json)
                               │
                               ▼
                 ingest: load → clean → chunk (en 400 / ta 300 chars)
                               │   title-prefixed search text
                               ▼
               embed (MiniLM) → centre per language → FAISS + metadata
            └───────────────────────────────────────────────┘

User question
  → input validation (empty / length / control chars / supported language)
  → language detection (Tamil Unicode block ratio)
  → query embedding, centred with that language's mean
  → hybrid score = w·dense + (1−w)·lexical   (w: en 0.5, ta 0.2)
  → relevance threshold (en 0.25, ta 0.18)  ── none pass ─► "verified information unavailable"
  → relative cutoff: other documents must score ≥ 0.85 × best
  → top-k passages (max 2 per document), labelled S1…Sk
  → prompt: rules + <passage> evidence (injection-neutralised) + question
  → LLM (hosted / Ollama)                    ── offline ─► show passages, labelled "not AI"
  → validation: citations, URLs, quantities, answer language
  → render answer + badges + sources built from stored metadata
  → session chat history
```

**Why per-language centring?** The multilingual MiniLM model places almost all Tamil
text in one narrow region of the vector space: before the fix, an unrelated Tamil
question ("recommend a good movie") scored 0.84 cosine against Tamil farming passages.
Subtracting each language's mean vector removed that shared component; the worst
off-topic score dropped to 0.44.

**Why hybrid with lexical?** Dense retrieval alone found the correct Tamil document in
the top 4 for only 5/12 Tamil questions. Character n-grams match shared Tamil stems
without a stemmer. With centring, hybrid scoring and title-prefixed chunks, Tamil
Hit@4 reached 12/12 on the tuning set and 9/10 on the held-out set (see §16).

Code map:

| Module | Responsibility |
|---|---|
| `config/settings.py` | Typed settings from env / secrets; secret redaction; logging |
| `knowledge_base/fetch_sources.py` | Download manifest URLs → cleaned text + metadata sidecars |
| `knowledge_base/preprocess.py` | HTML→text, Unicode/whitespace cleaning, chunking with overlap |
| `knowledge_base/ingest.py` | Loaders (PDF/TXT/MD/JSON), chunk metadata, build CLI |
| `knowledge_base/embeddings.py` | Sentence-transformers wrapper (lazy, CPU) |
| `knowledge_base/vector_store.py` | FAISS store, language centring, save/load/validate |
| `knowledge_base/lexical.py` | Character n-gram TF-IDF index |
| `chatbot/language.py` | Language detection, bilingual UI strings, examples |
| `chatbot/retriever.py` | Hybrid retrieval + threshold |
| `chatbot/prompts.py` | System prompt and evidence formatting |
| `chatbot/safety.py` | Input validation, injection neutralising, citation/URL validation, risk detection |
| `chatbot/llm_client.py` | Provider interface, OpenAI-compatible + Ollama, retries, error mapping |
| `chatbot/response_service.py` | End-to-end pipeline and response checks |
| `chatbot/bootstrap.py` | Wiring; auto-rebuild of missing/stale index |
| `ui/*.py`, `app.py` | Streamlit interface |
| `evaluation/` | Question sets, evaluation runner, measured reports |

## 8. Folder structure

```
.
├── app.py                      # Streamlit entry point
├── requirements.txt            # pinned runtime deps (CPU torch)
├── requirements-dev.txt        # + pytest
├── .env.example                # placeholder configuration
├── .streamlit/
│   ├── config.toml             # theme, server settings
│   └── secrets.toml.example
├── config/settings.py
├── chatbot/                    # retrieval, prompts, safety, LLM, pipeline
├── knowledge_base/             # fetch, preprocess, ingest, embeddings, FAISS, lexical
├── ui/                         # sidebar, chat, components, info pages
├── data/
│   ├── README.md               # how to add verified documents
│   ├── sources/manifest.json   # verified source list
│   ├── raw/                    # indexed documents + .meta.json sidecars
│   └── processed/              # FAISS index (committed; auto-rebuilt if missing)
├── evaluation/
│   ├── eval_questions.json     # tuning set (36 questions)
│   ├── heldout_questions.json  # held-out set (28 questions)
│   ├── run_eval.py
│   └── results/                # measured reports
├── tests/                      # pytest suite
└── docs/
    ├── TEST_REPORT.md
    └── REPORT_OUTLINE.md
```

## 9. Dataset sources and licensing

The starter corpus (`data/sources/manifest.json`) contains 41 pages from the
**TNAU Agritech Portal** (https://agritech.tnau.ac.in/) and one PDF hosted there,
*Natural Pest and Disease Control* by **HDRA – the organic organisation**:

- Irrigation: critical stages, drip, sprinkler, water quality (EN + TA)
- Nutrient management: INM, application methods, fertilizer-use tips, organic matter (EN + TA)
- Mineral nutrition: essential elements, deficiency symptoms, diagnosing disorders (EN + TA)
- Rice SRI: nursery, irrigation, weed management (EN + TA)
- IPM: introduction, components, paddy IPM (EN + TA)
- Rice pests (yellow stem borer, gall midge, leaf folder) and diseases (blast, brown
  spot, bacterial leaf blight, sheath blight) (EN, with TA for stem borer, blast, BLB)
- Crop selection (EN)

Every URL was checked to return HTTP 200 with substantive text on 2026-10-24. The
portal's robots.txt has no rules for general user agents. The fetcher downloads only
the listed URLs (no crawling), with a 1.5 s delay and a descriptive User-Agent.
Publication dates are `null` because the pages do not state them. ICAR and IMD sources
were not added: IMD provides forecasts, not reference text, so live weather is out of
scope.

**Licensing:** TNAU pages carry "© All Rights Reserved" and the portal disclaimer grants
no explicit reuse licence. The corpus is used as attributed excerpts for
non-commercial educational purposes. For that reason the public repository does
**not** contain `data/raw/` or the index built from it (`data/processed/`): both are
git-ignored, and users download the pages themselves with the fetch script, which
records the source URL and retrieval date for each one.

## 10. Installation

Requirements: Python 3.10–3.12, about 3 GB of free disk (CPU PyTorch plus the 470 MB
embedding model), 4 GB RAM or more, internet access for the first run.

```bash
git clone https://github.com/naresh4687/ARGI_BOT.git && cd ARGI_BOT
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # use requirements.txt if you don't need tests
cp .env.example .env                 # then edit .env

# The source documents are not in the repository (see §9). Download and index them:
python -m knowledge_base.fetch_sources --allow-insecure-tls   # ~1–2 min
python -m knowledge_base.ingest
```

## 11. Running locally

```bash
streamlit run app.py
```

Open http://localhost:8501. The first start downloads the embedding model (≈470 MB)
from Hugging Face and loads the index built in §10. With the default
`LLM_PROVIDER=offline` the app works without any API key, showing retrieved passages
labelled **"Offline demo · not AI-generated"**.

## 12. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `offline` | `offline`, `openai_compatible` or `ollama` |
| `LLM_BASE_URL` | Groq endpoint | Base URL of an OpenAI-compatible API |
| `LLM_MODEL` | – | Model name at that provider (required) |
| `LLM_API_KEY` | – | API key (required for hosted providers). Never logged |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `gemma3:4b` | Local Ollama |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` | 45, 2 | Network behaviour (retries only for 429/5xx/timeouts) |
| `TOP_K` | 4 | Passages given to the LLM |
| `DENSE_WEIGHT_EN/TA` | 0.5 / 0.2 | Hybrid mixing weights |
| `MIN_RELEVANCE_EN/TA` | 0.25 / 0.18 | Relevance thresholds |
| `RELATIVE_CUTOFF` | 0.85 | Passages from other documents must score at least this fraction of the best hit |
| `CHUNK_SIZE`, `CHUNK_SIZE_TA`, `CHUNK_OVERLAP` | 400, 300, 70 | Chunking (rebuild after changing) |
| `MAX_QUESTION_CHARS` | 1000 | Input limit |
| `AUTO_FETCH_SOURCES` | 0 | Download the manifest sources at start-up when `data/raw/` is empty (set to 1 on Streamlit Cloud) |

Values are read from real environment variables first, then `.env`, then Streamlit
secrets.

## 13. Knowledge-base ingestion

```bash
python -m knowledge_base.fetch_sources --allow-insecure-tls   # optional: re-download corpus
python -m knowledge_base.ingest                                # build data/processed/
```

Example output: `Loaded 42 documents -> 580 chunks.` Bad files are skipped with a
reason. Chunks that are mostly numbers (flattened tables whose rows and columns can no
longer be read, 7 in the current corpus) are left out. To add your own verified PDFs offline, see **`data/README.md`**. If the index is
missing, corrupt or built with a different embedding model, the app rebuilds it
automatically at start-up.

## 14. Model configuration

**Hosted (recommended for deployment).** Any provider with an OpenAI-compatible
`/chat/completions` endpoint works. For example:

```ini
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai   # Google Gemini
LLM_MODEL=<a current Gemini model name>
LLM_API_KEY=<your key>
```

or `https://api.groq.com/openai/v1` (Groq) or `https://openrouter.ai/api/v1`. Check
the provider's current model list: model names are retired over time. **Free tiers
have rate limits and daily quotas that providers change at any time. None is free
indefinitely.** When a quota is hit, the app shows a friendly message and still
displays the retrieved sources. For Tamil answers, choose a model with good Tamil
ability and check its output on the evaluation questions (§16).

Measured on 24 Oct 2026 with free keys: Gemini `gemini-2.5-flash` allowed only **20
requests per day**, too few for a public app. Groq `openai/gpt-oss-120b` allowed 1,000
requests per day but only 8,000 tokens per minute (about 2–3 answers per minute), with
answers in 1–12 s. On HTTP 429 the client waits for the provider's `Retry-After` (up to
20 s) before retrying, and gives up at once when the wait is longer (a daily quota).

**Local (Ollama), free, offline and no API key.** This is the setup used on the
development laptop (Linux, CPU only). Ollama was installed in the home folder without
admin rights:

```bash
mkdir -p ~/.local/ollama && cd ~/.local/ollama
curl -fL -o ollama.tar.zst https://ollama.com/download/ollama-linux-amd64.tar.zst
tar --zstd -xf ollama.tar.zst && rm ollama.tar.zst          # ~2.4 GB installed
OLLAMA_HOST=127.0.0.1:11434 ~/.local/ollama/bin/ollama serve  # keep this terminal open
~/.local/ollama/bin/ollama pull gemma3:4b                    # ~3.3 GB, in a 2nd terminal
```

(Windows/macOS: install from https://ollama.com, then run `ollama pull gemma3:4b`.)
Then set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=gemma3:4b` in `.env` and run
`streamlit run app.py`. `gemma3:4b` needs about 4–5 GB of free RAM. On a CPU, answers take
tens of seconds. Bind Ollama to `127.0.0.1` (as above) so it is not exposed to the
network. Ollama cannot run on Streamlit Community Cloud; use a hosted provider there.

**Offline demo.** `LLM_PROVIDER=offline` makes no external calls. When a provider is
configured, a sidebar toggle switches the session to offline mode.

## 15. Testing

```bash
pytest                         # unit + UI tests (uses the real model/index if present)
pytest -m "not model"          # fast tests only, no model download needed
AGRIBOT_RUN_INTEGRATION=1 LLM_PROVIDER=openai_compatible LLM_API_KEY=... LLM_MODEL=... \
  pytest tests/test_integration_provider.py -s     # real provider (uses quota)
```

LLM calls are mocked in unit tests. `tests/test_llm_client.py` also runs a local HTTP
server that speaks the OpenAI wire format, so the real `requests` code path is tested.
Results are in **`docs/TEST_REPORT.md`**.

## 16. Evaluation (measured)

```bash
python -m evaluation.run_eval                                            # tuning set
python -m evaluation.run_eval --questions evaluation/heldout_questions.json
python -m evaluation.run_eval --generate      # with a provider: saves answers for manual review
```

Retrieval results with the configured parameters (full tables in `evaluation/results/`):

| Set | Lang | Correct doc in top 4 | Answerable above threshold | Off-topic rejected |
|---|---|---|---|---|
| Tuning (36 q) | English | 14/14 | 14/14 | 7/7 |
| Tuning | Tamil | 12/12 | 11/12 | 3/3 |
| Held-out (28 q) | English | 10/10 | 10/10 | 4/4 |
| Held-out | Tamil | 9/10 | 8/10 | 4/4 |

Retrieval latency is about 25 ms per question on CPU once the model is loaded.

Caveats: these are small, hand-labelled sets. Thresholds were chosen on the tuning set.
One held-out question (h-ta06) was inspected while diagnosing a Tamil retrieval
problem, so the held-out set is not fully blind.

**Answer quality with a real model** (Ollama `gemma3:4b`, CPU). Each answer was checked
by hand against its passages. Counts are yes / partial / no, where *yes* means
every claim is in the cited passages:

| Lang | Answered | Round 1 (original) | Round 2 (after fixes) |
|---|---|---|---|
| English | 24 | 14 / 8 / 2 | **18 / 5 / 1** |
| Tamil | 20 | 8 / 10 / 2 | **14 / 6 / 0** |

All 18 off-topic questions were refused. 2 in-scope Tamil questions were wrongly
refused. Median answer time was about 26 s on CPU. The round-2 fixes were designed
after reading the round-1 answers, including the held-out ones, so round 2 is not a
blind test. The review was done by the AI assistant that built the project, not by a
native Tamil speaker or an agronomist. Details and remaining failure cases:
`docs/TEST_REPORT.md`.

## 17. Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub. `.env` and `secrets.toml` are git-ignored; check
   with `git status` that no key is committed. Read the licensing note in §9 before
   making the repository public.
2. At https://share.streamlit.io, choose **Create app**, select the repository, set the
   main file to `app.py` and, under *Advanced settings*, pick **Python 3.12**.
3. In **Secrets**, paste the contents of `.streamlit/secrets.toml.example` with your real
   values. Keep `AUTO_FETCH_SOURCES = "1"`: the repository has no source documents, so
   the app downloads them from the manifest on first boot.
4. Deploy. The first boot installs CPU PyTorch and downloads the embedding model, so
   expect several minutes, plus 1–2 minutes to download the corpus and build the index
   on the small cloud CPUs. The container filesystem is ephemeral, so this repeats
   after the app is restarted.

Notes: Community Cloud resources (RAM, CPU, sleep after inactivity) are limited and can
change. The embedding model plus PyTorch need roughly 1 GB of RAM. The container
filesystem is ephemeral, so nothing a user does is stored there, and chat history
lives only in the browser session. Use a hosted LLM provider in the cloud, because
Ollama cannot run there. This deployment has **not** been tested on Streamlit Cloud
from this workspace.

## 18. Limitations

- The knowledge base is small (42 documents, mostly rice, irrigation, nutrients and
  IPM). Many crops, schemes, market prices and weather are not covered.
- Tamil retrieval is weaker than English. The embedding model represents Tamil poorly,
  and the lexical component only matches words that appear in the Tamil pages. In
  evaluation, some Tamil questions fell just below the threshold (so the bot says
  "no verified information"). In one case an unrelated Tamil passage passed the
  threshold, which leaves the LLM's instructions as the only safeguard (see
  `docs/TEST_REPORT.md`).
- Tamil answers from a small local model (`gemma3:4b`) are mostly readable but sometimes
  misread flattened tables, mangle scientific names, or copy broken fragments (see
  `docs/TEST_REPORT.md`). A larger hosted model will likely do better, but that was not
  measured.
- The model can still state a fact from its own knowledge and attach a citation to it
  (h-en04). It can also present an unrelated passage as an answer when that passage is
  the only one above the threshold (h-ta01).
- Code-level checks reduce risk but cannot prove that every sentence is supported. A
  model can still paraphrase a source wrongly.
- No image-based diagnosis, voice input or persistent history.
- On small phones (about 375×667), Streamlit's chat auto-scroll hides the title on
  the welcome screen; the user can scroll up to see it.
- Source pages have no publication dates, so their age is unknown.
- The TNAU server intermittently presents an expired TLS certificate. Affected
  downloads are flagged `tls_verified: false`.

## 19. Future enhancements

- Expand the corpus with permission: TNAU crop production guides (PDF), ICAR
  publications, Tamil Nadu agriculture department advisories.
- A stronger multilingual embedding model (for example multilingual-e5 or BGE-M3)
  and a cross-encoder re-ranker.
- Voice input/output in Tamil (speech-to-text and text-to-speech).
- Image-based pest/disease suggestions with explicit uncertainty.
- District-level weather advisories via official IMD/agromet bulletins.
- An admin interface with authentication for uploading new verified documents.
- Answer feedback buttons and a reviewed evaluation dashboard.

## 20. Viva questions and answers

1. **What is RAG and why use it here?** Retrieval-Augmented Generation retrieves
   relevant documents first and gives them to the LLM as context. It keeps answers
   tied to verified agricultural sources and lets us show citations, instead of
   relying on what the model memorised.
2. **Why `paraphrase-multilingual-MiniLM-L12-v2`?** It is small (≈470 MB, 384
   dimensions), runs on CPU, and maps 50+ languages including Tamil and English into
   one vector space.
3. **What problem did you find with Tamil embeddings and how did you fix it?** Tamil
   vectors were clustered together, so unrelated Tamil sentences scored about 0.8. We
   subtract each language's mean vector (centring) and add character n-gram lexical
   scoring. Tamil top-4 accuracy on our tuning set went from 5/12 to 12/12.
4. **Why FAISS `IndexFlatIP`?** It does exact search. On normalised vectors, inner
   product equals cosine similarity, and for a few thousand chunks exact search is
   fast (about 25 ms per query).
5. **How do you prevent hallucination?** Four layers: (a) a relevance threshold and a
   relative cutoff in code, so the LLM is not called when nothing relevant exists and
   is not shown weaker neighbouring topics; (b) a strict system prompt; (c) citation
   validation that removes labels not retrieved; (d) URL stripping and flags for
   unsupported quantities and acre/hectare unit swaps. They reduce errors, but
   don't eliminate them. In the final real-model evaluation, 1 of 44 answers had a
   core claim that was not in its passages, and 11 more had a smaller unsupported or
   misread claim.
6. **How are citations guaranteed to be real?** The model can only write labels like
   `[S1]`. The app maps labels to stored metadata (title, URL, page). Unknown labels
   and model-written URLs are removed.
7. **What is prompt injection and how do you handle it?** Text inside a document that
   tries to give the model instructions. We strip instruction-like sentences from
   passages, escape delimiter characters and tell the model passages are untrusted
   evidence.
8. **Why chunk with overlap?** So that a fact split across a chunk boundary still
   appears whole in at least one chunk.
9. **Why are Tamil chunks smaller?** Tamil uses more tokens per character. The model
   reads only 128 tokens, so 31% of the original 450-character Tamil chunks were
   being cut off. With 300-character Tamil chunks (plus the title prefix), 4% of
   Tamil and 5% of English chunks exceed 128 tokens. Only their final words are
   ignored, and the overlap repeats those words in the next chunk.
10. **How did you choose the relevance threshold?** We swept thresholds on a labelled
    question set and picked values that kept in-scope questions answerable while
    rejecting off-topic ones, then checked them on a separate held-out set.
11. **What does offline mode do?** It skips the LLM and shows the top retrieved
    passages verbatim, labelled as not AI-generated. It is useful for demos without
    an API key.
12. **How are API keys protected?** They are read from env/secrets, sent only in the
    Authorization header, masked in the project page, never logged, and `.env` is
    git-ignored. A test asserts the key never appears in logs, `repr` or errors.
13. **How does the app handle API failures?** Errors are mapped to types (quota,
    timeout, auth, outage). It retries with exponential backoff only for 429, 5xx and
    timeouts, shows a friendly message and still shows the sources.
14. **Why not let users upload PDFs in the app?** A public upload would let anyone
    change the "verified" knowledge base. Documents are added by the maintainer
    through the ingestion CLI, which validates type, size and readability.
15. **What are the main limitations?** A small corpus, weaker Tamil retrieval,
    unreviewed generation quality, and no image diagnosis (see §18).

---

*AgriBot AI is an educational project. It does not replace advice from qualified
agricultural officers. Always follow product labels and local agricultural department
guidance.*
