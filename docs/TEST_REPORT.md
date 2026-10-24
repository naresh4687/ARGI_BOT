# Test Report — AgriBot AI

**Run date:** 2026-10-24 · **Python** 3.12.3 · **OS** Linux 6.8 (x86-64, CPU only)
**Command:** `pytest` (from the project root, with the real embedding model cached
and the index built)

## Summary

| Result | Count |
|---|---|
| Passed | **120** |
| Failed | 0 |
| Skipped | 1 (real-provider integration module, opt-in; run separately, 3/3 passed with Ollama `gemma3:4b`) |

Fast subset without the model (`pytest -m "not model"`): 104 passed, 1 skipped,
16 deselected.

Clean-environment check: a new virtual environment installed only from
`requirements-dev.txt` installed without errors, `pip check` reported no broken
requirements, and the suite passed (108 tests at that point, before the two UI tests
below were added). The Hugging Face model cache already present on the machine was
reused, so the first-time model download was not re-tested.

## Tests by area

| File | Tests | What is verified |
|---|---|---|
| `test_preprocess.py` | 10 | HTML→text removes scripts/nav/footers; Tamil NFC normalisation and zero-width removal; de-hyphenation; chunk size limit; overlap present; long-sentence hard split; Tamil chunking; empty input and invalid parameters; **table-debris detection** (number-only fragments in EN/TA flagged, prose and dosage tables kept) |
| `test_ingest.py` | 13 | **PDF extraction** with page numbers; running-header/footer removal; **corrupt PDF**, text-less (scanned) PDF, empty and unsupported files fail gracefully; **sidecar metadata preserved into every chunk**; missing metadata is not invented (URL/date stay `None`); Tamil detection; JSON passages; invalid JSON; bad files skipped and reported; missing directory; smaller Tamil chunks; table-debris chunks left out of the index |
| `test_vector_store.py` | 9 | Search ordering and metadata; scores consistent; save/load round trip including language centroids; **missing index**; corrupt index; inconsistent chunk file; embedding-model mismatch; dimension mismatch; misaligned build input |
| `test_language.py` | 11 | Detection of English, Tamil, mixed Tamil+English, Hindi/Telugu ("other"), and no-letter input; UI strings complete in both languages; Tamil strings in Tamil script; examples in the right language; string formatting and fallback |
| `test_retriever.py` | 18 | Relevant passage with metadata and S-labels; Tamil query → Tamil passage; **threshold rejects irrelevant queries**; threshold enforced in code; max passages per document; **relative cutoff drops weak passages from other documents**; page number carried through; empty query; Tamil n-gram stem sharing; lexical index. **Real model and index:** 3 English + 3 Tamil questions retrieve the expected TNAU documents with real TNAU URLs; 2 off-topic questions return nothing |
| `test_safety.py` | 12 | **Empty/whitespace**, **oversized**, too-short input; control-character stripping; unsupported language; Tamil accepted; **prompt-injection** sentences and delimiter-closing tags removed from evidence; prompt wraps evidence and escapes titles; language instruction; **invalid source identifiers removed**; model-invented URLs removed while real source URLs kept; high-risk topic detection |
| `test_llm_client.py` | 17 | **Missing credentials**; offline mode creates no provider; request format; **429 retried then quota error**; transient 503 recovers; **timeout**; connection error; 401 not retried; malformed responses (3 cases); **API key never in logs, repr, errors or redacted settings**; Ollama unreachable message; Ollama success; real HTTP round trip against a local fake OpenAI-compatible server (with Tamil text); wrong key against that server |
| `test_responses.py` | 20 | **English answering** with valid citations, and sources come only from retrieved metadata; **Tamil answering**; non-Tamil reply flagged; **missing knowledge → no LLM call** (EN and TA messages); invalid citations removed with a warning; uncited answer marked unverified; invented URL removed; unsupported quantities flagged; quantities present in sources not flagged; quota and timeout errors handled (EN/TA), with sources still shown; offline mode and its language note; language-mismatch notice; injection neutralised before the prompt; prompt carries the on-topic rule and the Tamil KVK name; **acre/hectare unit swap flagged** (Tamil), unit used by the source not flagged; empty/oversized input rejected before retrieval; unknown language code |
| `test_app.py` | 8 | Streamlit `AppTest` with the real model and index: app starts with welcome and examples; example question answered with TNAU source links; off-topic → uncertainty response; Tamil selection and clear chat; **duplicate submission processed once**; Help/About/Project pages render; **missing credentials → warning + offline fallback**; **provider outage → friendly error, sources still shown, key not displayed** |
| `test_bootstrap.py` | 2 | Empty corpus raises without `AUTO_FETCH_SOURCES`; with it, sources are downloaded (stubbed) and the index is built |
| `test_integration_provider.py` | (3, opt-in) | Real provider: grounded EN and TA answers; off-topic rejection. **3 passed** in 74 s with Ollama `gemma3:4b` on CPU (`AGRIBOT_RUN_INTEGRATION=1`) |

## Skipped tests and external integrations not tested

- `tests/test_integration_provider.py` is skipped in the default run because it needs a
  live model. It was run separately against local Ollama `gemma3:4b` and passed (3/3):
  `AGRIBOT_RUN_INTEGRATION=1 pytest tests/test_integration_provider.py`. Hosted
  OpenAI-compatible providers (Groq, Gemini) were **not** tested with a real key.
- Streamlit Community Cloud deployment was **not** tested (no hosting access from this
  workspace). The app was verified to start with `streamlit run app.py` locally
  (health endpoint returned `ok`, no errors in the log).
- Real-browser check (headless Chromium 131 via Playwright, offline mode): laptop
  (1366×900) and phone (390×844) layouts were driven and screenshotted
  (`docs/screenshots/`). Checked: welcome screen; an English example question with an
  expanded sources panel; switching to Tamil and asking a typed Tamil question; an
  off-topic question on the phone. No JavaScript errors and no horizontal scrolling.
  Two phone bugs were found and fixed: example buttons truncated with "…", and the
  title hidden under the top bar. A shorter Tamil input placeholder was also adopted.
  Remaining: on small phones (375×667), Streamlit's chat auto-scroll still pushes the
  title above the fold on the welcome screen; the examples and input stay visible.
  These browser checks were manual (screenshots reviewed by eye) and are not part of
  the pytest suite.

## Retrieval evaluation (separate from pytest)

`python -m evaluation.run_eval` and
`python -m evaluation.run_eval --questions evaluation/heldout_questions.json`. The
full tables are in `evaluation/results/`.

| Set | Lang | Hit@4 | In-scope answerable | Off-topic rejected |
|---|---|---|---|---|
| Tuning (36) | en | 14/14 | 14/14 | 7/7 |
| Tuning | ta | 12/12 | 11/12 | 3/3 |
| Held-out (28) | en | 10/10 | 10/10 | 4/4 |
| Held-out | ta | 9/10 | 8/10 | 4/4 |

With the relative cutoff (0.85) also applied, as in the app, the correct document still
reaches the LLM for 14/14 + 11/12 tuning and 10/10 + 7/10 held-out questions (EN + TA).
The cutoff removes the SRI-nursery passage for h-ta09, which did not contain the asked
fact (seedling age). Tamil questions get 1.2–1.7 passages on average, English 2.2–2.3.
Index: 580 chunks, after removing 7 table-debris chunks.

Observed failure cases (all Tamil):

- **h-ta01** "நெல் வயலில் எப்போது நீர் பாய்ச்ச வேண்டும்?" (when to irrigate a paddy
  field): no expected document is in the top 4, and an *unrelated* passage (bacterial
  leaf blight, score 0.206) passes the threshold. The LLM would receive irrelevant
  evidence and must rely on its instruction to say the information is unavailable.
  This is the most serious observed failure.
- **h-ta05** (how organic manures help soil): the correct document is ranked 4th, but
  scores 0.146, below the 0.18 threshold. The bot reports no verified information.
- **ta09** (tuning set; is the irrigation water suitable): ranked 1st, but scores 0.152,
  below the threshold. The bot reports no verified information.

Thresholds were tuned on the tuning set, and held-out question h-ta06 was inspected
while diagnosing a retrieval problem, so the held-out result is not fully blind.

## Answer evaluation with a real model (separate from pytest)

`python -m evaluation.run_eval --generate` (and `--questions evaluation/heldout_questions.json`)
with `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=gemma3:4b`, temperature 0.2, CPU only
(no GPU). Every answer was checked by hand against the passages it was given. The
verdicts and notes are in the `manual_review` field of
`evaluation/results/generation_*_ollama*.json`.

**Who reviewed:** the AI coding assistant that built the project (Claude), reading each
answer next to its retrieved passages. It is a single reviewer, not a native Tamil
speaker and not an agronomist, so the Tamil-quality and agronomy judgements should be
spot-checked by a person before they are quoted as final.

**Verdicts.** *Yes* = every factual claim is in the cited passages. *Partial* = the
main answer is supported, but at least one claim is unsupported, misread or
mis-cited. *No* = the core of the answer is wrong or not from the passages.

Round 1 used the original pipeline. Round 2 came after three fixes chosen from the
round-1 review: a relative score cutoff in the retriever, removal of table-debris
chunks at ingestion, and stricter prompt rules (stay on the asked topic; say when the
specific fact is missing; don't explain list items from memory; correct Tamil name
for KVK).

| Set | Lang | Answered | Round 1 yes / partial / no | Round 2 yes / partial / no |
|---|---|---|---|---|
| Tuning | English | 14 | 7 / 6 / 1 | **11 / 3 / 0** |
| Tuning | Tamil | 11 | 5 / 5 / 1 | **8 / 3 / 0** |
| Held-out | English | 10 | 7 / 2 / 1 | 7 / 2 / 1 |
| Held-out | Tamil | 9 | 3 / 5 / 1 | **6 / 3 / 0** |
| **All** | English | 24 | 14 / 8 / 2 | **18 / 5 / 1** |
| **All** | Tamil | 20 | 8 / 10 / 2 | **14 / 6 / 0** |

- **Off-topic questions:** 18/18 refused in both rounds, without calling the model.
- **False refusals:** 2 in-scope Tamil questions (ta09, h-ta05) in both rounds, because
  retrieval scores fall below the Tamil threshold (see above).
- **Tamil language quality** (good / fair / poor): round 1 4 / 15 / 1, round 2
  **13 / 7 / 0**. The KVK name went from mistranslated in 11 answers to correct in all.
- **Latency per AI answer** (CPU): median 29.1 s → **25.9 s** (tuning) and 27.7 s →
  **25.4 s** (held-out); maximum 55.7 s → 36.0 s. Fewer passages are sent in round 2.

**Not a blind result.** The fixes were designed after reading both tuning and held-out
round-1 answers, so round-2 held-out numbers are not an independent test. They show
that the fixes work on those cases, not how well the system generalises.

**Remaining failures in round 2:**

- **h-en04** (cause of brown spot): the passages describe symptoms but not the cause.
  The model still names *Helminthosporium oryzae* from its own knowledge and cites it
  as [S1, S2]. The name is correct, but the citation is false.
- **ta05**: copied a biocontrol dose as "per acre" (ஏக்கருக்கு) where the source says
  "per hectare" (எக்டருக்கு). This new error appeared in round 2. The number matched
  the source, so the quantity check missed it. An **acre/hectare unit-swap check** was
  added afterwards. Replayed over all 44 saved round-2 answers, it flags ta05 and
  nothing else. It was not part of the round-2 run.
- **h-ta01** (when to irrigate paddy): only a bacterial-leaf-blight passage passes
  the threshold. Round 1 said the information was unavailable. Round 2 presents the
  disease advice ("drain the field except at flowering") as irrigation guidance.
  This is a regression and the most misleading remaining answer.
- **h-ta08**: wrong pathogen name (*Stenotrophomonas* instead of *Xanthomonas*), not
  cited.
- **h-en09 / ta12**: list only the elements in the retrieved chunk and omit that 16
  are essential.
- Misreadings of flattened tables: SRI seed-rate table (ta03), deficiency tables
  (h-ta07), boron class B1 shown as "1-3 ppm" (en12; the source's "<" was lost).
- **en08**: "water with rose can 2 days before removing mats" (the source says to
  drain).
- **h-en06**: the quantity warning is a false positive ("5 days" vs "5 DAS").

## Known limitations

See README §18.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m knowledge_base.ingest      # only needed if data/processed/ is missing
pytest
```
