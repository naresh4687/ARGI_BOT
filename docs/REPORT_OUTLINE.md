# Mini-Project Report Outline — AgriBot AI

Use this outline for the formal report. Suggested page counts assume a total of
40–60 pages. Wherever a number is needed, copy it from the measured files
(`evaluation/results/*.md`, `docs/TEST_REPORT.md`). Do not estimate.

## Front matter
- Title page (project title, student names and register numbers, guide, department, college, year)
- Bonafide certificate · Declaration · Acknowledgement
- Abstract (≈250 words; adapt README §1)
- Table of contents · List of figures · List of tables · Abbreviations
  (RAG, LLM, FAISS, IPM, INM, SRI, TNAU, KVK, TF-IDF, MRR)

## Chapter 1 – Introduction (4–6 pages)
1.1 Background: agriculture in Tamil Nadu; information needs of farmers
1.2 Problem statement
1.3 Objectives
1.4 Scope (topics covered / not covered; English and Tamil)
1.5 Organisation of the report

## Chapter 2 – Literature survey (5–8 pages)
2.1 Agricultural information systems and portals (TNAU Agritech, Kisan Call Centre, mKisan)
2.2 Chatbots: rule-based vs. LLM-based
2.3 Retrieval-Augmented Generation (cite Lewis et al., 2020)
2.4 Sentence embeddings and multilingual models (cite Reimers & Gurevych, 2019, 2020)
2.5 Vector search (FAISS: Johnson et al., 2019) and lexical retrieval (TF-IDF, BM25)
2.6 Hallucination and prompt injection in LLM systems
2.7 Research gap: bilingual Tamil support plus source-grounded, safety-checked answers
*Only cite papers you have actually read; record full references.*

## Chapter 3 – System analysis (3–5 pages)
3.1 Existing system and its limitations (README §4 table)
3.2 Proposed system and advantages
3.3 Feasibility: technical (CPU-only), economic (free/open-source; API quotas), operational
3.4 Requirements
  - Functional: bilingual Q&A, citations, uncertainty response, clear chat, examples
  - Non-functional: response time, safety, privacy (no persistent storage), portability
  - Hardware and software requirements (README §10)

## Chapter 4 – System design (6–10 pages)
4.1 Architecture diagram (README §7)
4.2 Data-flow diagram, levels 0 and 1 (ingestion; question answering)
4.3 Module design (README §7 code-map table)
4.4 Data design: chunk metadata schema; index files (`index.faiss`, `chunks.jsonl`,
    `index_info.json`, `language_centroids.npz`)
4.5 Retrieval design: centring, hybrid scoring formula, thresholds
4.6 Prompt design: system rules, evidence delimiting
4.7 Safety design: validation layers (input, evidence, output)
4.8 UI design: wireframes or screenshots of chat (EN/TA), sources panel, info pages
4.9 Sequence diagram for one question

## Chapter 5 – Implementation (8–12 pages)
5.1 Environment and tools
5.2 Corpus collection: manifest, robots.txt check, TLS issue, licensing note
5.3 Preprocessing and chunking (with token-length analysis: 31% → 4% truncated Tamil chunks)
5.4 Embedding and indexing
5.5 Retrieval, including the Tamil hubness problem and its fix (before/after table)
5.6 LLM provider abstraction and error handling
5.7 Response validation
5.8 Streamlit interface
5.9 Short, important code excerpts (not entire files)

## Chapter 6 – Testing (4–6 pages)
6.1 Test strategy: unit, UI (AppTest), integration (real provider, opt-in)
6.2 Test cases table: ID, description, input, expected output, actual output, status
    (from `docs/TEST_REPORT.md`)
6.3 Results summary (passed / failed / skipped, with reasons)
6.4 What is not tested and why

## Chapter 7 – Results and evaluation (4–6 pages)
7.1 Evaluation datasets (tuning 36 q; held-out 28 q; how they were labelled)
7.2 Metrics: Hit@1, Hit@k, MRR, answerable rate, off-topic rejection, latency
7.3 Retrieval results (tables from `evaluation/results/`)
7.4 Threshold trade-off chart (answerable vs. rejected as threshold varies)
7.5 Generation results: from `--generate` output **after manual review** (source
    support, language quality). If not done, state that clearly.
7.6 Failure analysis: examples where retrieval failed or an off-topic question passed
7.7 Screenshots

## Chapter 8 – Conclusion and future work (2–3 pages)
8.1 Summary of achievements against the objectives
8.2 Limitations (README §18)
8.3 Future enhancements (README §19)

## References
IEEE or APA style; include the TNAU Agritech Portal pages used (from the manifest), the
model card for paraphrase-multilingual-MiniLM-L12-v2, FAISS, Streamlit and PyMuPDF
documentation.

## Appendices
A. Installation and user manual (README §10–14)
B. Source manifest (list of documents)
C. Sample questions and answers (English and Tamil)
D. Full evaluation tables
E. Viva questions (README §20)
