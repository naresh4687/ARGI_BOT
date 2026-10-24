# Retrieval evaluation – `heldout_questions.json`

Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (language-centred) + char-4-gram TF-IDF; chunks: 580; top_k = 4; dense weight en=0.5, ta=0.2.
Questions: 20 in-scope (10 English, 10 Tamil), 8 out-of-scope.

## Ranking quality (threshold disabled)

| Subset | n | Hit@1 | Hit@k | MRR@k |
|---|---|---|---|---|
| All in-scope | 20 | 0.85 | 0.95 | 0.89 |
| English | 10 | 1.00 | 1.00 | 1.00 |
| Tamil | 10 | 0.70 | 0.90 | 0.78 |

## Relevance-threshold sweep (per question language)

*Answerable* = in-scope question whose correct document is retrieved above the threshold. *Rejected* = out-of-scope question with no passage above the threshold (so the bot says verified information is unavailable).

| Lang | Threshold | In-scope answerable | Out-of-scope rejected |
|---|---|---|---|
| en | 0.10 | 10/10 | 0/4 |
| en | 0.14 | 10/10 | 0/4 |
| en | 0.18 | 10/10 | 3/4 |
| en | 0.22 | 10/10 | 4/4 |
| en | 0.25 **(configured)** | 10/10 | 4/4 |
| en | 0.30 | 10/10 | 4/4 |
| en | 0.35 | 8/10 | 4/4 |
| ta | 0.10 | 9/10 | 2/4 |
| ta | 0.14 | 9/10 | 4/4 |
| ta | 0.18 **(configured)** | 8/10 | 4/4 |
| ta | 0.22 | 5/10 | 4/4 |
| ta | 0.25 | 4/10 | 4/4 |
| ta | 0.30 | 1/10 | 4/4 |
| ta | 0.35 | 0/10 | 4/4 |

## With the configured settings (threshold + relative cutoff 0.85)

| Lang | Correct document reaches the LLM | Mean passages sent (in-scope) |
|---|---|---|
| en | 10/10 | 2.2 |
| ta | 7/10 | 1.2 |

Retrieval latency per question (CPU, model already loaded): median 14 ms, max 15 ms.

## Per-question results

| ID | Lang | Rank of correct doc | Top-1 document | Top-1 score | Correct-doc score | Passes threshold |
|---|---|---|---|---|---|---|
| h-en01 | en | 1 | tnau-irrigation-sprinkler-en | 0.543 | 0.543 | yes |
| h-en02 | en | 1 | tnau-rice-pest-gall-midge-en | 0.314 | 0.314 | yes |
| h-en03 | en | 1 | tnau-rice-pest-leaf-folder-en | 0.406 | 0.406 | yes |
| h-en04 | en | 1 | tnau-rice-disease-brown-spot-en | 0.537 | 0.537 | yes |
| h-en05 | en | 1 | tnau-nutrient-application-methods-en | 0.522 | 0.522 | yes |
| h-en06 | en | 1 | tnau-rice-sri-nursery-en | 0.424 | 0.424 | yes |
| h-en07 | en | 1 | tnau-crop-selection-en | 0.359 | 0.359 | yes |
| h-en08 | en | 1 | hdra-natural-pest-disease-control-en | 0.574 | 0.574 | yes |
| h-en09 | en | 1 | tnau-mineral-essential-elements-en | 0.421 | 0.421 | yes |
| h-en10 | en | 1 | tnau-rice-sri-weed-en | 0.319 | 0.319 | yes |
| h-ta01 | ta | not in top-k | tnau-rice-disease-blb-ta | 0.206 | – | yes |
| h-ta02 | ta | 1 | tnau-rice-disease-blast-ta | 0.295 | 0.295 | yes |
| h-ta03 | ta | 1 | tnau-rice-pest-stem-borer-ta | 0.328 | 0.328 | yes |
| h-ta04 | ta | 1 | tnau-ipm-components-ta | 0.267 | 0.267 | yes |
| h-ta05 | ta | 4 | tnau-nutrient-fertilizer-tips-ta | 0.165 | 0.145 | no |
| h-ta06 | ta | 1 | tnau-irrigation-drip-ta | 0.192 | 0.192 | yes |
| h-ta07 | ta | 1 | tnau-mineral-essential-elements-ta | 0.239 | 0.239 | yes |
| h-ta08 | ta | 1 | tnau-rice-disease-blb-ta | 0.284 | 0.284 | yes |
| h-ta09 | ta | 2 | tnau-rice-sri-weed-ta | 0.240 | 0.185 | yes |
| h-ta10 | ta | 1 | tnau-irrigation-water-quality-ta | 0.203 | 0.203 | yes |
| h-oos01 | en | (out of scope) | tnau-irrigation-sprinkler-en | 0.175 | – | no |
| h-oos02 | en | (out of scope) | tnau-ipm-introduction-ta | 0.144 | – | no |
| h-oos03 | en | (out of scope) | hdra-natural-pest-disease-control-en | 0.153 | – | no |
| h-oos04 | en | (out of scope) | tnau-irrigation-drip-en | 0.202 | – | no |
| h-oos05 | ta | (out of scope) | tnau-rice-disease-blast-ta | 0.122 | – | no |
| h-oos06 | ta | (out of scope) | tnau-mineral-essential-elements-ta | 0.111 | – | no |
| h-oos07 | ta | (out of scope) | tnau-ipm-components-ta | 0.095 | – | no |
| h-oos08 | ta | (out of scope) | tnau-rice-pest-stem-borer-ta | 0.084 | – | no |
