# Retrieval evaluation – `eval_questions.json`

Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (language-centred) + char-4-gram TF-IDF; chunks: 580; top_k = 4; dense weight en=0.5, ta=0.2.
Questions: 26 in-scope (14 English, 12 Tamil), 10 out-of-scope.

## Ranking quality (threshold disabled)

| Subset | n | Hit@1 | Hit@k | MRR@k |
|---|---|---|---|---|
| All in-scope | 26 | 1.00 | 1.00 | 1.00 |
| English | 14 | 1.00 | 1.00 | 1.00 |
| Tamil | 12 | 1.00 | 1.00 | 1.00 |

## Relevance-threshold sweep (per question language)

*Answerable* = in-scope question whose correct document is retrieved above the threshold. *Rejected* = out-of-scope question with no passage above the threshold (so the bot says verified information is unavailable).

| Lang | Threshold | In-scope answerable | Out-of-scope rejected |
|---|---|---|---|
| en | 0.10 | 14/14 | 0/7 |
| en | 0.14 | 14/14 | 3/7 |
| en | 0.18 | 14/14 | 6/7 |
| en | 0.22 | 14/14 | 7/7 |
| en | 0.25 **(configured)** | 14/14 | 7/7 |
| en | 0.30 | 14/14 | 7/7 |
| en | 0.35 | 13/14 | 7/7 |
| ta | 0.10 | 12/12 | 0/3 |
| ta | 0.14 | 12/12 | 1/3 |
| ta | 0.18 **(configured)** | 11/12 | 3/3 |
| ta | 0.22 | 9/12 | 3/3 |
| ta | 0.25 | 8/12 | 3/3 |
| ta | 0.30 | 5/12 | 3/3 |
| ta | 0.35 | 3/12 | 3/3 |

## With the configured settings (threshold + relative cutoff 0.85)

| Lang | Correct document reaches the LLM | Mean passages sent (in-scope) |
|---|---|---|
| en | 14/14 | 2.3 |
| ta | 11/12 | 1.7 |

Retrieval latency per question (CPU, model already loaded): median 14 ms, max 17 ms.

## Per-question results

| ID | Lang | Rank of correct doc | Top-1 document | Top-1 score | Correct-doc score | Passes threshold |
|---|---|---|---|---|---|---|
| en01 | en | 1 | tnau-rice-disease-blast-en | 0.532 | 0.532 | yes |
| en02 | en | 1 | tnau-irrigation-drip-en | 0.416 | 0.416 | yes |
| en03 | en | 1 | tnau-nutrient-fertilizer-tips-en | 0.475 | 0.475 | yes |
| en04 | en | 1 | tnau-ipm-components-en | 0.386 | 0.386 | yes |
| en05 | en | 1 | tnau-rice-pest-yellow-stem-borer-en | 0.369 | 0.369 | yes |
| en06 | en | 1 | tnau-irrigation-critical-stages-en | 0.530 | 0.530 | yes |
| en07 | en | 1 | tnau-nutrient-inm-en | 0.587 | 0.587 | yes |
| en08 | en | 1 | tnau-rice-sri-irrigation-en | 0.442 | 0.442 | yes |
| en09 | en | 1 | tnau-rice-disease-blb-en | 0.616 | 0.616 | yes |
| en10 | en | 1 | tnau-nutrient-organic-matter-en | 0.612 | 0.612 | yes |
| en11 | en | 1 | tnau-mineral-deficiency-symptoms-en | 0.328 | 0.328 | yes |
| en12 | en | 1 | tnau-irrigation-water-quality-en | 0.379 | 0.379 | yes |
| en13 | en | 1 | hdra-natural-pest-disease-control-en | 0.467 | 0.467 | yes |
| en14 | en | 1 | tnau-rice-disease-sheath-blight-en | 0.370 | 0.370 | yes |
| ta01 | ta | 1 | tnau-rice-disease-blast-ta | 0.285 | 0.285 | yes |
| ta02 | ta | 1 | tnau-irrigation-drip-ta | 0.186 | 0.186 | yes |
| ta03 | ta | 1 | tnau-rice-sri-nursery-ta | 0.353 | 0.353 | yes |
| ta04 | ta | 1 | tnau-nutrient-inm-ta | 0.194 | 0.194 | yes |
| ta05 | ta | 1 | tnau-rice-pest-stem-borer-ta | 0.411 | 0.411 | yes |
| ta06 | ta | 1 | tnau-rice-disease-blb-ta | 0.272 | 0.272 | yes |
| ta07 | ta | 1 | tnau-ipm-components-ta | 0.305 | 0.305 | yes |
| ta08 | ta | 1 | tnau-nutrient-organic-matter-ta | 0.543 | 0.543 | yes |
| ta09 | ta | 1 | tnau-irrigation-water-quality-ta | 0.157 | 0.157 | no |
| ta10 | ta | 1 | tnau-nutrient-fertilizer-tips-ta | 0.245 | 0.245 | yes |
| ta11 | ta | 1 | tnau-rice-sri-weed-ta | 0.251 | 0.251 | yes |
| ta12 | ta | 1 | tnau-mineral-essential-elements-ta | 0.340 | 0.340 | yes |
| oos01 | en | (out of scope) | hdra-natural-pest-disease-control-en | 0.168 | – | no |
| oos02 | en | (out of scope) | tnau-mineral-deficiency-symptoms-en | 0.161 | – | no |
| oos03 | en | (out of scope) | tnau-irrigation-drip-en | 0.130 | – | no |
| oos04 | en | (out of scope) | tnau-mineral-deficiency-symptoms-en | 0.133 | – | no |
| oos05 | en | (out of scope) | tnau-rice-sri-nursery-en | 0.138 | – | no |
| oos06 | en | (out of scope) | hdra-natural-pest-disease-control-en | 0.199 | – | no |
| oos07 | en | (out of scope) | hdra-natural-pest-disease-control-en | 0.153 | – | no |
| oos08 | ta | (out of scope) | tnau-rice-disease-blb-ta | 0.144 | – | no |
| oos09 | ta | (out of scope) | tnau-rice-sri-nursery-ta | 0.102 | – | no |
| oos10 | ta | (out of scope) | tnau-ipm-components-ta | 0.147 | – | no |
