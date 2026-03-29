# late-interaction-mm-graph-rag

**Late-interaction** (multi-vector / **MaxSim-style**) multimodal retrieval with **query-driven multimodal GraphRAG**-style reasoning.  
**ColEmbed-era** behavior and module boundaries follow `mmgraphrag/colembed_mm_graph_rag` (shortlist in vector space, then symbolic graph in `extraction` / `construct`).

## Quick start

```bash
python -m venv .venv
pip install -r requirements.txt
copy config.example.yaml config.yaml   # Windows; use cp on Unix — then edit, do not commit
```

## Configuration

- Never commit API keys or tokens. Use `config.yaml` (git-ignored) or environment variables.
- See `config.example.yaml` for schema placeholders.
- LLM calls use `util/request.py` (`.env` / `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, etc.).

## Pipeline (ported from `mmgraphrag/colembed_mm_graph_rag`)

Run each stage from the **repository root** (`cd late-interaction-mm-graph-rag`).

1. **`pattern.py`** — graph-pattern cache under `result/<RUN_ID>/phase2_pattern_cache/` (set `PATTERN_*` / `MMGRAPHRAG_RUN_ID` env vars as needed).
2. **`extraction.py`** — relation extraction cache → `phase3_extraction_cache/`.
3. **`construct.py`** — builds `.graphml` per question → `result/<RUN_ID>/phase4_graphs_real/`.
4. **`inference.py`** — dry-run by default (`INFERENCE_DRY_RUN=1`): reads graphs, writes predictions + retrieval JSON under `result/<RUN_ID>/`. Set `INFERENCE_DRY_RUN=0` only when the full ColEmbed + API stack is configured.

**Paths:** defaults expect MultiModalQA-style files under `data/multimodalqa/dataset/` (`MMQA_dev.jsonl`, `MMQA_tables.jsonl`, `MMQA_images.jsonl`, `MMQA_texts.jsonl`) and images under `data/multimodalqa/final_dataset_images/`. Adjust names to match your unpack, or set `MMQA_DATASET_DIR` / `MMQA_IMAGES_DIR`.

This repository keeps **core pipeline code only** (Phases 2–5 as in the Colembed run report). For **EM/F1 retrieval metrics**, gold alignment, and experiment write-ups, use the sibling project `mmgraphrag/colembed_mm_graph_rag` (see `.dev_document/md/20260321_00100_colembed_result_report.md`).

## Dataset

This project targets evaluation and development on **MultiModalQA** — complex QA over **text, tables, and images** (ICLR 2021). Download the official files and unpacked assets under your local `data/` directory (git-ignored); do not commit the corpus.

- **Repository & format:** [allenai/multimodalqa](https://github.com/allenai/multimodalqa)  
  Includes `dataset/` (`MultiModalQA_*.jsonl.gz`, `tables.jsonl.gz`, `texts.jsonl.gz`, `images.jsonl.gz`) and instructions to obtain `images.zip`.

Place extracted JSONL and images so they match the paths above (or set `MMQA_DATASET_DIR`). Upstream field definitions: questions, contexts, modalities, supporting context ids — see the [allenai/multimodalqa](https://github.com/allenai/multimodalqa) README.

## Acknowledgements

- **Query-Driven Multimodal GraphRAG (method / layout reference):**  
  [https://github.com/DMiC-Lab-HFUT/Query-Driven-Multimodal-GraphRAG](https://github.com/DMiC-Lab-HFUT/Query-Driven-Multimodal-GraphRAG)

- **Nemotron ColEmbed V2 (model / embedding reference):**  
  [https://huggingface.co/blog/nvidia/nemotron-colembed-v2](https://huggingface.co/blog/nvidia/nemotron-colembed-v2)

- **MultiModalQA (dataset):**  
  [https://github.com/allenai/multimodalqa](https://github.com/allenai/multimodalqa)

## License

[MIT License](LICENSE).
