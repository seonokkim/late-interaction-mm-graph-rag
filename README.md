# late-interaction-mm-graph-rag

Multimodal **Graph RAG**: late-interaction (**multi-vector / MaxSim-style**) retrieval for evidence shortlisting, then symbolic graph construction and inference.

## Quick start

```bash
python -m venv .venv
pip install -r requirements.txt
cp config.example.yaml config.yaml   # then edit; do not commit
```

## Configuration

- Do not commit secrets. Use `config.yaml` (git-ignored) or environment variables.
- See `config.example.yaml` for placeholders.
- LLM HTTP calls: `util/request.py` with `.env` (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, etc.).

## Pipeline

Run from the repository root.

1. **`pattern.py`** — pattern cache → `result/<RUN_ID>/phase2_pattern_cache/`
2. **`extraction.py`** — extraction cache → `phase3_extraction_cache/`
3. **`construct.py`** — `.graphml` per question → `result/<RUN_ID>/phase4_graphs_real/`
4. **`inference.py`** — default `INFERENCE_DRY_RUN=1` (graphs → predictions JSON). Set `INFERENCE_DRY_RUN=0` for full runs (vision model + APIs required).

Env: `PATTERN_*`, `EXTRACTION_*`, `CONSTRUCT_*`, `INFERENCE_*`, `MMGRAPHRAG_RUN_ID`, `MMQA_*`, and other vars read by `inference.py` / `paths.py`.

**Data layout:** `data/multimodalqa/dataset/` (`MMQA_*.jsonl`), images under `data/multimodalqa/final_dataset_images/`, or override with `MMQA_DATASET_DIR` / `MMQA_IMAGES_DIR`.

## Dataset

**MultiModalQA** (ICLR 2021): text, tables, images. Keep files under `data/` (git-ignored).

- [allenai/multimodalqa](https://github.com/allenai/multimodalqa)

## Acknowledgements

- [Query-Driven Multimodal GraphRAG](https://github.com/DMiC-Lab-HFUT/Query-Driven-Multimodal-GraphRAG)
- [MultiModalQA](https://github.com/allenai/multimodalqa)

## License

[MIT License](LICENSE).
