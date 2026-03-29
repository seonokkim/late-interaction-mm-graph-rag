# late-interaction-mm-graph-rag

**Multimodal GraphRAG with Late-Interaction Evidence Retrieval** builds a **multimodal graph**—nodes and edges over text, tables, and images—grounded in **late-interaction** (**multi-vector / MaxSim-style**) relevance scoring for evidence selection. Pattern-guided extraction populates the graph; **construct** materializes it (e.g. GraphML), and **inference** produces answers by reasoning over that structure. Experiments use **MultiModalQA**; the end-to-end procedure is `pattern` → `extraction` → `construct` → `inference`.

## Dependencies

Install packages listed in `requirements.txt`. A local copy of `config.example.yaml` can be used as a template for experiment settings.

## Pipeline

Execute the stages from the project root, in order:

1. **`pattern.py`** — graph-pattern cache under `result/<RUN_ID>/phase2_pattern_cache/`
2. **`extraction.py`** — relation extraction cache → `phase3_extraction_cache/`
3. **`construct.py`** — per-question graphs (GraphML) → `result/<RUN_ID>/phase4_graphs_real/`
4. **`inference.py`** — predictions from constructed graphs; module flags control whether scoring uses only cached graphs or full multimodal decoding.

Stage-specific options and data paths are set in each script (and in `paths.py`).

**Data:** place MultiModalQA-style JSONL and image assets under `data/multimodalqa/dataset/` and `data/multimodalqa/final_dataset_images/` (filenames `MMQA_*.jsonl` as expected by the loaders), or adjust paths in `paths.py` / environment variables.

## Dataset

**MultiModalQA** (ICLR 2021): joint reasoning over text, tables, and images.

- [allenai/multimodalqa](https://github.com/allenai/multimodalqa)

## Acknowledgements

- [Query-Driven Multimodal GraphRAG](https://github.com/DMiC-Lab-HFUT/Query-Driven-Multimodal-GraphRAG)
- [Nemotron ColEmbed V2](https://huggingface.co/blog/nvidia/nemotron-colembed-v2)
- [MultiModalQA](https://github.com/allenai/multimodalqa)

## License

[MIT License](LICENSE).
