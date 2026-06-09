# CLaMP: Contrastive Language-Materials Pretraining

This repository contains the source code for the CLaMP materials-discovery workflow described in the accompanying manuscript. CLaMP aligns crystal-graph representations with natural-language material descriptions and predicts band gap, formation energy, and energy above hull.

## Contents

Core scripts:

- `build_graphs.py`: convert CIF files to PyTorch Geometric graph objects.
- `split_dataset.py`: create train/validation/test metadata files.
- `extract_text_embeddings.py`: encode material descriptions with a sentence-embedding model.
- `train.py`: train graph-only and multimodal models.
- `evaluate.py`: evaluate property prediction, stability classification, and retrieval.
- `text_to_graph_retrieval.py`: run paired and natural-language text-to-graph retrieval.
- `prepare_2d_candidates.py` and `run_2d_retrieval.py`: prepare and rank 2D candidate materials.
- `llm_baseline_vllm.py`: LLM few-shot baseline for property prediction.

Small example inputs are provided under `examples/`. Large generated data and model artifacts are not included.

## Excluded Artifacts

The repository intentionally excludes:

- model checkpoints and tensor files (`*.pt`, `*.pth`, `*.ckpt`, `*.npz`, `*.safetensors`),
- local language-model directories (`Qwen/`, `llm_models/`),
- raw JARVIS tables, CIF collections, graph caches, and text embeddings,
- generated train/validation/test splits,
- figures, plotting scripts, logs, backups, and archives.

## Expected Data Layout

Default paths are configured in `config.py`. A full run expects the following local artifacts:

```text
.
|-- jarvis_properties.csv
|-- jarvis_clean_texts.jsonl
|-- cif_files_jarvis/
|-- pyg_graphs/
|-- text_embeddings_clean/
|-- dataset_split/
|   |-- train/train_metadata.jsonl
|   |-- val/val_metadata.jsonl
|   |-- test/test_metadata.jsonl
|-- checkpoints_multimodal/
|   |-- best_model.pt
|   |-- norm_stats.json
|-- results/
```

## Installation

Install PyTorch and PyTorch Geometric for your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Typical Workflow

```bash
python build_graphs.py
python extract_text_embeddings.py --model Qwen/Qwen3-Embedding-8B --output text_embeddings_clean
python split_dataset.py
python update_split_metadata.py --new_embed_dir text_embeddings_clean --filter_missing
python train.py --model_type multimodal --ckpt_dir checkpoints_multimodal
python evaluate.py --model_type multimodal --ckpt_dir checkpoints_multimodal --tag clean
python text_to_graph_retrieval.py --mode pair --split test --ckpt_dir checkpoints_multimodal --top_ks 1,5,10 --max_samples 5000
python text_to_graph_retrieval.py --mode nl --queries examples/nl_queries.jsonl --split test --ckpt_dir checkpoints_multimodal_v3 --qwen_path Qwen/Qwen3-Embedding-8B --local_files_only --top_ks 1,5,10,50,100 --max_samples 5000
```

## Citation

A citation entry will be added after publication.
