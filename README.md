# PCRL-MRG

This is the official code for the EMNLP 2024 paper:

> **See Detail Say Clear: Towards Brain CT Report Generation via Pathological Clue-driven Representation Learning**

The system automatically generates Chinese brain CT reports by fine-tuning LLaMA 3-8B-Instruct with LoRA on the [CTRG dataset](https://github.com/tangyuhao2016/CTRG).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Requirements](#requirements)
3. [Pre-trained Models](#pre-trained-models)
4. [Dataset Setup](#dataset-setup)
5. [Data Preparation Pipeline](#data-preparation-pipeline)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Configuration Reference](#configuration-reference)
9. [Hardcoded Paths — Action Required](#hardcoded-paths--action-required)
10. [Citation](#citation)

---

## Architecture Overview

Two models are implemented:

### PCRL-LLaMA (Main Contribution)
The full model described in the paper, located in `llama_recipes/models/PCRL_llama/`:
- **Visual encoder**: ResNet-101 producing 2048-dim features from 24 CT slices (14×14 spatial grid)
- **Segmentation head**: Per-image segmentation mask generation
- **Text encoder**: BERT for encoding pathological entity descriptions
- **Contrastive learning**: CLIP-style loss aligning image segments with entity descriptions
- **Language model**: LLaMA 3-8B-Instruct with INT4 quantization + LoRA

### ViT-MLP-LLaMA (Baseline)
A simpler LLaVA-inspired baseline, located in `llama_recipes/models/ViT_MLP_llama/`:
- **Visual features**: Pre-extracted CLIP-ViT-Large-patch14 (1024-dim per slice), loaded from `.npz` files
- **Projector**: Two-layer MLP: `Linear(1024→4096) → Linear(4096→4096)`
- **Language model**: LLaMA 3-8B-Instruct with INT4 quantization + LoRA

Each CT sample contains **24 slices** covering 8 anatomical layers (3 images per layer).

---

## Requirements

No `requirements.txt` is included. Install the following core packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft accelerate bitsandbytes
pip install datasets fire numpy scipy tqdm opencv-python
pip install wandb  # optional, for experiment tracking
```

> **Java is required** for the METEOR metric during evaluation. Install JRE 8+ and ensure `java` is on your PATH.

Python 3.11 is recommended. CUDA-capable GPU is required for training (single GPU with INT4 quantization is the standard setup).

---

## Pre-trained Models

Download the following models before running any scripts:

| Model | Purpose | Where to get |
|-------|---------|--------------|
| LLaMA 3-8B-Instruct | Base language model | [Meta / HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| CLIP-ViT-Large-patch14 | Visual features for ViT-MLP baseline | [HuggingFace](https://huggingface.co/openai/clip-vit-large-patch14) |
| CLIP-base-patch32 | Used in data preparation step 3 | [HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32) |
| SAM ViT-H | Segmentation mask generation (data prep step 1) | [Meta SAM](https://github.com/facebookresearch/segment-anything) — `sam_vit_h_4b8939.pth` |

Place the SAM checkpoint at:
```
data_peparation/segment_anything/sam_vit_h_4b8939.pth
```

---

## Dataset Setup

1. Download the CTRG dataset from https://github.com/tangyuhao2016/CTRG
2. Place the raw images and annotation file (`origin_samples.json`) under a local directory, e.g. `/data/CTRG_dataset/`
3. Pre-processed annotation files are already included in `dataset/` for reference

---

## Data Preparation Pipeline

Run the four steps **in order** to prepare training data from raw CTRG images.

> **Note:** The directory is named `data_peparation` (not `data_preparation`) — this is intentional.

### Step 1 — SAM Mask Generation

```bash
python data_peparation/step1_sam_gen_masks.py
```

- Reads `CTRG_dataset/train_part_1.json`
- Requires the SAM ViT-H checkpoint (see above)
- Outputs segmentation masks as `.pkl.gz` files in `CTRG_dataset/mask/`

### Step 2 — Entity Extraction & Dataset Splits

```bash
python data_peparation/step2_re_extract_entities.py
```

- Reads `CTRG_dataset/origin_samples.json`
- Extracts 26 pathological brain entities (frontal lobe, temporal lobe, lateral ventricle, etc.)
- Filters out non-brain entities (chest, lungs, heart, etc.)
- Outputs `train.json` / `validation.json` / `test.json` split files (70/20/10)

### Step 3 — CLIP-Based Mask Selection

```bash
python data_peparation/step3_clip_select_masks.py
```

- Uses CLIP-base-patch32 to match entity text descriptions to segmentation masks
- Maps Chinese entity names to English for CLIP text prompts
- Selects the best-matching mask per entity per image

### Step 4 — Merge Entity Annotations Per Image

```bash
python data_peparation/step4_merge_entities_per_img.py
```

- Merges all entity masks and text descriptions per CT image
- Produces final `dataset/CTRG_SAM_SEG_dataset/seg_mask_anno_train.json`

### Visual Feature Extraction (required for ViT-MLP-LLaMA baseline)

Features must be extracted **before training** — the model loads `.npz` files at runtime, not raw images.

```bash
# CLIP-ViT-Large-patch14 features (1024-dim) — used by ViT-MLP-LLaMA
python data_peparation/get_visual_feature/clip_vit_1024.py

# ResNet-101 features (2048-dim) — used by PCRL-LLaMA
python data_peparation/get_visual_feature/resnet101_2048.py
```

Output directories:
- `vit_img_features/{last_hidden_state,pooler_output}/` — CLIP features per sample
- `cnn_img_features/{att,fc}/` — ResNet features per sample

---

## Training

### Quick Start (ViT-MLP-LLaMA baseline)

```bash
python scripts/VIT_MLP_llama/finetuning_ViT_MLP_llama.py \
  --batching_strategy padding \
  --batch_size_training 4 \
  --num_epochs 3 \
  --use_peft \
  --seed 3578 \
  --peft_method lora \
  --quantization \
  --val_batch_size 40 \
  --test_batch_size 40 \
  --dataset ctrg_dataset \
  --model_name /path/to/Meta-Llama-3-8B-Instruct \
  --output_dir /path/to/output
```

### Using the Shell Script

Edit `scripts/VIT_MLP_llama/finetuning_CTRG_MRG.sh` to replace the hardcoded paths with your local paths, then run:

```bash
bash scripts/VIT_MLP_llama/finetuning_CTRG_MRG.sh
```

### Key Training Flags

| Flag | Description |
|------|-------------|
| `--use_peft` | Enable LoRA fine-tuning (strongly recommended) |
| `--peft_method lora` | Fine-tuning method (lora is the only tested option) |
| `--quantization` | Enable INT4 quantization to reduce VRAM usage |
| `--batch_size_training` | Per-GPU training batch size |
| `--num_epochs` | Number of training epochs |
| `--model_name` | Path to the LLaMA 3-8B-Instruct directory |
| `--output_dir` | Directory to save checkpoints and metrics |
| `--dataset ctrg_dataset` | **Must be set** (default is `samsum_dataset`) |

Checkpoints are saved as `.pth` files (LoRA adapter + projector weights only), not in HuggingFace format.

---

## Evaluation

### Testing (ViT-MLP-LLaMA)

```bash
python scripts/VIT_MLP_llama/test_ViT_MLP_llama.py \
  --use_peft \
  --quantization \
  --dataset ctrg_dataset \
  --model_name /path/to/Meta-Llama-3-8B-Instruct \
  --peft_model_name peft_model_epoch5 \
  --output_dir /path/to/output
```

Or use the shell script:

```bash
bash scripts/VIT_MLP_llama/test_CTRG_MRG.sh
```

### Metrics

Evaluation uses a bundled `pycocoevalcap` module (`llama_recipes/pycocoevalcap/`) computing:

- **BLEU** (1–4)
- **ROUGE-L**
- **METEOR** — requires Java (uses `meteor-1.5.jar`)
- **CIDEr**

Two metric scripts are available:
- `llama_recipes/utils/metrics_for_chs_mrg.py` — for Chinese reports (default)
- `llama_recipes/utils/metrics_for_eng_mrg.py` — for English reports

---

## Configuration Reference

All training settings are controlled by the `train_config` dataclass in `llama_recipes/configs/training.py`. Every field can be overridden via CLI flags.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | (hardcoded path) | Path to LLaMA 3-8B-Instruct |
| `context_length` | 256 | Max sequence length (reduced from 4096 for CT reports) |
| `batch_size_training` | 1 | Training batch size |
| `val_batch_size` | 1 | Validation batch size |
| `num_epochs` | 3 | Training epochs |
| `lr` | 1e-4 | Learning rate |
| `gamma` | 0.85 | StepLR scheduler gamma (step size = 1 epoch) |
| `peft_method` | `"lora"` | PEFT method |
| `quantization` | False | Enable INT4 quantization |
| `use_peft` | False | Enable PEFT |
| `seed` | 42 | Random seed for reproducibility |
| `dataset` | `"samsum_dataset"` | **Must override** to `"ctrg_dataset"` |
| `use_wandb` | False | Enable Weights & Biases logging |
| `save_metrics` | False | Save training metrics to JSON |

### LoRA Configuration (`llama_recipes/configs/peft.py`)

| Parameter | Value |
|-----------|-------|
| `r` | 8 |
| `lora_alpha` | 32 |
| `target_modules` | `["q_proj", "v_proj"]` |
| `lora_dropout` | 0.05 |
| `task_type` | `"CAUSAL_LM"` |

---

## Hardcoded Paths — Action Required

The codebase contains hardcoded absolute paths from the original development environment (`/home/bjutcv/data/zcx/...`). **You must update these before running any scripts.**

| File | What to change |
|------|---------------|
| `llama_recipes/configs/training.py` | Default `model_name` and `output_dir` |
| `llama_recipes/models/ViT_MLP_llama/modeling_ViT_MLP_llama.py` | Visual feature root directory |
| `llama_recipes/datasets/ctrg_dataset.py` | Dataset JSON paths |
| `data_peparation/get_visual_feature/clip_vit_1024.py` | Dataset and model paths |
| `data_peparation/get_visual_feature/resnet101_2048.py` | Dataset and model paths |
| `data_peparation/step3_clip_select_masks.py` | Mask and image directories |
| `scripts/VIT_MLP_llama/finetuning_CTRG_MRG.sh` | Python env path and all data/model paths |
| `scripts/VIT_MLP_llama/test_CTRG_MRG.sh` | Python env path and all data/model paths |

The simplest approach is to pass paths as CLI arguments (e.g. `--model_name`, `--output_dir`) rather than editing config files directly.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{pcrl-mrg-2024,
  title     = {See Detail Say Clear: Towards Brain CT Report Generation via Pathological Clue-driven Representation Learning},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2024}
}
```
