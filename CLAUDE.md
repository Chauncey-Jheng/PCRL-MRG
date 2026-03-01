# CLAUDE.md — AI Assistant Guide for PCRL-MRG

## Project Overview

**PCRL-MRG** is the official code for the EMNLP 2024 paper:
> "See Detail Say Clear: Towards Brain CT Report Generation via Pathological Clue-driven Representation Learning"

The project implements an automated **brain CT report generation** system using a multimodal LLM (LLaMA 3) fine-tuned on the **CTRG** (CT Report Generation) dataset. Reports are generated in **Chinese**.

Dataset source: https://github.com/tangyuhao2016/CTRG

---

## Repository Structure

```
PCRL-MRG/
├── README.md
├── LICENSE
│
├── data_peparation/               # [NOTE: typo in directory name, not "preparation"]
│   ├── segment_anything/          # Meta SAM model (Segment Anything Model)
│   │   ├── modeling/              # SAM architecture (encoder, decoder, transformer)
│   │   ├── utils/                 # SAM utilities (amg, onnx, transforms)
│   │   ├── automatic_mask_generator.py
│   │   ├── build_sam.py
│   │   └── predictor.py
│   ├── get_visual_feature/        # Feature extraction scripts
│   │   ├── clip_vit_1024.py       # Extracts CLIP-ViT-Large-patch14 features → .npz
│   │   └── resnet101_2048.py      # Extracts ResNet-101 features → .npz
│   ├── step1_sam_gen_masks.py     # Pipeline step 1: SAM mask generation
│   ├── step2_re_extract_entities.py  # Step 2: Entity extraction & dataset splits
│   ├── step3_clip_select_masks.py    # Step 3: CLIP-based mask selection per entity
│   └── step4_merge_entities_per_img.py  # Step 4: Merge entity annotations per image
│
├── dataset/                       # Dataset annotation files (no raw images)
│   ├── CTRG_SAM_SEG_dataset/
│   │   ├── seg_mask_anno_train.json   # SAM segmentation annotations (train)
│   │   └── splits/                    # JSON split files
│   │       ├── train.json
│   │       ├── validation.json
│   │       └── test.json
│   ├── origin_anno.json               # Original CTRG annotations
│   └── total_image_list_final.json    # Maps sample_id → list of 24 CT image paths
│
├── llama_recipes/                 # Main training framework (adapted from Meta's llama-recipes)
│   ├── configs/                   # Training configuration dataclasses
│   │   ├── training.py            # train_config (main config)
│   │   ├── peft.py                # LoRA / llama_adapter / prefix configs
│   │   ├── fsdp.py                # FSDP distributed training config
│   │   ├── datasets.py            # Dataset config dataclasses (ctrg_dataset)
│   │   └── wandb.py               # W&B logging config
│   ├── models/                    # Model implementations
│   │   ├── PCRL_llama/            # Main paper contribution: PCRL model
│   │   │   └── modeling_PCRL_llama.py
│   │   ├── ViT_MLP_llama/         # Simpler LLaVA-style baseline
│   │   │   └── modeling_ViT_MLP_llama.py
│   │   └── llama/                 # LLaMA base model code
│   │       ├── configuration_llama.py
│   │       ├── modeling_llama.py
│   │       ├── tokenization_llama.py
│   │       └── tokenization_llama_fast.py
│   ├── datasets/                  # Dataset loaders
│   │   └── ctrg_dataset.py        # CTRG dataset loader & tokenization
│   ├── data/                      # Data utilities
│   │   ├── concatenator.py        # ConcatDataset for packing strategy
│   │   ├── sampler.py             # Distributed samplers
│   │   └── llama_guard/           # Llama Guard fine-tuning utilities
│   ├── policies/                  # Training policies
│   │   ├── activation_checkpointing_functions.py
│   │   ├── anyprecision_optimizer.py
│   │   ├── mixed_precision.py
│   │   └── wrapping.py
│   ├── utils/                     # Training/evaluation utilities
│   │   ├── train_utils.py         # Training loop, eval, setup functions
│   │   ├── test_utils.py          # test_conditional_generation()
│   │   ├── config_utils.py        # update_config(), generate_peft_config()
│   │   ├── dataset_utils.py       # get_preprocessed_dataset()
│   │   ├── memory_utils.py        # MemoryTrace context manager
│   │   ├── flop_utils.py          # FlopMeasure for throughput profiling
│   │   ├── plot_metrics.py        # Metrics plotting
│   │   └── hf_llama_conversion/   # HuggingFace weight conversion utilities
│   └── model_checkpointing/       # Checkpoint save/load handlers
│       └── checkpoint_handler.py
│
└── scripts/                       # Entry-point scripts
    └── VIT_MLP_llama/
        ├── finetuning_ViT_MLP_llama.py  # Training entry point
        ├── test_ViT_MLP_llama.py        # Testing entry point
        ├── finetuning_CTRG_MRG.sh       # Training shell script (paths need updating)
        └── test_CTRG_MRG.sh             # Testing shell script (paths need updating)
```

---

## Architecture

### Two Models Are Implemented

#### 1. PCRL-LLaMA (Main Contribution) — `llama_recipes/models/PCRL_llama/`
The full PCRL model described in the paper:
- **Visual encoder**: ResNet-101 CNN producing 2048-dim features from 24 CT slices (14×14 spatial grid)
- **Segmentation head**: Per-image segmentation map generation
- **Text encoder**: BERT for encoding pathological entity descriptions
- **Contrastive learning**: CLIP-style loss aligning image segments with entity text descriptions
- **Language model**: LLaMA 3-8B-Instruct with INT4 quantization + LoRA
- **Visual projection**: Learned projection layer mapping CNN features → LLM token space

#### 2. ViT-MLP-LLaMA (Baseline) — `llama_recipes/models/ViT_MLP_llama/`
A simpler LLaVA-inspired baseline:
- **Visual features**: Pre-extracted CLIP-ViT-Large-patch14 (1024-dim per slice), loaded from `.npz` files at runtime
- **Projector**: Two-layer MLP: `Linear(1024→4096) → Linear(4096→4096)`
- **Language model**: LLaMA 3-8B-Instruct with INT4 quantization + LoRA
- **Input**: 24 image tokens (`<|image_feature|>` × 24) merged into the text sequence

### Key Special Tokens
```
<|image_feature|>   # Placeholder for one CT slice visual token (24 total)
[Img] / [/Img]      # Image container markers
[MRG]               # Medical Report Generation task indicator
[VQA]               # Visual Question Answering task indicator
```

### CT Prompt Template (Chinese)
```
[Img]<|image_feature|>×24[/Img][MRG]详细地用中文描述给定的多张脑CT图片并生成一份中文的脑CT报告。
```
Each sample has **24 CT slice images**. The visual features cover 8 anatomical layers (3 images per layer).

---

## Data Preparation Pipeline

Run these steps sequentially to prepare training data from raw CTRG images:

### Step 1 — SAM Mask Generation
```bash
python data_peparation/step1_sam_gen_masks.py
```
- Requires SAM checkpoint: `data_peparation/segment_anything/sam_vit_h_4b8939.pth`
- Reads `CTRG_dataset/train_part_1.json`
- Generates segmentation masks saved as `.pkl.gz` files in `CTRG_dataset/mask/`

### Step 2 — Entity Extraction & Dataset Splits
```bash
python data_peparation/step2_re_extract_entities.py
```
- Reads `CTRG_dataset/origin_samples.json`
- Filters non-brain entities (chest, lungs, heart, etc.) using keyword list
- Extracts 26 pathological brain entities (frontal lobe, temporal lobe, lateral ventricle, etc.)
- Splits data 70/20/10 → `train.json`, `test.json`, `validation.json`

### Step 3 — CLIP-Based Mask Selection
```bash
python data_peparation/step3_clip_select_masks.py
```
- Uses CLIP-base-patch32 for entity-to-mask matching
- Maps Chinese entity names → English for CLIP text prompts
- Selects the best matching mask per entity per image

### Step 4 — Merge Entity Annotations Per Image
```bash
python data_peparation/step4_merge_entities_per_img.py
```
- Merges all entity masks and text descriptions per CT image
- Produces final `CTRG_SAM_SEG_dataset/train.json`

### Visual Feature Extraction (for ViT-MLP-LLaMA baseline)
```bash
# Extract CLIP-ViT-Large-patch14 features (1024-dim)
python data_peparation/get_visual_feature/clip_vit_1024.py

# Extract ResNet-101 features (2048-dim)
python data_peparation/get_visual_feature/resnet101_2048.py
```
Output: `.npz` files in `vit_img_features/{last_hidden_state,pooler_output}/` or `cnn_img_features/{att,fc}/`

---

## Training & Evaluation Workflows

### Training (ViT-MLP-LLaMA)
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
Uses `fire.Fire` for CLI argument parsing — all `train_config` fields can be overridden via CLI flags.

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

### Evaluation Metrics
Evaluation uses a bundled `pycocoevalcap` module at `llama_recipes/pycocoevalcap/` (imported as `llama_recipes.pycocoevalcap`) with `compute_scores()`. It computes standard NLG metrics: **BLEU**, **ROUGE**, **METEOR**, **CIDEr** against reference reports. Two variants exist:
- `metrics_for_chs_mrg.py` — Chinese medical report metrics
- `metrics_for_eng_mrg.py` — English medical report metrics

METEOR requires Java (uses `meteor-1.5.jar`). Tokenization uses Stanford CoreNLP JAR (`stanford-corenlp-3.4.1.jar`).

---

## Key Configuration

All training settings are in `llama_recipes/configs/training.py` as the `train_config` dataclass:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | LLaMA 3-8B path | Path to base LLM |
| `context_length` | 256 | Reduced from 4096 for CT reports |
| `batch_size_training` | 1 | Training batch size |
| `val_batch_size` | 1 | Validation batch size |
| `num_epochs` | 3 | Training epochs |
| `lr` | 1e-4 | Learning rate |
| `gamma` | 0.85 | LR scheduler gamma (StepLR, step=1) |
| `peft_method` | "lora" | Fine-tuning method |
| `quantization` | False | INT4 quantization (set True for efficiency) |
| `use_peft` | False | Enable PEFT (set True for LoRA) |
| `seed` | 42 | Reproducibility seed |
| `dataset` | "samsum_dataset" | Must override to "ctrg_dataset" |

### LoRA Configuration (`llama_recipes/configs/peft.py`)
```python
r = 8
lora_alpha = 32
target_modules = ["q_proj", "v_proj"]
lora_dropout = 0.05
task_type = "CAUSAL_LM"
```

---

## Hardcoded Paths — Must Update for New Environments

The codebase has many hardcoded absolute paths from the original development environment (`/home/bjutcv/data/zcx/...`). When setting up on a new machine, update these in:

| File | Hardcoded path | Variable purpose |
|------|---------------|-----------------|
| `llama_recipes/configs/training.py` | `/home/bjutcv/data/zcx/llama3/...` | Default `model_name` |
| `llama_recipes/models/ViT_MLP_llama/modeling_ViT_MLP_llama.py` | `/home/bjutcv/data/zcx/Datasets/` | Visual feature root |
| `llama_recipes/datasets/ctrg_dataset.py` | `/home/bjutcv/data/zcx/Datasets/...` | Dataset JSON path |
| `data_peparation/get_visual_feature/clip_vit_1024.py` | `/home/bjutcv/data/zcx/...` | Dataset and model paths |
| `data_peparation/get_visual_feature/resnet101_2048.py` | `/home/bjutcv/data/zcx/...` | Dataset and model paths |
| `data_peparation/step3_clip_select_masks.py` | `/home/bjutcv/data/zcx/...` | Mask and image dirs |
| `scripts/VIT_MLP_llama/*.sh` | `/home/bjutcv/...` | Python env and all paths |

---

## Dependencies

No `requirements.txt` or `setup.py` is included. Core dependencies based on imports:

```
torch                    # PyTorch (with CUDA)
transformers             # HuggingFace Transformers
peft                     # LoRA and other PEFT methods
accelerate               # HuggingFace Accelerate
bitsandbytes             # INT4/INT8 quantization
datasets                 # HuggingFace Datasets
fire                     # CLI argument parsing
numpy
opencv-python (cv2)
scipy
tqdm
wandb                    # Optional: experiment tracking
```

**Pre-trained models needed**:
- LLaMA 3-8B-Instruct (from Meta)
- CLIP-ViT-Large-patch14 (for ViT-MLP baseline)
- CLIP-base-patch32 (for data preparation step 3)
- SAM ViT-H checkpoint (`sam_vit_h_4b8939.pth`) for data preparation
- ResNet-101 trained on CQ500 (for PCRL model, optional baseline)

---

## Code Conventions

### Python Style
- Python 3.11 (inferred from `__pycache__` filenames)
- No type annotations in most of the custom code
- Many comments are in **Chinese** — this is expected and intentional
- "modified by zcx" comments mark changes from the original Meta llama-recipes
- dataclasses used extensively for configuration

### Model Save/Load
Models are saved using a custom `save_pretrained()` that only saves LoRA and projector weights:
```python
# Save only LoRA + projector params
model.save_pretrained(output_dir, peft_model_name="peft_model_lora_adapter", epoch=5)
# Load
model.from_pretrained(load_dir, peft_model_name="peft_model_lora_adapter", epoch=5)
```
Saves as `.pth` files, not HuggingFace's standard format.

### Dataset Format
Each CTRG sample contains:
- `sample_id`: Integer identifier
- `images`: List of 24 CT image paths
- `findings`: Chinese radiology findings text
- `impression`: Chinese radiology impression text
- `finding_entities`: List of detected brain anatomy entities
- `finding_entities_seg_images`: Paths to CLIP-selected segmentation images

### Batching Strategy
Two modes via `--batching_strategy`:
- `padding`: Pad sequences to same length (default, used in production scripts)
- `packing`: Concatenate sequences with `ConcatDataset`

### Distributed Training
FSDP (Fully Sharded Data Parallel) is supported via `--enable_fsdp` but not used in the provided scripts. Single-GPU training with INT4 quantization is the standard setup.

---

## Data Flow Summary

```
Raw CTRG images (24 CT slices per patient)
    │
    ▼ step1_sam_gen_masks.py
SAM segmentation masks (.pkl.gz per image)
    │
    ▼ step2_re_extract_entities.py
Entity-annotated samples with train/val/test splits
    │
    ▼ step3_clip_select_masks.py
Best matching mask per entity per image
    │
    ▼ step4_merge_entities_per_img.py
Merged entity+mask annotations per image (CTRG_SAM_SEG_dataset/)
    │
    ▼ get_visual_feature/clip_vit_1024.py
Pre-extracted visual features (.npz per sample)
    │
    ▼ Training (finetuning_ViT_MLP_llama.py or PCRL equivalent)
Fine-tuned model: LLaMA 3-8B + LoRA adapter + MLP/PCRL projector
    │
    ▼ Evaluation (test_ViT_MLP_llama.py)
Generated Chinese brain CT reports → pycocoevalcap metrics
```

---

## Notes for AI Assistants

1. **The directory is spelled `data_peparation`** (not `data_preparation`) — this is the actual directory name and should not be corrected without explicit user request.

2. **Hard-coded server paths**: Files reference `/home/bjutcv/data/zcx/...` throughout. When suggesting code changes, note that these paths need to be updated by the user.

3. **Chinese text in code**: Comments, prompts, and entity lists in Chinese are intentional. The model generates Chinese medical reports.

4. **No unit tests**: The project has no test framework. Evaluation is done through model inference on the test split and NLG metric computation.

5. **Evaluation uses bundled pycocoevalcap**: Located at `llama_recipes/pycocoevalcap/` with BLEU, ROUGE, METEOR, CIDEr implementations. METEOR and tokenizer steps require **Java** (JRE) to run the included `.jar` files.

6. **Two distinct models**: PCRL-LLaMA (main contribution with contrastive learning + segmentation) and ViT-MLP-LLaMA (simpler baseline). Only scripts for ViT-MLP-LLaMA are provided in `scripts/`.

7. **Visual features are pre-extracted**: The ViT-MLP model loads `.npz` files at runtime (not processing images during training). Features must be extracted beforehand using `get_visual_feature/clip_vit_1024.py`.

8. **LoRA + INT4 quantization**: The recommended training setup uses `--use_peft --peft_method lora --quantization` for memory efficiency on a single GPU.
