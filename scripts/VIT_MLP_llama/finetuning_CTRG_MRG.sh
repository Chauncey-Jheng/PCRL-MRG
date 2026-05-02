set -euo pipefail

PYTHON_BIN=/home/bjutcv/anaconda3/envs/zcx_llama/bin/python
PROJECT_DIR=/home/bjutcv/data/zcx/PCRL-MRG
MODEL_NAME=/home/bjutcv/data/zcx/models/Meta-Llama-3-8B-Instruct
OUTPUT_DIR=${PROJECT_DIR}/results/baseline_LLM_MRG_v0.0.2
SPLIT_DIR=${PROJECT_DIR}/dataset/CTRG_SAM_SEG_dataset/splits
VISUAL_FEATURES_DIR=/home/bjutcv/data/zcx/datasets/CTRG-Brain/vit_img_features
SAMPLE_ID_KEY=id
VISUAL_TOKEN_COUNT=24
MRG_PROMPT='[Img]{visual_prompt}[/Img][MRG]详细地用中文描述给定的多张脑CT图片并生成一份中文的脑CT报告。'

"${PYTHON_BIN}" \
"${PROJECT_DIR}/scripts/VIT_MLP_llama/finetuning_ViT_MLP_llama.py" \
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
--model_name "${MODEL_NAME}" \
--output_dir "${OUTPUT_DIR}" \
--save_metrics "${OUTPUT_DIR}/metrics.json" \
--split_dir "${SPLIT_DIR}" \
--sample_id_key "${SAMPLE_ID_KEY}" \
--visual_token_count "${VISUAL_TOKEN_COUNT}" \
--mrg_prompt "${MRG_PROMPT}" \
--visual_features_dir "${VISUAL_FEATURES_DIR}"
