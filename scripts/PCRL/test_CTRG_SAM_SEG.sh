set -euo pipefail

PYTHON_BIN=/home/bjutcv/anaconda3/envs/zcx_llama/bin/python
PROJECT_DIR=/home/bjutcv/data/zcx/PCRL-MRG
MODEL_NAME=/home/bjutcv/data/zcx/models/Meta-Llama-3-8B-Instruct
OUTPUT_DIR=${PROJECT_DIR}/PCRL_v0.0.3_bert
SPLIT_DIR=${PROJECT_DIR}/dataset/CTRG_SAM_SEG_dataset/splits
VISUAL_FEATURES_DIR=/home/bjutcv/data/zcx/datasets/CTRG-Brain/vit_img_features
ORIGIN_IMG_DIR=/home/bjutcv/data/zcx/datasets/CTRG-Brain/samples
SEG_OUTPUT_DIR=${PROJECT_DIR}/dataset/CTRG_SAM_SEG_dataset/pred_gt_seg_mask
BERT_MODEL_NAME=bert-base-multilingual-uncased
SAMPLE_ID_KEY=id
VISUAL_TOKEN_COUNT=48
MRG_PROMPT='[Img]{visual_prompt}[/Img][MRG]详细地用中文描述给定的多张脑CT图片并生成一份中文的脑CT报告。'
PEFT_MODEL_NAME=peft_model_best

"${PYTHON_BIN}" \
"${PROJECT_DIR}/scripts/PCRL/test_PCRL.py" \
--batching_strategy padding \
--batch_size_training 4 \
--use_peft \
--peft_method lora \
--quantization \
--seed 8244 \
--val_batch_size 40 \
--test_batch_size 40 \
--dataset ctrg_sam_seg_dataset \
--model_name "${MODEL_NAME}" \
--peft_model_name "${PEFT_MODEL_NAME}" \
--coef_seg 1.0 \
--coef_local 1.0 \
--coef_global 1.0 \
--coef_caption_loss 1.0 \
--coef_image_loss 1.0 \
--output_dir "${OUTPUT_DIR}" \
--save_metrics "${OUTPUT_DIR}/metrics.json" \
--split_dir "${SPLIT_DIR}" \
--sample_id_key "${SAMPLE_ID_KEY}" \
--visual_token_count "${VISUAL_TOKEN_COUNT}" \
--mrg_prompt "${MRG_PROMPT}" \
--visual_features_dir "${VISUAL_FEATURES_DIR}" \
--origin_img_dir "${ORIGIN_IMG_DIR}" \
--seg_output_dir "${SEG_OUTPUT_DIR}" \
--bert_model_name "${BERT_MODEL_NAME}"
