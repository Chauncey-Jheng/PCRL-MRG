/home/bjutcv/anaconda3/envs/zcx_llama/bin/python \
/home/bjutcv/data/zcx/llama_baseline/recipes/vit_mlp_llama3/finetuning_ViT_MLP_llama.py \
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
--model_name /home/bjutcv/data/zcx/models/Meta-Llama-3-8B-Instruct \
--output_dir /home/bjutcv/data/zcx/llama_baseline/Llama3_PEFT/baseline_LLM_MRG_v0.0.1 \
--save_metrics /home/bjutcv/data/zcx/llama_baseline/Llama3_PEFT/baseline_LLM_MRG_v0.0.1/metrics.json
