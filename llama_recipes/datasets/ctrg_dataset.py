import json
import datasets
from pathlib import Path

def get_preprocessed_ctrg(train_config, dataset_config, tokenizer, split):

    if not dataset_config.split_dir:
        raise ValueError("ctrg_dataset requires --split_dir to point to the dataset split JSON directory.")
    if not dataset_config.mrg_prompt:
        raise ValueError("ctrg_dataset requires --mrg_prompt to define the report-generation prompt template.")
    text_json_dataset_path = Path(dataset_config.split_dir) / f"{split}.json"

    with text_json_dataset_path.open("r", encoding="utf-8") as f:
        dataset = datasets.Dataset.from_list(json.load(f))

    # 这里的24代表CT中24个层面的视觉特征
    visual_prompt = "<|image_feature|>" * int(dataset_config.visual_token_count)
    '''
    Prompt 修改需要到视觉融合层中修改对应视觉特征注入的位置，在文件modelling_llama.py中，LlamaModel方法
    '''
    prompt_MRG = dataset_config.mrg_prompt

    def apply_prompt_template(sample):
        return {
            "sample_id": sample[dataset_config.sample_id_key],
            "prompt": prompt_MRG.format(visual_prompt=visual_prompt),
            "findings": sample["findings"],
            "impression": sample["impression"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)  # 视觉token和文本提示
        findings = tokenizer.encode(sample["findings"] + tokenizer.eos_token, add_special_tokens=False)


        if split == "train":
            sample = {
                "sample_id": [int(sample["sample_id"])],
                "input_ids": prompt + findings,
                "attention_mask" : [1] * (len(prompt) + len(findings)),
                "labels": [-100] * len(prompt) + findings,
            }
        else:
            sample = {
            "sample_id": [int(sample["sample_id"])],
            "input_ids": prompt,
            "attention_mask" : [1] * len(prompt),
            "labels": findings,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
