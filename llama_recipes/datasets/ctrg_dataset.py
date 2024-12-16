import datasets

def get_preprocessed_ctrg(train_config, dataset_config, tokenizer, split):

    text_json_dataset_path = "/home/bjutcv/data/zcx/Datasets/CTRG-Brain/report_json_v0.0.0/splits/"

    dataset = datasets.load_dataset(text_json_dataset_path, split=split)

    # 这里的24代表CT中24个层面的视觉特征
    visual_prompt = "<|image_feature|>" * 24
    '''
    Prompt 修改需要到视觉融合层中修改对应视觉特征注入的位置，在文件modelling_llama.py中，LlamaModel方法
    '''
    prompt_MRG = (
        f"[Img]{{visual_prompt}}[/Img][MRG]详细地用中文描述给定的多张脑CT图片并生成一份中文的脑CT报告。"
    )

    def apply_prompt_template(sample):
        return {
            "sample_id": sample["sample_id"],
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