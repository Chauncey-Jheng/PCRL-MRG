"""Standalone inference engine for the ViT-MLP-LLaMA brain CT report generator.

`ViTLlamaModel.generate()` (in llama_recipes/models/ViT_MLP_llama/modeling_ViT_MLP_llama.py)
only knows how to resolve a sample's visual features by looking up pre-extracted
`.npz` files on disk via `sample_id` — that's fine for evaluating the fixed CTRG test
split, but a deployed server has no such sample_id: it only has whatever 24 CT slice
images were just uploaded. This module loads the fine-tuned checkpoint plus the CLIP
vision encoder once at process startup, extracts CLIP features from live images, and
calls the `generate_from_image_features()` entry point added to the model for exactly
this purpose.
"""

import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPProcessor, CLIPVisionModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llama_recipes.configs import train_config as TRAIN_CONFIG  # noqa: E402
from llama_recipes.models.ViT_MLP_llama.modeling_ViT_MLP_llama import ViTLlamaModel  # noqa: E402

VISUAL_TOKEN_COUNT = 24
MRG_PROMPT_TEMPLATE = (
    "[Img]{visual_prompt}[/Img][MRG]"
    "详细地用中文描述给定的多张脑CT图片并生成一份中文的脑CT报告。"
)


class ReportGenerationEngine:
    def __init__(
        self,
        model_name: str,
        clip_model_path: str,
        checkpoint_dir: str,
        peft_model_name: str = "peft_model_best",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|image_feature|>", "[Img]", "[/Img]", "[MRG]", "[MRS]"]}
        )
        image_token_index = self.tokenizer.encode("<|image_feature|>")[0]

        # Mirrors scripts/VIT_MLP_llama/test_CTRG_MRG.sh's model construction (same
        # quantization/LoRA settings the checkpoint was trained and evaluated with) —
        # just without the dataset/dataloader machinery, since inference gets its
        # "sample" from an HTTP request instead of a fixed JSON split.
        train_config = TRAIN_CONFIG()
        train_config.model_name = model_name
        train_config.quantization = True
        train_config.use_peft = True
        train_config.peft_method = "lora"
        train_config.output_dir = checkpoint_dir
        train_config.peft_model_name = peft_model_name

        self.model = ViTLlamaModel(
            train_config,
            use_cache=None,
            tokenizer=self.tokenizer,
            image_token_index=image_token_index,
            kwargs={},
            wandb_run=None,
        ).cuda()
        self.model.from_pretrained(train_config.output_dir, train_config.peft_model_name)
        self.model.eval()

        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_path).cuda().eval()

    @torch.no_grad()
    def _extract_global_features(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.clip_model(**inputs)
        return outputs.pooler_output  # (N, 1024), same feature CLIP-ViT-1024 extraction used at training time

    @torch.no_grad()
    def generate_report(
        self,
        images: List[Image.Image],
        max_gen_len: int = 300,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        if len(images) != VISUAL_TOKEN_COUNT:
            raise ValueError(
                f"需要恰好 {VISUAL_TOKEN_COUNT} 张CT切片图像（对应训练时8个层面each 3张），"
                f"实际收到 {len(images)} 张。"
            )

        image_features = self._extract_global_features(images)

        visual_prompt = "<|image_feature|>" * VISUAL_TOKEN_COUNT
        prompt_text = self.tokenizer.bos_token + MRG_PROMPT_TEMPLATE.format(visual_prompt=visual_prompt)
        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = torch.tensor([input_ids], device="cuda")
        attention_mask = torch.ones_like(input_ids)

        out_tokens = self.model.generate_from_image_features(
            image_features,
            input_ids,
            attention_mask,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        return self.tokenizer.decode(out_tokens[0], skip_special_tokens=True).strip()
