# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

#added by zcx
@dataclass
class ctrg_dataset:
    dataset: str = "ctrg_dataset"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    split_dir: str = ""
    sample_id_key: str = "id"
    visual_token_count: int = 24
    mrg_prompt: str = ""


@dataclass
class ctrg_sam_seg_dataset(ctrg_dataset):
    dataset: str = "ctrg_sam_seg_dataset"


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
