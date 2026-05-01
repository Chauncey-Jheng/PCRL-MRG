import os

import dataclasses
import random
import torch

from transformers import AutoTokenizer

import fire
from os.path import abspath, join, dirname
import sys
sys.path.insert(len(sys.path), join(abspath(dirname(dirname(dirname(__file__))))))

from llama_recipes.models.PCRL_llama.modeling_PCRL_llama import PCRLLlamaModel

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG

from llama_recipes.utils.config_utils import (
    update_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.train_utils import (
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
)
from llama_recipes.utils.test_utils import test_conditional_generation
from accelerate.utils import is_xpu_available


def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run


def main(**kwargs):
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)

    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
    else:
        local_rank = None
        rank = 0

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    wandb_run = None

    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank == 0:
            wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_feature|>', '[Img]', '[/Img]', '[MRG]', '[VQA]']})

    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_test = get_preprocessed_dataset(
        tokenizer,
        train_config,
        dataset_config,
        split="test",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Testing Set Length = {len(dataset_test)}")

    test_dl_kwargs = get_dataloader_kwargs(train_config, dataset_test, tokenizer, "test")
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **test_dl_kwargs,
    )

    use_cache = False if train_config.enable_fsdp else None
    image_token_index = tokenizer.encode("<|image_feature|>")[0]
    visual_language_model = PCRLLlamaModel(train_config, use_cache, tokenizer, image_token_index, kwargs, wandb_run).cuda()
    visual_language_model.from_pretrained(train_config.output_dir, train_config.peft_model_name)
    print_model_size(visual_language_model, train_config, rank if train_config.enable_fsdp else 0)

    test_conditional_generation(
        visual_language_model,
        train_config,
        test_dataloader,
        local_rank if train_config.enable_fsdp else None,
        tokenizer,
        wandb_run,
    )


if __name__ == "__main__":
    fire.Fire(main)
