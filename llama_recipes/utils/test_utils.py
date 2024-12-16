from datetime import datetime
import torch
from tqdm import tqdm
import json

from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available
from llama_recipes.pycocoevalcap import compute_scores

def test_conditional_generation(model, train_config, test_dataloader, local_rank, tokenizer, wandb_run):
    """
    Test the model on the test dataloader, using conditional generation method

    Args:
        model: The model to evaluate
        test_dataloader: The dataloader containing the test data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """

    model.eval()
    sample_ids = []
    quiz_ids = []
    eval_preds = []
    ground_truth = []

    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(test_dataloader,colour="yellow", desc="Testing Epoch", dynamic_ncols=True)):

            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass
                outputs = model.generate(**batch)

            gts = batch["labels"].cpu().numpy().tolist()
            gts = [list(filter(lambda x: x != -100, gt)) for gt in gts]
            
            # decode result without prompt, modified by zcx
            eval_preds.append(
                tokenizer.batch_decode(outputs, skip_special_tokens=True)
            )

            ground_truth.append(
                tokenizer.batch_decode(gts, skip_special_tokens=True)
            )            
            sample_ids.append(
                batch["sample_id"][:,0].tolist()
            )
            if "quiz_id" in batch.keys():
                quiz_ids.append(
                    batch["quiz_id"][:,0].tolist()
                )

    res_gts_set = []
    for i in range(len(eval_preds)):
        for j in range(len(eval_preds[i])):
            if len(quiz_ids) != 0:
                re_gt_sample = {
                "quiz_id": quiz_ids[i][j],
                "sample_id": sample_ids[i][j],
                "prediction": eval_preds[i][j],
                "ground_truth": ground_truth[i][j],
                }
            else:
                re_gt_sample = {
                    "sample_id": sample_ids[i][j],
                    "prediction": eval_preds[i][j],
                    "ground_truth": ground_truth[i][j],
                }
            res_gts_set.append(re_gt_sample)

    gts = {sample["sample_id"]:[sample["ground_truth"]] for sample in res_gts_set}
    res = {sample["sample_id"]:[sample["prediction"]] for sample in res_gts_set}
    scores = compute_scores(gts, res)

    save_file = f"{train_config.output_dir}/test_result-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(save_file,'w', encoding='utf-8') as json_file:
        json.dump(res_gts_set, json_file, ensure_ascii=False, indent=4)
    
    save_info_file = save_file[:-5] + "_info.txt"
    infos = [str(scores)+"\n", str(train_config)]
    with open(save_info_file, "w", encoding="utf-8") as info_file:
        info_file.writelines(infos)

    return 
