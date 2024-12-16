import numpy as np
import torch
import cv2
import json
import pickle
import gzip
import os
import sys
from tqdm import tqdm

def image_add_mask(image, anns):
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    mask = np.zeros_like(image)

    for ann in sorted_anns:

        m = ann['segmentation']
        color_mask = np.random.randint(0, 256, size=3)
        mask[m] = color_mask
    
    masked_image = cv2.addWeighted(image, 1, mask, 0.5, 0)

    return masked_image

def seg_and_save(image_path, seg_path):
    # print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    print(len(masks))
    masked_image = image_add_mask(image, masks)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)  # 将RGB格式转换为BGR格式
    cv2.imwrite(seg_path,masked_image)

def seg_save_as_pkl_gz(image_path, seg_path):
    # print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    # np.savez(seg_path, **masks)
    with gzip.open(seg_path, 'wb') as f:
        pickle.dump(masks, f)


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam_checkpoint = "data_peparation/segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    points_per_batch=32*8,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    min_mask_region_area=200,  # Requires open-cv to run post-processing
)

with open('CTRG_dataset/train_part_1.json', 'r') as file:
    data = json.load(file)

seg_save_dir = "./CTRG_dataset/mask/"

for sample in tqdm(data):
    images = set(sample["images"])

    for image in images:
        seg_path = os.path.join(seg_save_dir, os.path.basename(os.path.dirname(image)), os.path.basename(image)[:-4] + "_mask.pkl.gz")
        if os.path.exists(seg_path):
            continue
        if not os.path.exists(os.path.dirname(seg_path)):
            os.makedirs(os.path.dirname(seg_path))
        seg_save_as_pkl_gz(image, seg_path)
