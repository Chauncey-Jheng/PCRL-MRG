import json
import os
import numpy as np
import gzip
import pickle
import cv2
import torch
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel

device = "cuda"
model = CLIPModel.from_pretrained("clip-vit-base-patch32").to(device=device)
processor = CLIPProcessor.from_pretrained("clip-vit-base-patch32")

with open('CTRG_dataset/test_525.json', 'r') as file:
    data = json.load(file)

entities_chinese_to_english = {
    '中线': "Midline structure", '脑沟': 'Cerebral sulcus', '基底节区': 'Basal ganglia region', '小脑': 'Cerebellum', '脑干': 'Brain stem', '侧脑室': 'Lateral ventricle', '脑室': 'Ventricle',
    '脑回': 'Gyrus', '脑实质': 'Brain parenchyma', 
    '半卵圆中心': 'Semi-oval center', '放射冠': 'Corona radiata', '上颌窦': 'Maxillary sinus', '额叶': 'Frontal lobe', '丘脑': 'Thalamus', '颞叶': 'Temporal lobe', '顶叶': 'Parietal lobe',
    '枕叶': 'Occipital lobe', '筛窦': 'Ethmoidal sinus', 
    '小脑半球': 'Cerebellar hemisphere', '小脑幕': 'Cerebral tentorium', '大脑镰': 'Cerebral falx', '蝶窦': 'Sphenoid sinus', '颞枕叶': 'Temporal occipital lobe', '第三脑室': 'Third ventricle', '第四脑室': 'Fourth ventricle', '中脑': 'Midbrain',
}

layer_num_entities ={0: ['上颌窦', '筛窦', '蝶窦', '第四脑室', '脑干', '小脑', '小脑半球',],
    1: ['额叶', '颞叶', '颞枕叶', '枕叶', '中脑'],
    2: ['额叶', '丘脑', '侧脑室', '枕叶', '第三脑室', '小脑幕', '颞叶', '中线'],
    3: ['基底节区', '大脑镰', '侧脑室', '枕叶', '颞枕叶', '丘脑', '颞叶', '第三脑室', '额叶', '中线'],
    4: ['侧脑室', '额叶', '放射冠', '颞叶', '枕叶', '颞枕叶', '大脑镰', '中线'],
    5: ['侧脑室', '大脑镰', '额叶', '顶叶', '脑沟', '脑室', '枕叶', '中线'],
    6: ['额叶', '半卵圆中心', '顶叶', '大脑镰', '脑沟', '脑室', '脑实质', '中线'],
    7: ['大脑镰', '脑回', '顶叶', '放射冠', '额叶', '脑沟']}

mask_base_dir = "/home/bjutcv/data/zcx/datasets/CTRG-Brain/mask"
seg_image_base_dir = "/home/bjutcv/data/zcx/datasets/CTRG-Brain/ct_image_seg"

new_data = []
for sample in tqdm(data):
    entities = sample["finding_entities"]
    entities_eng = [entities_chinese_to_english[entity] for entity in entities]

    # Determine which layers each entity belongs to (in fact, you can also directly find all 24 images, with layer restrictions added here to lower computational complexity)
    # Find all the images included in these layers
    # Open the masks corresponding to these images
    # Obtain the images segmented by these masks, that is, change the True part of each mask to the corresponding pixel of the image
    # Encode these masked images with their corresponding text entities using clip encoding
    # Calculate the mask image with the highest probability for each text entity
    # Save these masks separately and record them in the fields of the sample
    
    entity_seg_image_path_list = []
    for entity in entities:
        images_list = []
        for key, value in layer_num_entities.items():
            if entity in value:
                images = sample["images"]
                images_list += images[3*key:3*(key+1)]
        images_set = set(images_list)

        all_seg_images_list = [] # List of segmentation images corresponding to all CT images related to the entity
        all_seg_images_path_list = [] # Record the location of the saved segmentation image and also record which CT image it belongs to
        all_seg_images_mask_list = [] # Record the original mask corresponding to the segmented image

        for img in images_set:
            img_id = os.path.basename(os.path.dirname(img))
            img_name = os.path.basename(img)
            mask_name = img_name[:-4] + "_mask.pkl.gz"
            mask_path = os.path.join(mask_base_dir, img_id, mask_name)
            
            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with gzip.open(mask_path, 'rb') as f:
                mask_data = pickle.load(f)
            
            # Preprocess the segmentation mask to preserve high-quality masks
            sorted_anns = sorted(mask_data, key=(lambda x: x['area']), reverse=True)
            # if len(sorted_anns) > 3:
            #     sorted_anns = sorted_anns[1:-1]
            
            if len(sorted_anns) > 5:
                # When there are too many masks, only keep the 5 masks starting from the second one
                sorted_anns = sorted_anns[1:6]

            # Convert these segmentation masks into corresponding segmentation images
            seg_image_list = [] # List of segmentation images corresponding to a single CT image
            seg_image_path_list = [] # Record the location of the saved segmentation image and also record which CT image it belongs to
            seg_image_mask_list = [] # Record the original mask corresponding to the segmented image
            for i,ann in enumerate(sorted_anns):
                seg_image = np.zeros_like(image)
                m = ann['segmentation']
                seg_image[m] = image[m]
                seg_image_save_path = os.path.join(seg_image_base_dir,img_id,img_name[:-4],str(i)+".jpg")
                seg_image_list.append(seg_image)
                seg_image_path_list.append(seg_image_save_path)
                seg_image_mask_list.append(m)

                # Save split image to local
                seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
                if os.path.exists(seg_image_save_path):
                    continue
                if not os.path.exists(os.path.dirname(seg_image_save_path)):
                    os.makedirs(os.path.dirname(seg_image_save_path))
                cv2.imwrite(seg_image_save_path, seg_image)

            all_seg_images_list += seg_image_list
            all_seg_images_path_list += seg_image_path_list
            all_seg_images_mask_list += seg_image_mask_list
        
        # Use clip encoding to encode these masked images and their corresponding text entities, obtaining the probabilities of each masked image for each text entity
        text_prompt = "A CT image contain " + entities_chinese_to_english[entity]
        inputs = processor(text=[text_prompt], images=all_seg_images_list, return_tensors="pt", padding=True).to(device=device)
        outputs = model(**inputs)
        del inputs
        torch.cuda.empty_cache()    
        logits_per_text = outputs.logits_per_text # this is the text-image similarity score
        del outputs
        torch.cuda.empty_cache()   
        probs = logits_per_text.softmax(dim=1) # we can take the softmax to get the label probabilities
        
        # Obtain the mask with the highest matching degree for the entity
        idx = torch.argmax(probs)
        seg_image_path = all_seg_images_path_list[idx]
        seg_image_mask = all_seg_images_mask_list[idx]

        entity_seg_image_path_list.append(seg_image_path)

        # Save the mask to the same position as its corresponding mask segmentation map
        with gzip.open(seg_image_path[:-4] + "_mask.pkl.gz", 'wb') as f:
            pickle.dump(seg_image_mask, f)
    
    sample["finding_entities_seg_images"] = entity_seg_image_path_list
    new_data.append(sample)

with open('CTRG_dataset/test_525_with_seg.json', 'w',encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)