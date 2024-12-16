import json
import os
import gzip
import pickle

with open('CTRG_SAM_SEG_dataset/seg_mask_anno_train.json', 'r') as file:
    data = json.load(file)

new_data = []

for sample in data:
    # Merge different entities on the same image
    finding_entities_seg_images = sample["finding_entities_seg_images"]
    finding_etities_lession = sample["finding_entities_lession"]
    images = [os.path.basename(img_path) for img_path in sample["images"]]
    image_contain_seg_entity = []

    image_with_txt = {i:"" for i in range(24)}
    all_img_seg_merged_path = []

    # Statistically analyze the images that appear
    for seg_image_path in finding_entities_seg_images:
        image_name = os.path.dirname(seg_image_path)
        if image_name not in image_contain_seg_entity:
            image_contain_seg_entity.append(image_name)
    # Search for the corresponding text entity according to the image and merge it
    for image_name in image_contain_seg_entity:
        # Obtain the corresponding number for the image
        img_idx = images.index(os.path.basename(image_name) + ".jpg")
        for idx, seg_image_path in enumerate(finding_entities_seg_images):
            if image_name in seg_image_path:
                # Obtain the corresponding entity text description
                text = finding_etities_lession[idx]
                # Store the text in the corresponding location of imagew_ith_txt
                image_with_txt[img_idx] += text + ";"

        img_seg_merged_path = os.path.join(image_name, "_merge.pkl.gz") 
        all_img_seg_merged_path.append(img_seg_merged_path)
    # Filter out the hollow parts in imagew_ith_txt
    image_with_txt = {k:v for k, v in image_with_txt.items() if v != ""}
    image_number = [k for k in image_with_txt.keys()]
    txts = [v[:-1] for v in image_with_txt.values()]
    sample["finding_idx_images_contain_entity"] = image_number
    sample["finding_entities_lession_per_img"] = txts
    sample["finding_merged_masks_path"] = all_img_seg_merged_path
    new_data.append(sample)


with open('CTRG_SAM_SEG_dataset/train.json', 'w',encoding="utf-8") as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)
