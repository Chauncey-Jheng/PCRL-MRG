import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
import json
import numpy as np
from tqdm import tqdm

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径设置

# 路径设置
bct_chr = "/home/bjutcv/data/zcx/dataset/bct_chr"
ctrg_brain = "/home/bjutcv/data/zcx/dataset/CTRG-Brain"

using_dataset = bct_chr

image_folder = os.path.join(using_dataset, "samples")
output_folder_lhs = os.path.join(using_dataset, "vit_img_features/last_hidden_state")
output_folder_po = os.path.join(using_dataset, "vit_img_features/pooler_output")
selected_img_record_file = os.path.join(using_dataset, "total_image_list_final.json")

# 确保输出文件夹存在
os.makedirs(output_folder_lhs, exist_ok=True)
os.makedirs(output_folder_po, exist_ok=True)

# 加载CLIP模型和处理器
model_path = "/home/bjutcv/data/zcx/models/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_path)
# model = CLIPModel.from_pretrained(model_name)
vision_model = CLIPVisionModel.from_pretrained(model_path)
vision_model.to(device)

# 定义函数来提取图像特征
def extract_features(image_paths):
    images = [Image.open(path) for path in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs.to(device)    

    with torch.no_grad():
        vision_outputs = vision_model(**inputs)

    #最后一层隐藏层特征，（257，1024），可以当作local feature
    last_hidden_state = vision_outputs.last_hidden_state
    
    #CLS token 特征，（1024），可以当作global feature
    pooler_output = vision_outputs.pooler_output

    return last_hidden_state, pooler_output

# 从JSON文件中读取图像路径
with open(selected_img_record_file, 'r') as f:
    image_data = json.load(f)

# 遍历JSON文件中的所有图像路径并提取特征
for sample_id, image_paths in tqdm(image_data.items()):
    if len(image_paths) > 0:
        last_hidden_state, pooler_output = extract_features(image_paths)

        # 移动到CPU并转换为numpy数组以保存
        last_hidden_state_np = last_hidden_state.cpu().numpy()
        pooler_output_np = pooler_output.cpu().numpy()

        # 构建特征保存路径
        lhs_save_path = os.path.join(output_folder_lhs, f"{sample_id}.npz")
        po_save_path = os.path.join(output_folder_po, f"{sample_id}.npz")
        
        # 保存 last_hidden_state 特征为 npz 文件
        np.savez_compressed(lhs_save_path, last_hidden_state=last_hidden_state_np)
        
        # 保存 pooler_output 特征为 npz 文件
        np.savez_compressed(po_save_path, pooler_output=pooler_output_np)