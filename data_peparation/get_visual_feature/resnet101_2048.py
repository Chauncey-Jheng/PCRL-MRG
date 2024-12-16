import os
import torch
from PIL import Image
import json
import numpy as np
from tqdm import tqdm

from resnet import resnet101
# import resnet101

from torchvision import transforms as trn

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 路径设置
bct_chr = "/home/bjutcv/data/zcx/dataset/bct_chr"
ctrg_brain = "/home/bjutcv/data/zcx/dataset/CTRG-Brain"

using_dataset = ctrg_brain

image_folder = os.path.join(using_dataset, "samples")
output_folder_att = os.path.join(using_dataset, "cnn_img_features/att")
output_folder_fc = os.path.join(using_dataset, "cnn_img_features/fc")
selected_img_record_file = os.path.join(using_dataset, "total_image_list_final.json")

# 确保输出文件夹存在
os.makedirs(output_folder_att, exist_ok=True)
os.makedirs(output_folder_fc, exist_ok=True)


# 加载Resnet模型
model_path = "/home/bjutcv/data/zcx/models/ResNet101_CQ500/dan_CQ500_resnet101.pth"
net = resnet101(num_classes=2)
net.load_state_dict(torch.load(model_path), strict=False)
net.cuda()
net.eval()

bbox_per_persion = []


preprocess = trn.Compose([
    trn.ToTensor(),
    # trn.Normalize([0.189, 0.189, 0.189], [0.056, 0.056, 0.056]),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])


# 定义函数来提取图像特征
def extract_features(image_paths):
    fc = np.zeros([24, 2048], dtype='float32')
    att = np.zeros([24, 14, 14, 2048], dtype='float32')
    for i_, i in enumerate(image_paths):
        print(i_, ':', image_paths)
        I = np.array(Image.open(image_paths).convert('RGB'))
        I = preprocess(I)
        with torch.no_grad():
            tmp_fc, tmp_att = net(I.cuda(), 14)
            fc[i_] = tmp_fc.data.cpu().float().numpy()
            att[i_] = tmp_att.data.cpu().float().numpy()

# 从JSON文件中读取图像路径
with open(selected_img_record_file, 'r') as f:
    image_data = json.load(f)

# 遍历JSON文件中的所有图像路径并提取特征
for sample_id, image_paths in tqdm(image_data.items()):
    if len(image_paths) > 0:
        att, fc = extract_features(image_paths)

        # 移动到CPU并转换为numpy数组以保存
        att_np = att.cpu().numpy()
        fc_np = fc.cpu().numpy()

        # 构建特征保存路径
        att_save_path = os.path.join(output_folder_att, f"{sample_id}.npz")
        fc_save_path = os.path.join(output_folder_fc, f"{sample_id}.npz")
        
        # 保存 last_hidden_state 特征为 npz 文件
        np.savez_compressed(att_save_path, att=att_np)
        
        # 保存 pooler_output 特征为 npz 文件
        np.savez_compressed(fc_save_path, fc=fc_np)