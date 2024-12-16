from typing import List, Optional, Tuple, Union
from ..llama.modeling_llama import LlamaForCausalLM
from ..llama.modeling_llama import LlamaDecoderLayer
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
from peft import get_peft_model, prepare_model_for_kbit_training
from ...utils.config_utils import(
    update_config,
    generate_dataset_config,
    get_dataloader_kwargs,
    generate_peft_config,
)

from transformers import BertTokenizer, BertModel

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import ModelOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto import AutoModel, AutoModelForCausalLM

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)

import os
from torch.nn.parameter import Parameter

import json
import gzip
import pickle


import cv2
from scipy.ndimage import zoom
        
#During the data processing phase
#For each sample, there are corresponding entities,
#Each entity has a corresponding mask, and these masks correspond to which of the 24 images
#Merge masks of the same image together
#In this way, we obtained masks for 24 images
#Similarly, each entity has a corresponding entity description text, which can be mapped to which of the 24 images
#Concatenate the entity description text of the same image together and remove duplicate text
#In this way, we obtained text descriptions corresponding to 24 images
#Extract features from 24 images using a shared downsampling layer with dimensions of 24 * 14 * 14 * 2048
#Generate segmentation maps for 24 images using a shared segmentation head
#Transform 24 image features into corresponding visual embeddings using a shared visual embedding layer
#Convert the text features corresponding to 24 images into text embeddings using a shared text embedding layer
#Expand the embeddings according to the number of images: [batchsize*img_num, 512]
#Calculate the cross entropy between the two and perform comparative learning optimization

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor,  coef_caption_loss = 1.0, coef_image_loss = 1.0) -> torch.Tensor:
    coef_caption_loss = coef_caption_loss
    coef_image_loss = coef_image_loss
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (coef_caption_loss * caption_loss + coef_image_loss * image_loss) / 2.0

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    # original padding is 1; original dilation is 1
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)
    
class BasicConvBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicConvBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SegAlignment(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seg_pred, masks):
        loss_mask = self.dice_bce_loss(seg_pred, masks)
        return loss_mask

    def dice_bce_loss(self, input, targets):
        # input [batchsize_local, 28, 28]
        # targets [batchsize_local, 512, 512]
        # Scale the target mask to the same shape as the input
        targets = targets.float()
        targets = F.interpolate(targets.unsqueeze(1), size=input.shape[1:], mode='nearest').squeeze(1)
        # Obtain the size N of each batch
        N = targets.size()[0]
        smooth = 1 # Smooth variable
        # Reshape the width and height to the same latitude
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        # Calculate intersection
        intersection = input_flat * targets_flat 
        # Calculate the average loss per image in a batch
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss

class LocalAlignment(nn.Module):
    def __init__(self, coef_caption_loss=1, coef_image_loss=1):
        super().__init__()    
        self.coef_caption_loss = coef_caption_loss
        self.coef_image_loss = coef_image_loss

        # using for textual-visual alignment
        self.logit_scale_init_value_local = 2.6592
        self.logit_scale_local = nn.Parameter(torch.tensor(self.logit_scale_init_value_local))

    def forward(self, visual_local_embed, textual_local_embed):
        # compute local_loss
        # normalized features
        visual_local_embed = visual_local_embed / visual_local_embed.norm(p=2, dim=-1, keepdim=True)
        textual_local_embed = textual_local_embed / textual_local_embed.norm(p=2, dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale_local = self.logit_scale_local.exp()
        logits_per_text_local = torch.matmul(textual_local_embed, visual_local_embed.t()) * logit_scale_local
        # logits_per_image = logits_per_text.t()
        loss_local = clip_loss(logits_per_text_local, self.coef_caption_loss, self.coef_image_loss)

        return loss_local


class GlobalAlignment(nn.Module):
    def __init__(self, coef_caption_loss=1, coef_image_loss=1):
        super().__init__()
        self.coef_caption_loss = coef_caption_loss
        self.coef_image_loss = coef_image_loss

        # using for textual-visual alignment
        self.logit_scale_init_value_global = 2.6592
        self.logit_scale_global = nn.Parameter(torch.tensor(self.logit_scale_init_value_global))

    def forward(self, visual_global_embed, textual_global_embed):
        # compute global_loss
        # normalized features
        visual_global_embed = visual_global_embed / visual_global_embed.norm(p=2, dim=-1, keepdim=True)
        textual_global_embed = textual_global_embed / textual_global_embed.norm(p=2, dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale_global = self.logit_scale_global.exp()
        logits_per_text_global = torch.matmul(textual_global_embed, visual_global_embed.t()) * logit_scale_global
        # logits_per_image = logits_per_text.t()
        loss_global = clip_loss(logits_per_text_global, self.coef_caption_loss, self.coef_image_loss)

        return loss_global


class PCRLLlamaModel(nn.Module):
    '''
    This model has three parts in total;
    The first part is text and visual alignment, which includes two branches: local fine-grained alignment and global alignment;
    The second part is visual and segmentation alignment;
    The third part is to generate text from images;
    '''

    def __init__(self, train_config, use_cache, tokenizer, image_token_index, kwargs, wandb_run):
        super().__init__()
        
        self.tokenizer = tokenizer

        self.image_token_index = image_token_index
        self.ignore_index = -100
        self.stop_token_id = 128001
        self.pad_token_id = self.stop_token_id # or -1
        self.pad_token_id_bert = 0
        self.context_length = train_config.context_length
        self.save_peft_model_name = "peft_model_lora_adapter.pth"

        self.projection_dim_local = 512
        self.projection_dim_global = 512

        with open('/home/bjutcv/data/zcx/llama3/CTRG_SAM_SEG_dataset/splits/train.json', 'r') as file:
             self.train_data_json = json.load(file)

        with open('/home/bjutcv/data/zcx/llama3/CTRG_SAM_SEG_dataset/splits/test.json', 'r') as file:
             self.test_data_json = json.load(file)
        
        with open('/home/bjutcv/data/zcx/llama3/CTRG_SAM_SEG_dataset/splits/validation.json', 'r') as file:
             self.validation_data_json = json.load(file)        

        ## local branch ####################################
        self.local_res = self._make_res_layer(2048,2048,BasicConvBlock,2,2,2)
        self.local_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.local_visual_projector = nn.Linear(2048,4096)
        self.local_visual_embed = nn.Linear(4096, self.projection_dim_local)
        self.local_text_embed = nn.Linear(4096, self.projection_dim_local)
        self.local_text_embed_bert = nn.Linear(768, self.projection_dim_global)
        self.local_seg = self._make_seg_layer(2048, 2048)

        ## global_branch ####################################
        self.global_visual_projector = nn.Sequential(
            nn.Linear(2048,4096),
            nn.Linear(4096,4096),
        )
        self.global_avg_pool = nn.AdaptiveMaxPool1d(1)
        self.global_visual_embed = nn.Linear(4096, self.projection_dim_global)
        self.global_text_embed = nn.Linear(4096, self.projection_dim_global)
        self.global_text_embed_bert = nn.Linear(768, self.projection_dim_global)


        # alignment #########################################
        self.coef_seg = train_config.coef_seg
        self.coef_local = train_config.coef_local
        self.coef_global = train_config.coef_global
        self.coef_caption_loss = train_config.coef_caption_loss
        self.coef_image_loss = train_config.coef_image_loss
        self.seg_alignment = SegAlignment()
        self.local_alignment = LocalAlignment(self.coef_caption_loss,self.coef_image_loss)
        self.global_alignment = GlobalAlignment(self.coef_caption_loss,self.coef_image_loss)

        # textual encoder ####################################
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.bert_encoder =  BertModel.from_pretrained(
            "bert-base-multilingual-uncased", 
            ).to("cuda")

        # language model #####################################
        self.language_model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_4bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        )
        # If there is a mismatch between tokenizer vocab size and embedding matrix,
        # throw a warning and then expand the embedding matrix
        if len(tokenizer) > self.language_model.get_input_embeddings().weight.shape[0]:
            print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
            self.language_model.resize_token_embeddings(len(tokenizer))
        # print_model_size(self.text_model, train_config, rank if train_config.enable_fsdp else 0)
        # Prepare the model for int8 training if quantization is enabled
        if train_config.quantization:
            self.language_model = prepare_model_for_kbit_training(self.language_model)
        # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
        if train_config.use_peft:
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(self.language_model, peft_config)
            model.print_trainable_parameters()
            if wandb_run:
                wandb_run.config.update(peft_config)

    def _get_data_by_id(self, split, id):
        if split == "train":
            for sample in self.train_data_json:
                if int(sample["id"]) == int(id):
                    return sample
        if split == "test":
            for sample in self.test_data_json:
                if int(sample["id"]) == int(id):
                    return sample
        if split == "validation":
            for sample in self.validation_data_json:
                if int(sample["id"]) == int(id):
                    return sample

    def _prepare_visual_feature_data(self, sample_id):
        '''
        Take the preprocessed visual data based on the sample ID
        The size of the returned batch_image_feature-local is [24,14,142048]
        The size of batch_image_feature_global is [24,2048]
        '''
        batch_image_feature_local = []
        batch_image_feature_global = []
        for i in range(len(sample_id)):
            id = int(sample_id[i])
            visual_features_dir_path = "/home/bjutcv/data/zcx/dataset/CTRG-Brain/feature/"
            image_feature_global_path = visual_features_dir_path + "fc/" + str(id) +".npy"
            image_feature_local_path = visual_features_dir_path + "att/" + str(id) +".npz"
            cuda = torch.device('cuda:0')
            image_feature_global = torch.from_numpy(np.load(image_feature_global_path)).to(cuda)
            batch_image_feature_global.append(image_feature_global)
            
            npz_file = np.load(image_feature_local_path)
            arrays = {key: npz_file[key] for key in npz_file.files}
            tensors = {key: torch.tensor(value).to(cuda) for key, value in arrays.items()}
            image_feature_local = tensors["feat"]
            batch_image_feature_local.append(image_feature_local)
        batch_image_feature_local = torch.stack(batch_image_feature_local, dim=0).to(cuda)
        batch_image_feature_global = torch.stack(batch_image_feature_global, dim=0).to(cuda)
        return batch_image_feature_local, batch_image_feature_global
    
    def visualize_seg_mask(self, sample_id, logits_seg, batch_masks):
        cursor = 0
        save_seg_dir = "/home/bjutcv/data/zcx/llama3/CTRG_SAM_SEG_dataset/pred_gt_seg_mask"
        origin_img_dir = "/home/bjutcv/data/zcx/dataset/CTRG-Brain/samples/"

        for i in range(len(sample_id)):
            id = int(sample_id[i])
            # Retrieve text data from JSON based on ID
            sample = self._get_data_by_id(split="train",id=id)
            # Obtain image numbers containing entities
            idx_images = sample["finding_idx_images_contain_entity"]
            # Obtain the segmentation masks corresponding to these images
            cur_logits_seg = logits_seg[cursor:cursor+len(idx_images)]
            cur_pred_masks = (cur_logits_seg >= 0.5).int()
            cur_gt_masks = batch_masks[cursor:cursor+len(idx_images)].int()
            # Save these segmentation masks to the corresponding folder
            for j, idx_image in enumerate(idx_images):
                # Read the corresponding original image
                origin_image_path = os.path.join(origin_img_dir,str(id),os.path.basename(os.path.dirname(sample["finding_merged_masks_path"][j]))+".jpg")
                origin_image = cv2.imread(origin_image_path)
                save_origin_image_path = os.path.join(save_seg_dir, str(id), os.path.basename(origin_image_path))
                save_gt_mask_path = save_origin_image_path[:-4]+"_gt_seg.jpg"
                save_pred_mask_path = save_origin_image_path[:-4] + "_pred_seg.jpg"
                save_gt_mask_in_img_path = save_origin_image_path[:-4]+"_gt_seg_in_img.jpg"
                save_pred_mask_in_img_path = save_origin_image_path[:-4] + "_pred_seg_in_img.jpg"                
                pred_mask = cur_pred_masks[j].cpu().numpy().astype(np.uint8) * 255 # Expand the value to the range of 0-255
                if not os.path.exists(os.path.dirname(save_origin_image_path)):
                    os.makedirs(os.path.dirname(save_origin_image_path))
                cv2.imwrite(save_origin_image_path, origin_image)
                cv2.imwrite(save_pred_mask_path, pred_mask)
                gt_mask = cur_gt_masks[j].cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(save_gt_mask_path, gt_mask)
                
                cur_gt_mask = cur_gt_masks[j].cpu().numpy()
                # Extend the mask to the same shape as the image
                cur_gt_mask_3d = np.repeat(cur_gt_mask[:, :, np.newaxis], 3, axis=2)
                # Apply Mask to Image
                gt_masked_image = np.where(cur_gt_mask_3d == 1, origin_image, 0)
                cv2.imwrite(save_gt_mask_in_img_path, gt_masked_image)

                cur_pred_mask = cur_pred_masks[j].cpu().numpy()
                # Use nearest neighbor interpolation to enlarge
                zoom_factor = 512 / 14
                cur_pred_mask = zoom(cur_pred_mask, zoom_factor, order=0)
                # Extend the mask to the same shape as the image
                cur_pred_mask_3d = np.repeat(cur_pred_mask[:, :, np.newaxis], 3, axis=2)
                # Apply Mask to Image
                pred_masked_image = np.where(cur_pred_mask_3d == 1, origin_image, 0)
                cv2.imwrite(save_pred_mask_in_img_path, pred_masked_image)             

    def _prepare_local_alignment_batch(self, batch_image_feature_local, sample_id):
        '''
        Prepare a batch for the local alignment module based on the sample_id in the batch
        The batch size here is larger than the original batch, and each sample contains multiple fine-grained entities of indefinite length
        '''
        batch_image_feature_for_alignment = []
        batch_summary_ids = []
        batch_seg_masks = []
        cuda = torch.device('cuda:0')
        for i in range(len(sample_id)):
            id = int(sample_id[i])

            sample = self._get_data_by_id(split="train", id=id)
            
            # Obtain image features containing entities
            idx_images = sample["finding_idx_images_contain_entity"]
            # image_feature_local = torch.index_select(image_feature_local, 0, idx_images)
            for idx in idx_images:
                batch_image_feature_for_alignment.append(batch_image_feature_local[i][idx])

            # Get text tokens containing entities
            # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
            head_prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant for summarizing brain CT report.<|eot_id|><|start_header_id|>user<|end_header_id|>
""" 
            entities_lession_per_img = sample["finding_entities_lession_per_img"]
            tail_prompt = ". Please summarize the above cranial diagnosis text into one word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

            summary_ids = [self.tokenizer.encode(head_prompt + entity_lession + tail_prompt) 
                           for entity_lession in entities_lession_per_img]
            batch_summary_ids += summary_ids
            
            # Obtain a mask containing entities
            masks_path = sample["finding_merged_masks_path"]
            masks_list = []
            for m_path in masks_path:
                # Open the segmentation mask corresponding to the image
                with gzip.open(m_path, 'rb') as f:
                    mask_data = pickle.load(f)
                    masks_list.append(mask_data)
            batch_seg_masks += masks_list

        batch_image_feature_for_alignment = torch.stack(batch_image_feature_for_alignment, dim=0).to(cuda)

        # Fill in summary_ids to the same length using pad_tokeniid
        pad_token_id = self.pad_token_id
        max_summary_ids_len = max([len(id) for id in batch_summary_ids])
        batch_summary_ids_att_masks =[[1]*len(summary_id) + [0]*(max_summary_ids_len-len(summary_id))
                                      for summary_id in batch_summary_ids]
        batch_summary_ids = [summary_id + [pad_token_id]*(max_summary_ids_len-len(summary_id)) 
                             for summary_id in batch_summary_ids]
        
        batch_summary_ids = torch.tensor(np.array(batch_summary_ids)).to(cuda)
        batch_summary_ids_att_masks = torch.tensor(batch_summary_ids_att_masks).to(cuda)
        batch_seg_masks = torch.tensor(np.array(batch_seg_masks)).to(cuda)

        return batch_image_feature_for_alignment, batch_summary_ids, batch_seg_masks, batch_summary_ids_att_masks

    def _prepare_global_alignment_batch(self, sample_id):
        '''
        Only text tokens are returned here, The batch size remains the same as before
        '''
        batch_global_summary_ids = []
        cuda = torch.device('cuda:0')
        for i in range(len(sample_id)):
            id = int(sample_id[i])

            # Retrieve text data from JSON based on ID
            sample = self._get_data_by_id(split="train", id=id)

            # Get text tokens containing entities
            findings = sample["findings"]
            head_prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant for summarizing brain CT report.<|eot_id|><|start_header_id|>user<|end_header_id|>
""" 
            tail_prompt = ". Please summarize the above cranial diagnosis text into one word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

            summary_ids = self.tokenizer.encode(head_prompt + findings + tail_prompt)
            batch_global_summary_ids.append(summary_ids)

        # Fill in summary_ids to the same length using pad_tokeniid
        pad_token_id = self.pad_token_id
        max_summary_ids_len = max([len(id) for id in batch_global_summary_ids])
        batch_global_summary_ids_att_masks =[[1]*len(summary_id) + [0]*(max_summary_ids_len-len(summary_id))
                                      for summary_id in batch_global_summary_ids]
        batch_global_summary_ids = [summary_id + [pad_token_id]*(max_summary_ids_len-len(summary_id))
                             for summary_id in batch_global_summary_ids]
        
        batch_global_summary_ids = torch.tensor(np.array(batch_global_summary_ids)).to(cuda)
        batch_global_summary_ids_att_masks = torch.tensor(batch_global_summary_ids_att_masks).to(cuda)

        return batch_global_summary_ids, batch_global_summary_ids_att_masks

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_res_layer(self, inplanes, planes, block, blocks, downsample_kernel_size=2, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_seg_layer(self, inplanes, planes):
        layers = []
        layers.append(nn.ConvTranspose2d(inplanes, planes, 2, 2, 0))
        layers.append(nn.ReLU(inplace=True))
        # 这里仅作单个实体的前景背景分割，故输出通道数为1
        layers.append(nn.Conv2d(planes, 1, 1, 1, 0))

        return nn.Sequential(*layers)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()
    

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(
        self,
        sample_id: torch.IntTensor = None, 
        input_ids: torch.LongTensor = None,
        image_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True, 
    ):
        r"""
        image_features shape must be (num_images, num_image_patches, embed_dim)
        ```"""

        # Obtain visual data using sample_id
        sample_id = sample_id.squeeze(1).tolist()
        image_feature_local, image_feature_global = self._prepare_visual_feature_data(sample_id)

        # The local visual features are further processed by a ResNet layer, where the visual features are [4,24,14,14,2048]
        # It needs to be converted into [4,24,2048,14,14] first
        image_feature_local = image_feature_local.permute(0,1,4,2,3)
        
        # Selectors, use sample_id to retrieve data for local alignment and SEG alignment
        # local_image_feat_for_align: [batchsize_local, 2048, 14, 14]
        # local_summary_ids: [batchsize_local, -1]
        # batch_masks: [batchsize_local, 512, 512]
        local_image_feat_for_align, local_summary_ids, batch_masks, local_summary_att_masks = self._prepare_local_alignment_batch(image_feature_local, sample_id)
        # local_image_feat_for_align, local_summary_ids, batch_masks, local_summary_att_masks = self._prepare_local_alignment_batch_bert(image_feature_local, sample_id)
        
        # Processing local visual features
        # Downsampling, [batchsize_local, 2048, 7, 7]
        local_image_feat_for_align = self.local_res(local_image_feat_for_align)

        # Obtain the predicted segmentation mask
        pred_seg = self.local_seg(local_image_feat_for_align)
        logits_seg = F.sigmoid(pred_seg) # [batchsize_local, 1, 14, 14]
        logits_seg = logits_seg.squeeze(dim=1)

        # Visualize and save segmentation masks
        # self.visualize_seg_mask(sample_id, logits_seg, batch_masks)

        # Calculate loss using predicted segmentation mask and real mask
        seg_loss = self.seg_alignment(logits_seg, batch_masks)

        # [batchsize_local, 2048]
        local_image_feat_for_align = self.local_avg_pool(local_image_feat_for_align).squeeze(-1).squeeze(-1)
        # [batchsize_local, 4096]
        local_image_feat_for_align = self.local_visual_projector(local_image_feat_for_align)
        # [batchsize_local, embeding_size]
        local_image_embed_for_align = self.local_visual_embed(local_image_feat_for_align)

        # Calculate the representation of local text features
        ret_local = self.language_model(
            input_ids = local_summary_ids,
            attention_mask = local_summary_att_masks,
            output_hidden_states = True,
            )
        # ret_local = self.bert_encoder(
        #     input_ids = local_summary_ids,
        #     attention_mask = local_summary_att_masks,
        # )
        
        # Use the state of the last hidden layer as a representation of the text，[batchsize_local, sequence_length, 4096]
        local_text_feat_for_align = ret_local.hidden_states[-1][:,-1]
        # local_text_feat_for_align = ret_local.last_hidden_state[:,-1]
        
        # Embedding local text features
        local_text_embed_for_align = self.local_text_embed(local_text_feat_for_align)
        # local_text_embed_for_align = self.local_text_embed_bert(local_text_feat_for_align)

        # Calculate the loss of local alignment
        local_loss = self.local_alignment(local_image_embed_for_align, local_text_embed_for_align)

        # Embedding global visual features, originally [batchsize, 24, 2048]
        # [batchsize, 24, 4096]
        global_image_feat = self.global_visual_projector(image_feature_global)
        # [batchsize, 4096]
        global_image_feat_for_align = self.global_avg_pool(global_image_feat.permute(0,2,1)).squeeze(-1)
        # [batchsize, embeding_size]
        global_image_embed_for_align = self.global_visual_embed(global_image_feat_for_align)

        global_summary_ids, global_summary_att_masks = self._prepare_global_alignment_batch(sample_id)
        # global_summary_ids, global_summary_att_masks = self._prepare_global_alignment_batch_bert(sample_id)

        # Calculate the representation of global text features
        ret_global = self.language_model(
            input_ids = global_summary_ids,
            attention_mask = global_summary_att_masks,
            output_hidden_states = True,
        )
        # ret_global = self.bert_encoder(
        #     input_ids = global_summary_ids[:,:512],
        #     attention_mask = global_summary_att_masks,
        # )

        # Using the state of the last hidden layer as a representation of the text, [batchsize_local, 4096]
        global_text_feat_for_align = ret_global.hidden_states[-1][:,-1]
        # global_text_feat_for_align = ret_global.last_hidden_state[:,-1]
        
        # Embedding global text features
        global_text_embed_for_align = self.global_text_embed(global_text_feat_for_align)
        # global_text_embed_for_align = self.global_text_embed_bert(global_text_feat_for_align)

        # Calculate the loss of global alignment
        global_loss = self.global_alignment(global_image_embed_for_align, global_text_embed_for_align)

        #Prepare visual features for the next step
        #Local visual features
        local_image_feature_for_mrg = image_feature_local.reshape(-1,2048,14,14)
        local_image_feature_for_mrg = self.local_res(local_image_feature_for_mrg)
        local_image_feature_for_mrg = self.local_avg_pool(local_image_feature_for_mrg).squeeze(-1).squeeze(-1)
        local_image_feature_for_mrg = self.local_visual_projector(local_image_feature_for_mrg).reshape(-1,24,4096)
        # Global Visual Features
        global_image_feature_for_mrg = global_image_feat

        image_features = torch.cat((local_image_feature_for_mrg,global_image_feature_for_mrg), dim=1) 
        image_features = image_features.reshape(-1,1,4096)

        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else False

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            # if pixel_values is not None and input_ids.shape[1] != 1: # origin
            if image_features is not None and input_ids.shape[1] != 1: # modified by zcx
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.ignore_index).to(torch.long)

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            # elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            elif past_key_values is not None and image_features is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        loss = (loss + 
                self.coef_seg * seg_loss + 
                self.coef_global * global_loss + 
                self.coef_local * local_loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # return LlavaCausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, 
        past_key_values=None, 
        inputs_embeds=None, 
        # pixel_values=None, 
        image_features=None,
        attention_mask=None, 
        **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                # "pixel_values": pixel_values, # origin
                "image_features": image_features,
            }
        )
        return model_inputs

    def generate(
        self,
        max_gen_len: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        sample_id: torch.IntTensor = None,
        input_ids: torch.LongTensor = None,
        # pixel_values: torch.FloatTensor = None,
        image_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # vision_feature_layer: Optional[int] = None,
        # vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True, # 默认返回dict形式
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            input_ids (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """


        sample_id = sample_id.squeeze(1).tolist()
        image_feature_local, image_feature_global = self._prepare_visual_feature_data(sample_id)

        # The local visual features are further processed by a ResNet layer, where the visual features are [4,24,14,14,2048]
        # It needs to be converted into [4,24,2048,14,14] first
        image_feature_local = image_feature_local.permute(0,1,4,2,3)

        # Embedding global visual features, originally [batchsize, 24, 2048]
        # The visual projection is [batchsize, 24, 4096]
        global_image_feat = self.global_visual_projector(image_feature_global)

        local_image_feature_for_mrg = image_feature_local.reshape(-1,2048,14,14)
        local_image_feature_for_mrg = self.local_res(local_image_feature_for_mrg)
        local_image_feature_for_mrg = self.local_avg_pool(local_image_feature_for_mrg).squeeze(-1).squeeze(-1)
        local_image_feature_for_mrg = self.local_visual_projector(local_image_feature_for_mrg).reshape(-1,24,4096)

        global_image_feature_for_mrg = global_image_feat

        image_features = torch.cat((local_image_feature_for_mrg,global_image_feature_for_mrg), dim=1) 
        image_features = image_features.reshape(-1,1,4096)

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            # if pixel_values is not None and input_ids.shape[1] != 1: # origin
            if image_features is not None and input_ids.shape[1] != 1: # modified by zcx
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            # elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            elif past_key_values is not None and image_features is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        bsz = len(inputs_embeds)
        min_prompt_len = min(len(t) for t in inputs_embeds)
        max_prompt_len = max(len(t) for t in inputs_embeds)
        total_len = max_gen_len + max_prompt_len


        pad_id = self.pad_token_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.int, device="cuda")
        embeds = torch.full((bsz, total_len, inputs_embeds.shape[-1]), 0, dtype=torch.float, device="cuda")
        for k, t in enumerate(inputs_embeds):
            embeds[k, : len(t)] = t
        for k, t in enumerate(input_ids):
            tokens[k, : len(t)] = t
        # if logprobs:
        #     token_logprobs = torch.zeros_like(embeds, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id

        # stop_tokens = torch.tensor(list(self.stop_token_id))
        stop_tokens = torch.tensor([self.stop_token_id], device="cuda")

        for cur_pos in range(min_prompt_len, total_len):
            # logits = self.model.forward(embeds[:, prev_pos:cur_pos], prev_pos)
            outputs = self.language_model(
                # attention_mask=attention_mask[:, prev_pos:cur_pos],
                # position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=embeds[:, prev_pos:cur_pos],
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )

            logits = outputs.logits
            past_key_values = outputs.past_key_values
            
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            tokens[:, cur_pos] = next_token
            next_token_embed = self.get_input_embeddings()(next_token)
            embeds[:, cur_pos] = next_token_embed
            # if logprobs:
            #     token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
            #         input=logits.transpose(1, 2),
            #         target=tokens[:, prev_pos + 1 : cur_pos + 1],
            #         reduction="none",
            #         ignore_index=pad_id,
            #     )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        # if logprobs:
        #     token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(input_ids[i])
            toks = toks[start : len(input_ids[i]) + max_gen_len]
            probs = None
            # if logprobs:
            #     probs = token_logprobs[i][start : len(input_ids[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in [self.stop_token_id]:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    # probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            # out_logprobs.append(probs)
        # return (out_tokens, out_logprobs if logprobs else None)
        return out_tokens

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
    
    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        local_params = {name: param for name, param in self.named_parameters() if 'local' in name}
        global_params = {name: param for name, param in self.named_parameters() if 'global' in name}
        # projector_params = {name: param for name, param in self.named_parameters() if 'projector' in name}
        lora_params = {name: param for name, param in self.named_parameters() if 'lora' in name}
        save_params = {**local_params, **global_params, **lora_params}
        save_path = os.path.join(save_directory, self.save_peft_model_name)

        torch.save(save_params, save_path)
        print(f"Model weights saved to {save_path}")
    
    def from_pretrained(self, load_directory):
        load_params = torch.load(os.path.join(load_directory, self.save_peft_model_name))
        self.load_state_dict(load_params, strict=False)


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token