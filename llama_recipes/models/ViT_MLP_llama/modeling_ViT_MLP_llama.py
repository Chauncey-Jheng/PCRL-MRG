from typing import List, Optional, Tuple, Union
from ..llama.modeling_llama import LlamaForCausalLM
from ..llama.modeling_llama import LlamaDecoderLayer
import torch
from torch import nn
import numpy as np

from peft import get_peft_model, prepare_model_for_kbit_training
from ...utils.config_utils import(
    generate_peft_config,
)

from transformers.cache_utils import Cache

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

import os

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device("cpu")

class ViTLlamaModel(nn.Module):
    '''
    A simplified vision-language model, inspired by the LLAVA prototype,
    takes 1024-dimensional visual features as input (these visual features are pre-extracted using CLIP-Large-patch14 and stored in files).
    An MLP is used as a projector to map these features into a 4096-dimensional space, serving as special visual tokens.
    The language model utilized is LLaMA 3, quantized with INT4 precision.
    '''

    def __init__(self, train_config, use_cache, tokenizer, image_token_index, kwargs, wandb_run):
        super().__init__()
        
        # self.tokenizer = tokenizer

        self.image_token_index = image_token_index
        self.ignore_index = -100
        self.stop_token_id = 128001
        self.pad_token_id = self.stop_token_id # or -1
        self.context_length = train_config.context_length
        self.train_config = train_config

        self.multi_modal_projector = nn.Sequential(nn.Linear(1024, 4096), nn.Linear(4096,4096))

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

    def _get_image_features_by_id(self, sample_id):
        '''
        根据sample id 拿去预先处理好的视觉数据
        返回的batch_image_feature_local的size为[24,257,1024]
        batch_image_feature_global的size为[24,1024]
        '''
        if not self.train_config.visual_features_dir:
            raise ValueError("ViTLlamaModel requires --visual_features_dir to point to the pre-extracted ViT feature directory.")
        batch_image_feature_local = []
        batch_image_feature_global = []
        for i in range(len(sample_id)):
            id = int(sample_id[i])
            # 根据id，获取npy和npz格式的视觉特征数据
            image_feature_global_path = os.path.join(self.train_config.visual_features_dir, "pooler_output", str(id) + ".npz")
            image_feature_local_path = os.path.join(self.train_config.visual_features_dir, "last_hidden_state", str(id) + ".npz")
            # device = torch.device('cuda:0')

            npz_file = np.load(image_feature_global_path)
            arrays = {key: npz_file[key] for key in npz_file.files}
            tensors = {key: torch.tensor(value).to(device) for key, value in arrays.items()}
            image_feature_global = tensors["pooler_output"]
            batch_image_feature_global.append(image_feature_global)

            npz_file = np.load(image_feature_local_path)
            arrays = {key: npz_file[key] for key in npz_file.files}
            tensors = {key: torch.tensor(value).to(device) for key, value in arrays.items()}
            image_feature_local = tensors["last_hidden_state"]
            batch_image_feature_local.append(image_feature_local)

        batch_image_feature_local = torch.stack(batch_image_feature_local, dim=0).to(device)
        batch_image_feature_global = torch.stack(batch_image_feature_global, dim=0).to(device)
        return batch_image_feature_local, batch_image_feature_global


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
        sample_id: torch.IntTensor = None, # CT Scans and report pair sample ID
        quiz_id: torch.IntTensor = None, # VQA quiz ID
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # vision_feature_layer: Optional[int] = None,
        # vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True, # 默认返回dict形式
    ):


        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else False
        # vision_feature_layer = (
        #     vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        # )
        # vision_feature_select_strategy = (
        #     vision_feature_select_strategy
        #     if vision_feature_select_strategy is not None
        #     else self.config.vision_feature_select_strategy
        # )

        # image_features shape must be (num_images, num_image_patches, embed_dim)
        image_feature_local, image_feature_global = self._get_image_features_by_id(sample_id)
        image_features = image_feature_global

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            # if pixel_values is not None and input_ids.shape[1] != 1: # origin
            if image_features is not None and input_ids.shape[1] != 1: # modified by zcx
                # image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                # selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                # if vision_feature_select_strategy == "default":
                #     selected_image_feature = selected_image_feature[:, 1:]
                # elif vision_feature_select_strategy == "full":
                #     selected_image_feature = selected_image_feature
                # else:
                #     raise ValueError(
                #         f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                #     )
                selected_image_feature = image_features.reshape((-1,1,1024))  
                image_features = self.multi_modal_projector(selected_image_feature)
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
        sample_id: torch.IntTensor = None, # just use to receive the sample id if have one
        quiz_id: torch.IntTensor = None, # just use to receive the quiz id if have one
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
        # image_features shape must be (num_images, num_image_patches, embed_dim)
        image_feature_local, image_feature_global = self._get_image_features_by_id(sample_id)
        image_features = image_feature_global
        
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            # if pixel_values is not None and input_ids.shape[1] != 1: # origin
            if image_features is not None and input_ids.shape[1] != 1: # modified by zcx

                selected_image_feature = image_features.reshape((-1,1,1024))  
                image_features = self.multi_modal_projector(selected_image_feature)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, None
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
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.int, device=device)
        embeds = torch.full((bsz, total_len, inputs_embeds.shape[-1]), 0, dtype=torch.float, device=device)
        for k, t in enumerate(inputs_embeds):
            embeds[k, : len(t)] = t
        for k, t in enumerate(input_ids):
            tokens[k, : len(t)] = t
        # if logprobs:
        #     token_logprobs = torch.zeros_like(embeds, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id

        # stop_tokens = torch.tensor(list(self.stop_token_id))
        stop_tokens = torch.tensor([self.stop_token_id], device=device)

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
    
    def save_pretrained(self, save_directory, peft_model_name="peft_model_lora_adapter", epoch=None):
        # No need extension name
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        projector_params = {name: param for name, param in self.named_parameters() if 'projector' in name}
        lora_params = {name: param for name, param in self.named_parameters() if 'lora' in name}
        save_params = {**projector_params, **lora_params}
        if epoch == None:
            save_path = os.path.join(save_directory, peft_model_name + ".pth")
        else:
            save_path = os.path.join(save_directory, peft_model_name + "_epoch" + str(epoch) + ".pth")

        torch.save(save_params, save_path)
        print(f"Model weights saved to {save_path}")
    
    def from_pretrained(self, load_directory, peft_model_name="peft_model_lora_adapter", epoch=None):
        if epoch == None:
            load_path = os.path.join(load_directory, peft_model_name + ".pth")
        else:
            load_path = os.path.join(load_directory, peft_model_name + "_epoch" + str(epoch) + ".pth")
    
        load_params = torch.load(load_path)
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
