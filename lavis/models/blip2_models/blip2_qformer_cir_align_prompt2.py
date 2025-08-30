"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from ntm.ntm import CM_NTM


@registry.register_model("blip2_cir_align_prompt2")
class Blip2QformerCirAlignPrompt2(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            cross_attention_freq=2,
            embed_dim=256,
            max_turn=5,
            max_txt_len=32,
            num_ntms=8,
            vector_length=16,
            hidden_size=200,
            memory_size=(256, 32)
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.ntm_proj = nn.Linear(embed_dim, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        # new tokens
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
        self.max_turn = max_turn

        self.cm_ntm = CM_NTM(num_ntms, embed_dim, vector_length, hidden_size, memory_size)

    def forward(self, samples):
        device = self.device

        n_turns = samples["n_turns"]
        images = samples["images"]  # (6, B, 3, 224, 224)
        mod_input_ids = samples["mod_input_ids"]  # (5, B, 40)
        mod_attention_mask = samples["mod_attention_mask"]  # (5, B, 40)
        batch_size = images[0].size(0)

        with torch.no_grad():
            images = [img.to(device, non_blocking=True) for img in images]
            image_feats = [self.ln_vision(self.visual_encoder(img)).detach() for img in images]
            image_atts_list = [torch.ones(f.size()[:-1], dtype=torch.long).to(f.device) for f in image_feats]

        mod_attention_mask = [attn.to(device) for attn in mod_attention_mask]
        mod_input_ids = [input_id.to(device) for input_id in mod_input_ids]

        cached_query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        loss_total = 0
        loss_per_sample = torch.zeros(images[0].size(0), device=device)
        query_tokens = None

        states, read_vectors = self.cm_ntm.get_initial_state(batch_size)

        for turn_i in range(1, self.max_turn + 1):
            valid_mask = (n_turns >= turn_i)
            if valid_mask.sum() == 0:
                continue

            image_embeds = image_feats[turn_i - 1]
            image_atts = image_atts_list[turn_i - 1]

            if query_tokens is None:
                query_tokens = cached_query_tokens.clone()
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)

            attention_mask = torch.cat([query_atts, mod_attention_mask[turn_i - 1]], dim=1)

            fusion_output = self.Qformer.bert(
                mod_input_ids[turn_i - 1],
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            text_output = self.Qformer.bert(
                mod_input_ids[turn_i - 1],
                query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
                attention_mask=attention_mask,
                return_dict=True,
            )
            ntm_inputs = self.text_proj(text_output.last_hidden_state[:, 32, :])
            ntm_inputs = [ntm_inputs for _ in range(self.cm_ntm.num_ntms)]
            ntm_outputs, (new_states, new_read_vectors) = self.cm_ntm(ntm_inputs, (states, read_vectors))
            states = new_states
            read_vectors = new_read_vectors

            fusion_feats = F.normalize(self.ntm_proj(ntm_outputs[-1]), dim=-1)

            target_img_embeds = image_feats[turn_i]
            target_img_atts = image_atts_list[turn_i]

            target_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=target_img_embeds,
                encoder_attention_mask=target_img_atts,
                use_cache=True,
                return_dict=True,
            )
            target_feats = F.normalize(
                self.vision_proj(target_output.last_hidden_state), dim=-1
            )

            sim_t2q = torch.matmul(
                fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
            ).squeeze()

            sim_i2t, _ = sim_t2q.max(-1)
            sim_i2t = sim_i2t / self.temp
            bs = batch_size
            targets = torch.linspace(0, bs - 1, bs, dtype=int).to(device)
            loss_itc = F.cross_entropy(sim_i2t, targets)

            prompt_tokens = self.prompt_tokens.expand(batch_size, -1, -1)

            text_only_output = self.Qformer.bert(
                mod_input_ids[turn_i - 1],
                query_embeds=prompt_tokens,
                attention_mask=attention_mask,
                return_dict=True,
                no_img=True
            )

            text_only_feat = F.normalize(
                self.text_proj(text_only_output.last_hidden_state[:, 0, :]), dim=-1
            )

            sim_r2t = torch.matmul(
                text_only_feat.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
            ).squeeze()

            sim_r2t, _ = sim_r2t.max(-1)
            sim_r2t = sim_r2t / self.temp
            loss_rtc = F.cross_entropy(sim_r2t, targets)

            loss_align = F.mse_loss(fusion_output.last_hidden_state[:, : query_tokens.size(1), :].mean(1),
                                    prompt_tokens.clone().detach().mean(1))

            loss_dict = {
                'loss_itc': loss_itc,
                'loss_rtc': loss_rtc,
                'loss_align': loss_align
            }

            loss_per_turn = sum(loss_dict.values())
            loss_per_sample += loss_per_turn
            mask = (n_turns == turn_i)
            filtered_loss = loss_per_sample[mask]
            loss_total += filtered_loss.sum() / turn_i

        return loss_total / batch_size

    @torch.no_grad()
    def inference(self, samples):
        device = self.device
        target_feats = samples["target_feats"]
        n_turns = samples["n_turns"]  # (B,)
        images = samples["images"]  # (6, B, 3, 224, 224)
        mod_input_ids = samples["mod_input_ids"]  # (5, B, 40)
        mod_attention_mask = samples["mod_attention_mask"]  # (5, B, 40)
        batch_size = images[0].size(0)

        cached_query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        images = [img.to(device, non_blocking=True) for img in images]
        image_feats = [self.ln_vision(self.visual_encoder(img)).detach() for img in images]
        image_atts_list = [torch.ones(f.size()[:-1], dtype=torch.long).to(f.device) for f in image_feats]

        last_fusion_feats_all = torch.zeros(
            batch_size, self.text_proj.out_features, device=device
        )
        first_fusion_feats_all = torch.zeros(
            batch_size, self.text_proj.out_features, device=device
        )
        second_fusion_feats_all = torch.zeros(
            batch_size, self.text_proj.out_features, device=device
        )
        query_tokens = None

        states, read_vectors = self.cm_ntm.get_initial_state(batch_size)

        for turn_i in range(1, self.max_turn + 1):
            valid_mask = (n_turns >= turn_i)
            final_mask = (n_turns == turn_i)
            if valid_mask.sum() == 0:
                continue

            image_embeds = image_feats[turn_i - 1]
            image_atts = image_atts_list[turn_i - 1]
            if query_tokens is None:
                query_tokens = cached_query_tokens.clone()

            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
            attention_mask = torch.cat([query_atts, mod_attention_mask[turn_i - 1]], dim=1)

            fusion_output = self.Qformer.bert(
                mod_input_ids[turn_i - 1],
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            text_output = self.Qformer.bert(
                mod_input_ids[turn_i - 1],
                query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
                attention_mask=attention_mask,
                return_dict=True,
            )

            ntm_inputs = self.text_proj(text_output.last_hidden_state[:, 32, :])
            ntm_inputs = [ntm_inputs for _ in range(self.cm_ntm.num_ntms)]
            ntm_outputs, (new_states, new_read_vectors) = self.cm_ntm(ntm_inputs, (states, read_vectors))

            states = new_states
            read_vectors = new_read_vectors

            if turn_i == 1:
                first_fusion_feats_all = F.normalize(self.ntm_proj(ntm_outputs[-1]), dim=-1)
            elif turn_i == 2:
                second_fusion_feats_all = F.normalize(self.ntm_proj(ntm_outputs[-1]), dim=-1)
            if final_mask.sum() > 0:
                selected_outputs = ntm_outputs[-1][final_mask]
                projected_feats = F.normalize(self.ntm_proj(selected_outputs), dim=-1)
                last_fusion_feats_all[final_mask] = projected_feats

        first_sim_matrix = self.compute_distance_matrix(first_fusion_feats_all, target_feats)
        second_sim_matrix = self.compute_distance_matrix(second_fusion_feats_all, target_feats)
        last_sim_matrix = self.compute_distance_matrix(last_fusion_feats_all, target_feats)
        return first_sim_matrix, second_sim_matrix, last_sim_matrix

    @torch.no_grad()
    def extract_target_features(self, image):
        device = self.device

        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(device)
        query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features

    def compute_distance_matrix(self, fusion_feats, target_feats, chunk_size=1024 * 4):
        device = fusion_feats.device  # cuda
        num_fusion = fusion_feats.size(0)
        num_target = target_feats.size(0)
        num_chunks = (num_target + chunk_size - 1) // chunk_size
        distance_matrix = torch.zeros(num_fusion, num_target).cpu()
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_target)
            chunk_target_feats = target_feats[start:end, :].to(device)
            chunk_distance = torch.matmul(fusion_feats, chunk_target_feats.T).detach().cpu()
            distance_matrix[:, start:end] = chunk_distance
        return distance_matrix

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
