import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures


@registry.register_model("blip2_qformer_cir_align_prompt")
class Blip2QformerCirAlignPrompt(Blip2Base):
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
            num_query_token=40,  # 32
            cross_attention_freq=2,
            embed_dim=256,
            max_txt_len=32,
            max_turn=5,
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
        self.target_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len
        self.token_alpha = nn.Parameter(1.0 * torch.ones([]))
        self.num_query_token = num_query_token

        # new tokens
        self.prompt_tokens = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
        self.visual_tokens = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
        self.visual_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)

        self.max_turn = max_turn

    def BBC_loss(self, feats1, feats2):
        prediction = 100 * feats1 @ feats2.T
        labels = torch.arange(0, feats1.size(0)).long().to(feats1.device)
        return F.cross_entropy(prediction, labels)

    def token_contrast_loss(self, token_feats, visual_token_feats):
        sim_f2t = torch.matmul(token_feats.unsqueeze(1).unsqueeze(1), visual_token_feats.permute(0, 2, 1)).squeeze()
        sim_matrix, _ = sim_f2t.max(-1)
        sim_matrix = sim_matrix / self.temp
        bs = token_feats.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(token_feats.device)
        return F.cross_entropy(sim_matrix, targets)

    def forward(self, samples):
        n_turns = samples["n_turns"]  # (B,)
        images = samples["images"]  # (6, B, 3, 224, 224)
        cap_input_ids = samples["cap_input_ids"]  # (6, B, 20)
        cap_attention_mask = samples["cap_attention_mask"]  # (6, B, 20)
        mod_input_ids = samples["mod_input_ids"]  # (5, B, 40)
        mod_attention_mask = samples["mod_attention_mask"]  # (5, B, 40)

        is_rollback = samples["is_rollback"]  # (B,)
        is_combination = samples["is_combination"]  # (B,)

        rollback_input_ids = samples["rollback_input_ids"].to(self.device)  # (B, 20)
        rollback_attention_mask = samples["rollback_attention_mask"].to(self.device)  # (B, 20)
        rollback_images = samples["rollback_images"]  # (B, 3, 224, 224)
        combination_input_ids = samples["combination_input_ids"].to(self.device)  # (B, 20)
        combination_attention_mask = samples["combination_attention_mask"].to(self.device)  # (B, 20)

        cached_query_tokens = self.query_tokens.expand(images[0].size(0), -1, -1)

        mod_attention_mask = [attn.to(self.device) for attn in mod_attention_mask]
        cap_attention_mask = [attn.to(self.device) for attn in cap_attention_mask]

        cap_input_ids = [input_id.to(self.device) for input_id in cap_input_ids]
        mod_input_ids = [input_id.to(self.device) for input_id in mod_input_ids]

        with torch.no_grad():
            images = [img.to(self.device, non_blocking=True) for img in images]
            image_feats = [self.ln_vision(self.visual_encoder(img)).detach() for img in images]
            image_atts_list = [torch.ones(f.size()[:-1], dtype=torch.long).to(self.device) for f in image_feats]
            rollback_image = rollback_images.to(self.device, non_blocking=True)
            rollback_image_embeds = self.ln_vision(self.visual_encoder(rollback_image)).detach()
            rollback_image_atts = torch.ones(rollback_image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        loss_total = 0
        loss_per_sample = torch.zeros(images[0].size(0), device=self.device)
        query_tokens = None
        for turn_i in range(1, self.max_turn + 1):
            valid_mask = (n_turns >= turn_i)
            if valid_mask.sum() == 0:
                continue

            rollback_mask_bool = (n_turns == turn_i) & is_rollback
            combination_mask_bool = (n_turns == turn_i) & is_combination

            image_embeds = image_feats[turn_i - 1]
            image_atts = image_atts_list[turn_i - 1]

            if query_tokens is None:
                query_tokens = cached_query_tokens.clone()
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            tar_query_tokens = cached_query_tokens.clone()
            tar_query_atts = torch.ones(tar_query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            attention_mask = torch.cat([query_atts, mod_attention_mask[turn_i - 1]], dim=1)
            ref_cap_attn_mask = torch.cat([query_atts, cap_attention_mask[turn_i - 1]], dim=1)

            base_input_ids = cap_input_ids[turn_i - 1].clone().to(self.device)
            base_attn_mask = ref_cap_attn_mask.clone().to(self.device)

            rollback_mask = torch.cat([query_atts, rollback_attention_mask], dim=1)
            combination_mask = torch.cat([query_atts, combination_attention_mask], dim=1)

            rollback_mask_broadcast = rollback_mask_bool.view(-1, 1).to(self.device)
            combination_mask_broadcast = combination_mask_bool.view(-1, 1)  .to(self.device)

            # rollback
            current_input_ids = torch.where(rollback_mask_broadcast, rollback_input_ids, base_input_ids)
            current_attn_mask = torch.where(rollback_mask_broadcast, rollback_mask, base_attn_mask)

            # combination
            current_input_ids = torch.where(combination_mask_broadcast, combination_input_ids, current_input_ids)
            current_attn_mask = torch.where(combination_mask_broadcast, combination_mask, current_attn_mask)

            is_comb = combination_mask_bool
            is_not_comb = ~is_comb
            is_roll = rollback_mask_bool
            is_not_roll = ~is_roll
            is_conv = is_not_comb & is_not_roll

            assert torch.all((is_comb.int() + is_roll.int() + is_conv.int()) == 1)

            batch_size = current_input_ids.size(0)
            len_token = query_tokens.size(1)
            hidden_size = query_tokens.size(-1)

            fusion_output_hidden = torch.zeros(batch_size, len_token, hidden_size, device=self.device)

            if is_conv.any():
                out = self.Qformer(
                    current_input_ids[is_conv],
                    query_embeds=query_tokens[is_conv],
                    attention_mask=current_attn_mask[is_conv],
                    encoder_hidden_states=image_embeds[is_conv],
                    encoder_attention_mask=image_atts[is_conv],
                    return_dict=True,
                )
                fusion_output_hidden[is_conv] = out.last_hidden_state[:, :query_tokens.size(1), :]
            if is_comb.any():
                out = self.Qformer(
                    current_input_ids[is_comb],
                    query_embeds=query_tokens[is_comb],
                    attention_mask=current_attn_mask[is_comb],
                    return_dict=True,
                )
                fusion_output_hidden[is_comb] = out.last_hidden_state[:, :query_tokens.size(1), :]
            if is_roll.any():
                out = self.Qformer(
                    current_input_ids[is_roll],
                    query_embeds=query_tokens[is_roll],
                    attention_mask=current_attn_mask[is_roll],
                    encoder_hidden_states=rollback_image_embeds[is_roll],
                    encoder_attention_mask=rollback_image_atts[is_roll],
                    return_dict=True,
                )
                fusion_output_hidden[is_roll] = out.last_hidden_state[:, :query_tokens.size(1), :]
            fusion_output = type(out)(last_hidden_state=fusion_output_hidden)

            text_output = self.Qformer(
                mod_input_ids[turn_i - 1],
                query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],  # [b, 40, 768]
                attention_mask=attention_mask,
                return_dict=True,
            )

            fusion_feats = F.normalize(self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1)  # [b, 256]
            query_tokens = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]

            target_img_embeds = image_feats[turn_i]
            target_img_atts = image_atts_list[turn_i]

            tar_attention_mask = torch.cat([tar_query_atts, cap_attention_mask[turn_i]],
                                           dim=1)
            tar_fusion_output = self.Qformer(
                cap_input_ids[turn_i],
                query_embeds=tar_query_tokens,  # Qformer里query embeds和modifier_tokens会拼接
                attention_mask=tar_attention_mask,
                encoder_hidden_states=target_img_embeds,  # cross attention
                encoder_attention_mask=target_img_atts,
                return_dict=True,
            )
            tar_fusion_feats = F.normalize(self.target_proj(tar_fusion_output.last_hidden_state[:, 32, :]), dim=-1)

            tar_text_tokens = self.prompt_tokens.expand(image_embeds.shape[0], -1, -1)
            tar_text_output = self.Qformer(
                cap_input_ids[turn_i],
                query_embeds=tar_text_tokens,
                attention_mask=tar_attention_mask,
                return_dict=True,
                no_img=True,
            )
            tar_cap_text_feat = F.normalize(self.text_proj(tar_text_output.last_hidden_state[:, 0, :]), dim=-1)

            mod_text_tokens = self.prompt_tokens.expand(
                image_embeds.shape[0], -1, -1)
            mod_text_output = self.Qformer(
                mod_input_ids[turn_i - 1],
                query_embeds=mod_text_tokens,
                attention_mask=attention_mask,
                return_dict=True,
                no_img=True,
            )
            mod_cap_feat = F.normalize(self.text_proj(mod_text_output.last_hidden_state[:, 0, :]), dim=-1)

            loss_dict = {
                "loss_fus2tar": self.BBC_loss(fusion_feats.to(self.device), tar_fusion_feats.to(self.device)),
                'loss_fus2cap': self.BBC_loss(fusion_feats.to(self.device), tar_cap_text_feat.to(self.device)),
                'loss_mod2fus': self.BBC_loss(mod_cap_feat.to(self.device), tar_fusion_feats.to(self.device)),
                'loss_mod2cap': self.BBC_loss(mod_cap_feat.to(self.device), tar_cap_text_feat.to(self.device))
            }
            loss_per_turn = sum(loss_dict.values())
            loss_per_sample += loss_per_turn
            mask = (n_turns == turn_i)
            filtered_loss = loss_per_sample[mask]
            loss_total += filtered_loss.sum() / turn_i

        return loss_total / images[0].size(0)

    @torch.no_grad()
    def inference(self, samples):
        target_feats = samples["target_feats"]
        n_turns = samples["n_turns"]  # (B,)
        images = samples["images"]  # (6, B, 3, 224, 224)
        cap_input_ids = samples["cap_input_ids"]  # (6, B, 20)
        cap_attention_mask = samples["cap_attention_mask"]  # (6, B, 20)
        mod_input_ids = samples["mod_input_ids"]  # (5, B, 40)
        mod_attention_mask = samples["mod_attention_mask"]  # (5, B, 40)
        batch_size = images[0].size(0)

        is_rollback = samples["is_rollback"]  # (B,)
        is_combination = samples["is_combination"]  # (B,)

        rollback_input_ids = samples["rollback_input_ids"]  # (B, 20)
        rollback_attention_mask = samples["rollback_attention_mask"]  # (B, 20)
        rollback_images = samples["rollback_images"]  # (B, 3, 224, 224)
        combination_input_ids = samples["combination_input_ids"]  # (B, 20)
        combination_attention_mask = samples["combination_attention_mask"]  # (B, 20)

        cached_query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        mod_attention_mask = [attn.to(self.device) for attn in mod_attention_mask]
        cap_attention_mask = [attn.to(self.device) for attn in cap_attention_mask]

        cap_input_ids = [input_id.to(self.device) for input_id in cap_input_ids]
        mod_input_ids = [input_id.to(self.device) for input_id in mod_input_ids]

        images = [img.to(self.device, non_blocking=True) for img in images]
        image_feats = [self.ln_vision(self.visual_encoder(img)).detach() for img in images]
        image_atts_list = [torch.ones(f.size()[:-1], dtype=torch.long).to(f.device) for f in image_feats]

        rollback_image = rollback_images.to(self.device, non_blocking=True)
        rollback_image_embeds = self.ln_vision(self.visual_encoder(rollback_image)).detach()
        rollback_image_atts = torch.ones(rollback_image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        last_fusion_feats_all = torch.zeros(
            batch_size, self.text_proj.out_features, device=self.device
        )
        first_fusion_feats_all = torch.zeros(
            batch_size, self.text_proj.out_features, device=self.device
        )
        second_fusion_feats_all = torch.zeros(
            batch_size, self.text_proj.out_features, device=self.device
        )
        query_tokens = None
        for turn_i in range(1, self.max_turn + 1):
            valid_mask = (n_turns >= turn_i)
            final_mask = (n_turns == turn_i)
            if valid_mask.sum() == 0:
                continue

            rollback_mask_bool = (n_turns == turn_i) & is_rollback
            combination_mask_bool = (n_turns == turn_i) & is_combination

            image_embeds = image_feats[turn_i - 1]
            image_atts = image_atts_list[turn_i - 1]
            if query_tokens is None:
                query_tokens = cached_query_tokens.clone()

            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            attention_mask = torch.cat([query_atts, mod_attention_mask[turn_i - 1]], dim=1)
            ref_cap_attn_mask = torch.cat([query_atts, cap_attention_mask[turn_i - 1]], dim=1)

            base_input_ids = cap_input_ids[turn_i - 1].clone()
            base_attn_mask = ref_cap_attn_mask.clone()

            rollback_input = rollback_input_ids
            rollback_mask = rollback_attention_mask

            combination_input = combination_input_ids
            combination_mask = combination_attention_mask

            rollback_mask_broadcast = rollback_mask_bool.unsqueeze(1).expand_as(base_input_ids)
            combination_mask_broadcast = combination_mask_bool.unsqueeze(1).expand_as(base_input_ids)

            # rollback
            current_input_ids = torch.where(rollback_mask_broadcast, rollback_input, base_input_ids)
            current_attn_mask = torch.where(rollback_mask_broadcast, rollback_mask, base_attn_mask)

            # combination
            current_input_ids = torch.where(combination_mask_broadcast, combination_input, current_input_ids)
            current_attn_mask = torch.where(combination_mask_broadcast, combination_mask, current_attn_mask)

            is_comb = combination_mask_bool
            is_not_comb = ~is_comb
            is_roll = rollback_mask_bool
            is_not_roll = ~is_roll
            is_conv = is_not_comb & is_not_roll

            assert torch.all((is_comb.int() + is_roll.int() + is_conv.int()) == 1)

            batch_size = current_input_ids.size(0)
            len_token = query_tokens.size(1)
            hidden_size = query_tokens.size(-1)

            fusion_output_hidden = torch.zeros(batch_size, len_token, hidden_size, device=self.device)

            if is_conv.any():
                out = self.Qformer.bert(
                    current_input_ids[is_conv],
                    query_embeds=query_tokens[is_conv],
                    attention_mask=current_attn_mask[is_conv],
                    encoder_hidden_states=image_embeds[is_conv],
                    encoder_attention_mask=image_atts[is_conv],
                    return_dict=True,
                )
                fusion_output_hidden[is_conv] = out.last_hidden_state[:, :query_tokens.size(1), :]
            if is_comb.any():
                out = self.Qformer.bert(
                    current_input_ids[is_comb],
                    query_embeds=query_tokens[is_comb],
                    attention_mask=current_attn_mask[is_comb],
                    return_dict=True,
                )
                fusion_output_hidden[is_comb] = out.last_hidden_state[:, :query_tokens.size(1), :]
            if is_roll.any():
                out = self.Qformer.bert(
                    current_input_ids[is_roll],
                    query_embeds=query_tokens[is_roll],
                    attention_mask=current_attn_mask[is_roll],
                    encoder_hidden_states=rollback_image_embeds[is_roll],
                    encoder_attention_mask=rollback_image_atts[is_roll],
                    return_dict=True,
                )
                fusion_output_hidden[is_roll] = out.last_hidden_state[:, :query_tokens.size(1), :]
            fusion_output = type(out)(last_hidden_state=fusion_output_hidden)

            text_output = self.Qformer.bert(
                mod_input_ids[turn_i - 1][:, :query_tokens.size(1)],
                query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],  # [b, 32, 768]
                attention_mask=attention_mask,
                return_dict=True,
            )

            query_tokens = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]

            if turn_i == 1:
                first_fusion_feats_all = F.normalize(self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1)
            elif turn_i == 2:
                second_fusion_feats_all = F.normalize(self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1)
            if final_mask.sum() > 0:
                selected_feats = text_output.last_hidden_state[:, 32, :][final_mask]
                projected_feats = F.normalize(self.text_proj(selected_feats), dim=-1)
                last_fusion_feats_all[final_mask] = projected_feats
        first_sim_matrix = self.compute_distance_matrix(first_fusion_feats_all, target_feats)
        second_sim_matrix = self.compute_distance_matrix(second_fusion_feats_all, target_feats)
        last_sim_matrix = self.compute_distance_matrix(last_fusion_feats_all, target_feats)
        return first_sim_matrix, second_sim_matrix, last_sim_matrix

    @torch.no_grad()
    def extract_target_features(self, images, cap_input_ids, cap_attention_mask):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(images))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)

        cap_input_ids = cap_input_ids.to(self.device)
        cap_attention_mask = cap_attention_mask.to(self.device)  # (256, 32)

        tar_query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)
        tar_query_atts = torch.ones(tar_query_tokens.size()[:-1], dtype=torch.long).to(self.device)
        tar_cap_attn_mask = torch.cat([tar_query_atts, cap_attention_mask], dim=1)
        tar_fusion_output = self.Qformer.bert(
            cap_input_ids,
            query_embeds=tar_query_tokens,
            attention_mask=tar_cap_attn_mask,
            encoder_hidden_states=image_embeds_frozen,  # cross attention
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        tar_fusion_feats = F.normalize(self.target_proj(tar_fusion_output.last_hidden_state[:, 32, :]), dim=-1)
        return tar_fusion_feats

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
