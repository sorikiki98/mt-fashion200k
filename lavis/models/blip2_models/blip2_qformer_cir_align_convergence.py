import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    disabled_train,
)


@registry.register_model("blip2_qformer_cir_align_convergence")
class Blip2QformerCirAlignConvergence(Blip2Base):
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
            num_query_token, self.visual_encoder.num_features, cross_attention_freq=cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        self.bertLM, _ = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, add_cross_attention=False
        )
        self.bertLM.resize_token_embeddings(len(self.tokenizer))

        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.target_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.query_proj = nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size)

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
        logits = 100 * feats1 @ feats2.T  # (B, B)
        labels = torch.arange(feats1.size(0)).long().to(feats1.device)
        loss = F.cross_entropy(logits, labels, reduction="none")
        return loss

    def forward(self, samples):
        n_turns = samples["n_turns"]
        images = samples["images"]  # (6, B, 3, 225, 225)
        cap_input_ids = samples["cap_input_ids"]  # (6, B, 32)
        cap_attention_mask = samples["cap_attention_mask"]  # (6, B, 32)
        mod_input_ids = samples["mod_input_ids"]  # (5, B, 40)
        mod_attention_mask = samples["mod_attention_mask"]  # (5, B, 40)

        cached_query_tokens = self.query_tokens.expand(images[0].size(0), -1, -1)

        mod_attention_mask = [attn.to(self.device) for attn in mod_attention_mask]
        cap_attention_mask = [attn.to(self.device) for attn in cap_attention_mask]

        cap_input_ids = [input_id.to(self.device) for input_id in cap_input_ids]
        mod_input_ids = [input_id.to(self.device) for input_id in mod_input_ids]

        with torch.no_grad():
            images = [img.to(self.device, non_blocking=True) for img in images]
            image_feats = [self.ln_vision(self.visual_encoder(img)).detach() for img in images]
            image_atts_list = [torch.ones(f.size()[:-1], dtype=torch.long).to(f.device) for f in image_feats]

        loss_total = 0
        loss_per_sample = torch.zeros(images[0].size(0), device=self.device)
        query_tokens = None
        for turn_i in range(1, self.max_turn + 1):
            valid_mask = (n_turns >= turn_i)
            if valid_mask.sum() == 0:
                continue

            image_embeds = image_feats[turn_i - 1]
            image_atts = image_atts_list[turn_i - 1]
            if query_tokens is None:
                query_tokens = cached_query_tokens.clone()
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            tar_query_tokens = cached_query_tokens.clone()
            tar_query_atts = torch.ones(tar_query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            attention_mask = torch.cat([query_atts, mod_attention_mask[turn_i - 1][:query_tokens.size(1)]], dim=1)
            ref_cap_attn_mask = torch.cat([query_atts, cap_attention_mask[turn_i - 1][:query_tokens.size(1)]], dim=1)

            fusion_output = self.Qformer.bert(
                cap_input_ids[turn_i - 1][:query_tokens.size(1)],
                query_embeds=query_tokens,
                attention_mask=ref_cap_attn_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            fusion_processed = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
            text_output = self.bertLM.bert(
                mod_input_ids[turn_i - 1][:query_tokens.size(1)],
                query_embeds=fusion_processed,  # [b, 32, 768]
                attention_mask=attention_mask,
                return_dict=True,
            )
            text_processed = text_output.last_hidden_state[:, : query_tokens.size(1), :]
            query_tokens = self.query_proj(fusion_processed + text_processed)
            fusion_feats = F.normalize((self.text_proj(query_tokens[:, -1, :])), dim=-1)

            target_img_embeds = image_feats[turn_i]
            target_img_atts = image_atts_list[turn_i]

            tar_attention_mask = torch.cat([tar_query_atts, cap_attention_mask[turn_i][:query_tokens.size(1)]], dim=1)
            tar_fusion_output = self.Qformer.bert(
                cap_input_ids[turn_i][:query_tokens.size(1)],
                query_embeds=tar_query_tokens,  # Qformer里query embeds和modifier_tokens会拼接
                attention_mask=tar_attention_mask,
                encoder_hidden_states=target_img_embeds,  # cross attention
                encoder_attention_mask=target_img_atts,
                return_dict=True,
            )
            tar_fusion_processed = tar_fusion_output.last_hidden_state[:, : tar_query_tokens.size(1), :]
            tar_fusion_feats = F.normalize((self.target_proj(tar_fusion_processed[:, -1, :])),
                                           dim=-1)

            tar_text_tokens = self.prompt_tokens.expand(image_embeds.shape[0], -1, -1)
            tar_text_output = self.Qformer.bert(
                cap_input_ids[turn_i][:query_tokens.size(1)],
                query_embeds=tar_text_tokens,
                attention_mask=tar_attention_mask,
                return_dict=True,
                no_img=True,
            )
            tar_cap_text_feat = F.normalize(self.text_proj(tar_text_output.last_hidden_state[:, 0, :]), dim=-1)

            mod_text_tokens = self.prompt_tokens.expand(
                image_embeds.shape[0], -1, -1)
            mod_text_output = self.Qformer.bert(
                mod_input_ids[turn_i - 1][:query_tokens.size(1)],
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

    def forward_fusion(self, attention_mask, query_tokens, input_ids=None, image_embeds=None, image_atts=None,
                       no_img=False, learned_embeds=None):
        if no_img:  # mod_text_output
            if learned_embeds is not None:
                output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    attention_mask=attention_mask,
                    return_dict=True,
                    learned_embeds=learned_embeds
                )
                last_hidden_state = output.last_hidden_state
            else:  # tar_text_output
                output = self.Qformer.bert(
                    input_ids,
                    query_embeds=query_tokens,
                    attention_mask=attention_mask,
                    return_dict=True,
                    no_img=no_img
                )
                last_hidden_state = output.last_hidden_state
        else:  # (ref) fusion output
            fusion_output = self.Qformer.bert(
                input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            last_hidden_state = fusion_output.last_hidden_state
        return last_hidden_state

    def forward_text(self, attention_mask, learned_embeds, query_tokens):
        text_output = self.bertLM.bert(
            query_embeds=query_tokens,  # [b, 32, 768]
            attention_mask=attention_mask,  # [b, 64]
            learned_embeds=learned_embeds,  # [b, 31, 768]
            return_dict=True,
        )
        text_processed = text_output.last_hidden_state
        return text_processed

    @torch.no_grad()
    def inference(self, samples):
        target_feats = samples["target_feats"]
        n_turns = samples["n_turns"]
        images = samples["images"]  # (6, B, 3, 225, 225)
        cap_input_ids = samples["cap_input_ids"]  # (6, B, 32)
        cap_attention_mask = samples["cap_attention_mask"]  # (6, B, 32)
        mod_input_ids = samples["mod_input_ids"]  # (5, B, 40)
        mod_attention_mask = samples["mod_attention_mask"]  # (5, B, 40)

        cached_query_tokens = self.query_tokens.expand(images[0].size(0), -1, -1)

        images = [img.to(self.device, non_blocking=True) for img in images]
        image_feats = [self.ln_vision(self.visual_encoder(img)).detach() for img in images]
        image_atts_list = [torch.ones(f.size()[:-1], dtype=torch.long).to(f.device) for f in image_feats]

        mod_attention_mask = [attn.to(self.device) for attn in mod_attention_mask]
        cap_attention_mask = [attn.to(self.device) for attn in cap_attention_mask]

        cap_input_ids = [input_id.to(self.device) for input_id in cap_input_ids]
        mod_input_ids = [input_id.to(self.device) for input_id in mod_input_ids]

        last_fusion_feats_all = torch.zeros(
            images[0].size(0), self.text_proj.out_features, device=self.device
        )
        first_fusion_feats_all = torch.zeros(
            images[0].size(0), self.text_proj.out_features, device=self.device
        )
        second_fusion_feats_all = torch.zeros(
            images[0].size(0), self.text_proj.out_features, device=self.device
        )
        query_tokens = None
        for turn_i in range(1, self.max_turn + 1):
            valid_mask = (n_turns >= turn_i)
            final_mask = (n_turns == turn_i)
            if valid_mask.sum() == 0:
                continue

            image_embeds = image_feats[turn_i - 1]
            image_atts = image_atts_list[turn_i - 1]
            if query_tokens is None:
                query_tokens = cached_query_tokens.clone()

            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            attention_mask = torch.cat([query_atts, mod_attention_mask[turn_i - 1][:, :query_tokens.size(1)]], dim=1)
            ref_cap_attn_mask = torch.cat([query_atts, cap_attention_mask[turn_i - 1][:, :query_tokens.size(1)]], dim=1)
            fusion_output = self.Qformer.bert(
                cap_input_ids[turn_i - 1][:, :query_tokens.size(1)],
                query_embeds=query_tokens,
                attention_mask=ref_cap_attn_mask,
                encoder_hidden_states=image_embeds,  # cross attention
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            fusion_processed = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
            text_output = self.bertLM.bert(
                mod_input_ids[turn_i - 1][:, :query_tokens.size(1)],
                query_embeds=fusion_processed,  # [b, 32, 768]
                attention_mask=attention_mask,
                return_dict=True,
            )
            text_processed = text_output.last_hidden_state[:, : query_tokens.size(1), :]
            query_tokens = self.query_proj(fusion_processed + text_processed)

            if turn_i == 1:
                first_fusion_feats_all = F.normalize(self.text_proj(query_tokens[:, -1, :]), dim=-1)
            elif turn_i == 2:
                second_fusion_feats_all = F.normalize(self.text_proj(query_tokens[:, -1, :]), dim=-1)
            if final_mask.sum() > 0:
                selected_feats = self.text_proj(query_tokens[:, -1, :])[final_mask]
                projected_feats = F.normalize(selected_feats, dim=-1)
                last_fusion_feats_all[final_mask] = projected_feats

        first_sim_matrix = self.compute_distance_matrix(first_fusion_feats_all.to(self.device), target_feats)
        second_sim_matrix = self.compute_distance_matrix(second_fusion_feats_all.to(self.device), target_feats)
        last_sim_matrix = self.compute_distance_matrix(last_fusion_feats_all.to(self.device), target_feats)
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

        tar_cap_attn_mask = torch.cat([tar_query_atts, cap_attention_mask[:, :tar_query_tokens.size(1)]], dim=1)

        tar_fusion_last_hidden = self.forward_fusion(
            input_ids=cap_input_ids[:, :tar_query_tokens.size(1)],
            query_tokens=tar_query_tokens,
            attention_mask=tar_cap_attn_mask,
            image_embeds=image_embeds_frozen,  # cross attention
            image_atts=image_atts
        )
        tar_fusion_processed = tar_fusion_last_hidden[:, : tar_query_tokens.size(1), :]

        tar_fusion_feats = F.normalize((self.target_proj(tar_fusion_processed[:, -1, :])),
                                       dim=-1)
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
