import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    disabled_train,
)

from gating import TurnGatingModule


@registry.register_model("blip2_qformer_cir_align_retrospective")
class Blip2QformerCirAlignRetrospective(Blip2Base):
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
            num_mod_token=40,
            cross_attention_freq=2,
            embed_dim=256,
            max_turn=5,
            max_txt_len=32,
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

        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.bertLM, _ = self.init_Qformer(
            num_mod_token, self.visual_encoder.num_features, add_cross_attention=False
        )
        self.bertLM.resize_token_embeddings(len(self.tokenizer))

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.target_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.query_proj = nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size)
        self.num_query_token = num_query_token

        self.max_txt_len = max_txt_len

        # new tokens
        self.prompt_tokens = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
        self.max_turn = max_turn

        self.gating = TurnGatingModule(vocab_size=len(self.tokenizer), max_turns=self.max_turn)
        self.gating_loss_weight = nn.Parameter(torch.tensor(1.0))

    def BBC_loss(self, feats1, feats2):
        logits = 100 * feats1 @ feats2.T  # (B, B)
        labels = torch.arange(feats1.size(0)).long().to(feats1.device)
        loss = F.cross_entropy(logits, labels, reduction="none")
        return loss

    def forward(self, samples):
        n_turns = samples["n_turns"]
        images = samples["images"]  # (6, B, 3, 224, 224)
        cap_input_ids = samples["cap_input_ids"]  # (6, B, 8)
        cap_attention_mask = samples["cap_attention_mask"]  # (6, B, 8)
        mod_input_ids = samples["mod_input_ids"]  # (5, B, 40)
        mod_attention_mask = samples["mod_attention_mask"]  # (5, B, 40)
        batch_size = images[0].size(0)

        probs = list(zip(*samples["probs"]))
        probs = torch.tensor([list(row) for row in probs]).float().to(self.device)

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

        mod_attention_mask = [attn.to(self.device) for attn in mod_attention_mask]
        cap_attention_mask = [attn.to(self.device) for attn in cap_attention_mask]

        cap_input_ids = [input_id.to(self.device) for input_id in cap_input_ids]
        mod_input_ids = [input_id.to(self.device) for input_id in mod_input_ids]

        with torch.no_grad():
            images = [img.to(self.device, non_blocking=True) for img in images]
            image_feats = [self.ln_vision(self.visual_encoder(img)).detach() for img in images]
            image_atts_list = [torch.ones(f.size()[:-1], dtype=torch.long).to(f.device) for f in image_feats]

        loss_total = 0
        loss_per_sample = torch.zeros(batch_size, device=self.device)
        gating_loss_total = 0

        ref_image_embeds = image_feats[0]
        ref_image_atts = image_atts_list[0]
        ref_cap_input_ids = cap_input_ids[0]
        ref_cap_attn_mask = torch.cat([query_atts, cap_attention_mask[0]], dim=1)

        ref_fusion_output = self.Qformer(
            ref_cap_input_ids,
            query_embeds=query_tokens.clone(),
            attention_mask=ref_cap_attn_mask,
            encoder_hidden_states=ref_image_embeds,
            encoder_attention_mask=ref_image_atts,
            return_dict=True,
        )
        ref_fusion_processed = ref_fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
        tar_query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        for turn_i in range(1, self.max_turn + 1):
            target_gates = torch.zeros(batch_size, 5, device=self.device)
            valid_mask = (n_turns >= turn_i)
            if valid_mask.sum() == 0:
                continue
            final_mask = (n_turns == turn_i)
            target_gates[final_mask] = probs[final_mask]
            not_final_mask = (n_turns != turn_i) & valid_mask
            if not_final_mask.sum() > 0:
                temp_target = torch.zeros(5, device=self.device)
                temp_target[:turn_i - 1] = 1.0
                target_gates[not_final_mask] = temp_target

            current_mod_ids = mod_input_ids[turn_i - 1]
            current_mod_mask = mod_attention_mask[turn_i - 1]

            turn_gates = self.gating(current_mod_ids, current_mod_mask)  # (B, 5)
            with torch.cuda.amp.autocast(enabled=False):
                gating_loss = F.binary_cross_entropy(
                         turn_gates[valid_mask].to(torch.float32),
                         target_gates[valid_mask].to(torch.float32),
                         reduction='mean')
            gating_loss_total += gating_loss

            fusion_processed = ref_fusion_processed

            for turn_j in range(1, turn_i + 1):
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
                mod_attn_mask = torch.cat([query_atts, mod_attention_mask[turn_j - 1]],
                                          dim=1)
                tar_cap_attn_mask = torch.cat([query_atts, cap_attention_mask[turn_j]],
                                              dim=1)

                text_output = self.bertLM(
                    mod_input_ids[turn_j - 1],
                    query_embeds=fusion_processed,  # [b, 32, 768]
                    attention_mask=mod_attn_mask,
                    return_dict=True,
                )
                text_processed = text_output.last_hidden_state[:, : query_tokens.size(1), :]
                query_tokens = self.query_proj(fusion_processed + text_processed)

                fusion_feats = F.normalize((self.text_proj(query_tokens[:, -1, :])), dim=-1)

                next_image_embeds = image_feats[turn_j]
                next_image_atts = image_atts_list[turn_j]

                next_fusion_output = self.Qformer(
                    cap_input_ids[turn_j],
                    query_embeds=query_tokens,
                    attention_mask=tar_cap_attn_mask,
                    encoder_hidden_states=next_image_embeds,
                    encoder_attention_mask=next_image_atts,
                    return_dict=True,
                )
                next_fusion_processed = next_fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
                alpha = turn_gates[:, turn_j - 1].view(batch_size, 1, 1)  # (B, 1, 1)
                fusion_processed = alpha * next_fusion_processed + (1 - alpha) * fusion_processed

                if turn_j == turn_i:
                    tar_fusion_output = self.Qformer(
                        cap_input_ids[turn_j],
                        query_embeds=tar_query_tokens,
                        attention_mask=tar_cap_attn_mask,
                        encoder_hidden_states=next_image_embeds,
                        encoder_attention_mask=next_image_atts,
                        return_dict=True,
                    )
                    tar_fusion_processed = tar_fusion_output.last_hidden_state[:, : tar_query_tokens.size(1), :]
                    tar_fusion_feats = F.normalize((self.target_proj(tar_fusion_processed[:, -1, :])),
                                                   dim=-1)

                    prompt_tokens_expanded = self.prompt_tokens.expand(batch_size, -1, -1)
                    tar_text_output = self.Qformer(
                        cap_input_ids[turn_j],
                        query_embeds=prompt_tokens_expanded,
                        attention_mask=tar_cap_attn_mask,
                        return_dict=True,
                        no_img=True,
                    )
                    tar_cap_text_feat = F.normalize(self.text_proj(tar_text_output.last_hidden_state[:, 0, :]), dim=-1)

                    mod_text_output = self.Qformer(
                        mod_input_ids[turn_j - 1],
                        query_embeds=prompt_tokens_expanded,
                        attention_mask=mod_attn_mask,
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

                    loss_per_turn = sum(loss_dict.values())  # (batch_size,)
                    loss_per_sample += loss_per_turn

                    mask = (n_turns == turn_i)
                    filtered_loss = loss_per_sample[mask]
                    loss_total += filtered_loss.sum() / turn_i

        main_loss_total = loss_total / batch_size
        gating_loss_total = gating_loss_total / self.max_turn

        total_loss = main_loss_total + gating_loss_total

        return total_loss

    @torch.no_grad()
    def inference(self, samples):
        target_feats = samples["target_feats"]
        n_turns = samples["n_turns"]
        images = samples["images"]  # (6, B, 3, 225, 225)
        cap_input_ids = samples["cap_input_ids"]  # (6, B, 32)
        cap_attention_mask = samples["cap_attention_mask"]  # (6, B, 32)
        mod_input_ids = samples["mod_input_ids"]  # (5, B, 40)
        mod_attention_mask = samples["mod_attention_mask"]  # (5, B, 40)
        batch_size = images[0].size(0)

        cached_query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        images = [img.to(self.device, non_blocking=True) for img in images]
        image_feats = [self.ln_vision(self.visual_encoder(img)).detach() for img in images]
        image_atts_list = [torch.ones(f.size()[:-1], dtype=torch.long).to(f.device) for f in image_feats]

        mod_attention_mask = [attn.to(self.device) for attn in mod_attention_mask]
        cap_attention_mask = [attn.to(self.device) for attn in cap_attention_mask]

        cap_input_ids = [input_id.to(self.device) for input_id in cap_input_ids]
        mod_input_ids = [input_id.to(self.device) for input_id in mod_input_ids]

        query_tokens = cached_query_tokens.clone()
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

        ref_image_embeds = image_feats[0]
        ref_image_atts = image_atts_list[0]
        ref_cap_input_ids = cap_input_ids[0]
        ref_cap_attn_mask = torch.cat([query_atts, cap_attention_mask[0]], dim=1)

        ref_fusion_output = self.Qformer(
            ref_cap_input_ids,
            query_embeds=query_tokens,
            attention_mask=ref_cap_attn_mask,
            encoder_hidden_states=ref_image_embeds,
            encoder_attention_mask=ref_image_atts,
            return_dict=True,
        )
        ref_fusion_processed = ref_fusion_output.last_hidden_state[:, : query_tokens.size(1), :]

        last_fusion_feats_all = torch.zeros(
            batch_size, self.text_proj.out_features, device=self.device
        )
        first_fusion_feats_all = torch.zeros(
            batch_size, self.text_proj.out_features, device=self.device
        )
        second_fusion_feats_all = torch.zeros(
            batch_size, self.text_proj.out_features, device=self.device
        )
        for turn_i in range(1, self.max_turn + 1):
            valid_mask = (n_turns >= turn_i)
            final_mask = (n_turns == turn_i)
            if valid_mask.sum() == 0:
                continue

            current_mod_ids = mod_input_ids[turn_i - 1]
            current_mod_mask = mod_attention_mask[turn_i - 1]
            turn_gates = self.gating(current_mod_ids, current_mod_mask)  # (B, 5)
            print(f"turn-{turn_i}", turn_gates[:, :turn_i])
            threshold = 0.5
            binary_turn_gates = (turn_gates >= threshold).float()
            fusion_processed = ref_fusion_processed

            for turn_j in range(1, turn_i + 1):
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
                mod_attn_mask = torch.cat([query_atts, mod_attention_mask[turn_j - 1]],
                                          dim=1)
                tar_cap_attn_mask = torch.cat([query_atts, cap_attention_mask[turn_j]],
                                              dim=1)

                text_output = self.bertLM(
                    mod_input_ids[turn_j - 1],
                    query_embeds=fusion_processed,  # [b, 32, 768]
                    attention_mask=mod_attn_mask,
                    return_dict=True,
                )
                text_processed = text_output.last_hidden_state[:, : query_tokens.size(1), :]
                query_tokens = self.query_proj(fusion_processed + text_processed)

                next_image_embeds = image_feats[turn_j]
                next_image_atts = image_atts_list[turn_j]

                next_fusion_output = self.Qformer(
                    cap_input_ids[turn_j],
                    query_embeds=query_tokens,
                    attention_mask=tar_cap_attn_mask,
                    encoder_hidden_states=next_image_embeds,
                    encoder_attention_mask=next_image_atts,
                    return_dict=True,
                )
                next_fusion_processed = next_fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
                alpha = binary_turn_gates[:, turn_j - 1].view(batch_size, 1, 1)  # (B, 1, 1)
                fusion_processed = alpha * next_fusion_processed + (1 - alpha) * fusion_processed

                if turn_j == turn_i == 1:
                    first_fusion_feats_all = F.normalize(self.text_proj(query_tokens[:, -1, :]), dim=-1)
                elif turn_j == turn_i == 2:
                    second_fusion_feats_all = F.normalize(self.text_proj(query_tokens[:, -1, :]), dim=-1)
                if turn_j == turn_i and final_mask.sum() > 0:
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

        tar_fusion_output = self.Qformer(
            cap_input_ids,
            query_embeds=tar_query_tokens,
            attention_mask=tar_cap_attn_mask,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        tar_fusion_processed = tar_fusion_output.last_hidden_state[:, : tar_query_tokens.size(1), :]
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
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", True)
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
