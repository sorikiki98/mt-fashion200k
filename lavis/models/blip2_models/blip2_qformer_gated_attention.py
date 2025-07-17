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
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures


@registry.register_model("blip2_qformer_gated_attention")
class Blip2QformerGatedAttention(Blip2Base):
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
        ref_img = samples["ref_img"]
        ref_cap = samples["ref_cap"]
        mod1_inputs = samples["mod1"]
        tar1_img = samples["tar1_img"]
        tar1_cap = samples["tar1_cap"]
        mod2_inputs = samples["mod2"]
        tar2_img = samples["tar2_img"]
        tar2_cap = samples["tar2_cap"]
        mod3_inputs = samples["mod3"]
        tar3_img = samples["tar3_img"]
        tar3_cap = samples["tar3_cap"]
        mod4_inputs = samples["mod4"]
        tar4_img = samples["tar4_img"]
        tar4_cap = samples["tar4_cap"]
        mod5_inputs = samples["mod5"]
        tar5_img = samples["tar5_img"]
        tar5_cap = samples["tar5_cap"]

        imgs = [ref_img, tar1_img, tar2_img, tar3_img, tar4_img, tar5_img]
        caps = [ref_cap, tar1_cap, tar2_cap, tar3_cap, tar4_cap, tar5_cap]
        mods = [mod1_inputs, mod2_inputs, mod3_inputs, mod4_inputs, mod5_inputs]

        cached_query_tokens = self.query_tokens.expand(ref_img.size(0), -1, -1)
        with torch.no_grad():
            image_feats = [self.ln_vision(self.visual_encoder(img)).detach() for img in imgs]
            image_atts_list = [torch.ones(f.size()[:-1], dtype=torch.long).to(f.device) for f in image_feats]
            mod_tokens_all = [self.tokenizer(mods[i],
                                             padding="max_length",
                                             truncation=True,
                                             max_length=self.max_txt_len,
                                             return_tensors="pt").to(ref_img.device)
                              for i in range(5)]
            cap_tokens_all = [self.tokenizer(caps[i],
                                             padding="max_length",
                                             truncation=True,
                                             max_length=self.max_txt_len,
                                             return_tensors="pt").to(ref_img.device)
                              for i in range(6)]

        loss_total = 0
        loss_per_sample = torch.zeros(ref_img.size(0), device=ref_img.device)
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

            modifier_tokens = mod_tokens_all[turn_i - 1]
            ref_caption_tokens = cap_tokens_all[turn_i - 1]
            tar_caption_tokens = cap_tokens_all[turn_i]

            attention_mask = torch.cat([query_atts, modifier_tokens.attention_mask], dim=1)
            ref_cap_attn_mask = torch.cat([query_atts, ref_caption_tokens.attention_mask], dim=1)

            fusion_output = self.Qformer.bert(
                ref_caption_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=ref_cap_attn_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            fusion_processed = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
            text_output = self.bertLM.bert(
                modifier_tokens.input_ids,
                query_embeds=fusion_processed,  # [b, 32, 768]
                attention_mask=attention_mask,
                return_dict=True,
            )
            text_processed = text_output.last_hidden_state[:, : query_tokens.size(1), :]
            query_tokens = self.query_proj(fusion_processed + text_processed)
            fusion_feats = F.normalize((self.text_proj(query_tokens[:, -1, :])), dim=-1)

            target_img_embeds = image_feats[turn_i]
            target_img_atts = image_atts_list[turn_i]

            tar_attention_mask = torch.cat([tar_query_atts, tar_caption_tokens.attention_mask], dim=1)
            tar_fusion_output = self.Qformer.bert(
                tar_caption_tokens.input_ids,
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
                tar_caption_tokens.input_ids,
                query_embeds=tar_text_tokens,
                attention_mask=tar_attention_mask,
                return_dict=True,
                no_img=True,
            )
            tar_cap_text_feat = F.normalize(self.text_proj(tar_text_output.last_hidden_state[:, 0, :]), dim=-1)

            mod_text_tokens = self.prompt_tokens.expand(
                image_embeds.shape[0], -1, -1)
            mod_text_output = self.Qformer.bert(
                modifier_tokens.input_ids,
                query_embeds=mod_text_tokens,
                attention_mask=attention_mask,
                return_dict=True,
                no_img=True,
            )
            mod_cap_feat = F.normalize(self.text_proj(mod_text_output.last_hidden_state[:, 0, :]), dim=-1)

            loss_dict = {
                "loss_fus2tar": self.BBC_loss(fusion_feats.to(ref_img.device), tar_fusion_feats.to(ref_img.device)),
                'loss_fus2cap': self.BBC_loss(fusion_feats.to(ref_img.device), tar_cap_text_feat.to(ref_img.device)),
                'loss_mod2fus': self.BBC_loss(mod_cap_feat.to(ref_img.device), tar_fusion_feats.to(ref_img.device)),
                'loss_mod2cap': self.BBC_loss(mod_cap_feat.to(ref_img.device), tar_cap_text_feat.to(ref_img.device))
            }
            loss_per_turn = sum(loss_dict.values())
            loss_per_sample += loss_per_turn

            mask = (n_turns == turn_i)
            filtered_loss = loss_per_sample[mask]
            loss_total += filtered_loss.sum() / turn_i

        return loss_total / ref_img.size(0)

    @torch.no_grad()
    def inference(self, samples):
        target_feats = samples["target_feats"]
        n_turns = samples["n_turns"]
        ref_img = samples["ref_img"]
        ref_cap = samples["ref_cap"]
        mod1_inputs = samples["mod1"]
        tar1_img = samples["tar1_img"]
        tar1_cap = samples["tar1_cap"]
        mod2_inputs = samples["mod2"]
        tar2_img = samples["tar2_img"]
        tar2_cap = samples["tar2_cap"]
        mod3_inputs = samples["mod3"]
        tar3_img = samples["tar3_img"]
        tar3_cap = samples["tar3_cap"]
        mod4_inputs = samples["mod4"]
        tar4_img = samples["tar4_img"]
        tar4_cap = samples["tar4_cap"]
        mod5_inputs = samples["mod5"]
        tar5_img = samples["tar5_img"]
        tar5_cap = samples["tar5_cap"]

        imgs = [ref_img, tar1_img, tar2_img, tar3_img, tar4_img, tar5_img]
        caps = [ref_cap, tar1_cap, tar2_cap, tar3_cap, tar4_cap, tar5_cap]
        mods = [mod1_inputs, mod2_inputs, mod3_inputs, mod4_inputs, mod5_inputs]
        cached_query_tokens = self.query_tokens.expand(ref_img.size(0), -1, -1)

        image_feats = [self.ln_vision(self.visual_encoder(img)).detach() for img in imgs]
        image_atts_list = [torch.ones(f.size()[:-1], dtype=torch.long).to(f.device) for f in image_feats]
        mod_tokens_all = [self.tokenizer(mods[i],
                                         padding="max_length",
                                         truncation=True,
                                         max_length=self.max_txt_len,
                                         return_tensors="pt").to(ref_img.device)
                          for i in range(5)]
        cap_tokens_all = [self.tokenizer(caps[i],
                                         padding="max_length",
                                         truncation=True,
                                         max_length=self.max_txt_len,
                                         return_tensors="pt").to(ref_img.device)
                          for i in range(6)]

        last_fusion_feats_all = torch.zeros(
            ref_img.size(0), self.text_proj.out_features, device=ref_img.device
        )
        first_fusion_feats_all = torch.zeros(
            ref_img.size(0), self.text_proj.out_features, device=ref_img.device
        )
        second_fusion_feats_all = torch.zeros(
            ref_img.size(0), self.text_proj.out_features, device=ref_img.device
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
            modifier_tokens = mod_tokens_all[turn_i - 1]
            ref_caption_tokens = cap_tokens_all[turn_i - 1]

            attention_mask = torch.cat([query_atts, modifier_tokens.attention_mask], dim=1)
            ref_cap_attn_mask = torch.cat([query_atts, ref_caption_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                ref_caption_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=ref_cap_attn_mask,
                encoder_hidden_states=image_embeds,  # cross attention
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            fusion_processed = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
            text_output = self.bertLM.bert(
                modifier_tokens.input_ids,
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
    def extract_target_features(self, image, caption):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)

        cap_tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        ##
        tar_query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)
        tar_query_atts = torch.ones(tar_query_tokens.size()[:-1], dtype=torch.long).to(self.device)

        tar_cap_attn_mask = torch.cat([tar_query_atts, cap_tokens.attention_mask], dim=1)

        tar_fusion_output = self.Qformer.bert(
            cap_tokens.input_ids,
            query_embeds=tar_query_tokens,  # Qformer里query embeds和modifier_tokens会拼接
            attention_mask=tar_cap_attn_mask,
            encoder_hidden_states=image_embeds_frozen,  # cross attention
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        tar_fusion_processed = tar_fusion_output.last_hidden_state[:, : tar_query_tokens.size(1), :]

        tar_fusion_feats = F.normalize((self.target_proj(tar_fusion_processed)[:, -1, :]),
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
            distance_matrix[:, start:end] = chunk_distance  # 更新总距离矩阵
        return distance_matrix

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert image is not None, "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)
            query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert caption is not None, "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(self.device)

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodal query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)
            query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(self.device)
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

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

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(image_inputs.device)
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_inputs.device)
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=3,
            max_length=30,
            min_length=10,
            top_p=0.9,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = torch.LongTensor(image.size(0), 1).fill_(self.tokenizer.bos_token_id).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions
