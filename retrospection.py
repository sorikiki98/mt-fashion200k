import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import math
from lavis.models.blip2_models.Qformer import BertModel
from transformers import BertTokenizer


class RetrospectiveMultiTurnCirModel(nn.Module):
    def __init__(
            self,
            blip_model,
            len_turn,
            max_turn):
        super().__init__()
        self.blip_model = blip_model
        hidden_dim = blip_model.Qformer.config.hidden_size
        vocab_size = blip_model.tokenizer.vocab_size

        self.text_decoder = RetrospectiveTextDecoder(hidden_dim, vocab_size, len_turn, max_turn)
        self.pooling = PoolingProjector(len_turn, blip_model.max_txt_len - 1)
        self.len_turn = len_turn
        self.max_turn = max_turn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, samples):
        self.text_decoder.clear_cache()

        n_turns = samples["n_turns"]  # (B,)
        images = samples["images"]  # (6, B, 3, 225, 225)
        cap_input_ids = samples["cap_input_ids"]  # (6, B, 32)
        cap_attention_mask = samples["cap_attention_mask"]  # (6, B, 32)
        mod_input_ids = samples["mod_input_ids"]  # (5, B, 40)
        mod_attention_mask = samples["mod_attention_mask"]  # (5, B, 40)
        rollback_img_ids = samples["rollback_img_id"]
        rollback_captions = samples["rollback_caption"]
        combination_captions = samples["combination_caption"]

        cached_query_tokens = self.blip_model.query_tokens.expand(images[0].size(0), -1, -1)

        mod_attention_mask = [attn.to(self.device) for attn in mod_attention_mask]
        cap_attention_mask = [attn.to(self.device) for attn in cap_attention_mask]

        cap_input_ids = [input_id.to(self.device) for input_id in cap_input_ids]
        mod_input_ids = [input_id.to(self.device) for input_id in mod_input_ids]

        with torch.no_grad():
            images = [img.to(self.device, non_blocking=True) for img in images]
            image_feats = [self.blip_model.ln_vision(self.blip_model.visual_encoder(img)).detach() for img in images]
            image_atts_list = [torch.ones(f.size()[:-1], dtype=torch.long).to(f.device) for f in image_feats]

        loss_total = 0
        loss_per_sample = torch.zeros(images[0].size(0), device=self.device)
        query_tokens = None
        for turn_i in range(1, self.max_turn + 1):
            valid_mask = (n_turns >= turn_i)
            if valid_mask.sum() == 0:
                continue
            if query_tokens is None:
                query_tokens = cached_query_tokens.clone()
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            tar_query_tokens = cached_query_tokens.clone()
            tar_query_atts = torch.ones(tar_query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            attn_mask = torch.cat([query_atts, query_atts], dim=1)
            ref_cap_attn_mask = torch.cat([query_atts, cap_attention_mask[turn_i - 1][:, :query_tokens.size(1)]], dim=1)

            fusion_last_hidden = self.blip_model.forward_fusion(
                input_ids=cap_input_ids[turn_i - 1][:, :query_tokens.size(1)],
                attention_mask=ref_cap_attn_mask,
                image_embeds=image_feats[turn_i - 1],
                image_atts=image_atts_list[turn_i - 1],
                query_tokens=query_tokens)
            fusion_processed = fusion_last_hidden[:, :query_tokens.size(1), :]
            learned_embeds, _ = self.text_decoder(input_ids=mod_input_ids[turn_i - 1],
                                                  attention_mask=mod_attention_mask[turn_i - 1],
                                                  current_position=(turn_i - 1) * self.len_turn,
                                                  use_cache=True)
            learned_embeds = self.pooling(learned_embeds)

            text_last_hidden = self.blip_model.forward_text(attention_mask=attn_mask,
                                                            learned_embeds=learned_embeds,
                                                            query_tokens=fusion_processed)
            text_processed = text_last_hidden[:, 1:query_tokens.size(1) + 1, :]
            query_tokens = self.blip_model.query_proj(fusion_processed + text_processed)
            fusion_feats = F.normalize((self.blip_model.text_proj(query_tokens[:, -1, :])), dim=-1)

            tar_attn_mask = torch.cat([tar_query_atts, cap_attention_mask[turn_i][:, :tar_query_tokens.size(1)]], dim=1)
            tar_fusion_last_hidden = self.blip_model.forward_fusion(
                input_ids=cap_input_ids[turn_i][:, :tar_query_tokens.size(1)],
                attention_mask=tar_attn_mask,
                image_embeds=image_feats[turn_i],
                image_atts=image_atts_list[turn_i],
                query_tokens=tar_query_tokens)
            tar_fusion_processed = tar_fusion_last_hidden[:, :tar_query_tokens.size(1), :]
            tar_fusion_feats = F.normalize((self.blip_model.target_proj(tar_fusion_processed[:, -1, :])),
                                           dim=-1)

            tar_text_tokens = self.blip_model.prompt_tokens.expand(images[0].size(0), -1, -1)
            tar_text_last_hidden = self.blip_model.forward_fusion(
                input_ids=cap_input_ids[turn_i][:, :tar_text_tokens.size(1)],
                attention_mask=tar_attn_mask,
                query_tokens=tar_text_tokens,
                no_img=True)
            tar_cap_text_feat = F.normalize(self.blip_model.text_proj(tar_text_last_hidden[:, 0, :]), dim=-1)

            mod_text_tokens = self.blip_model.prompt_tokens.expand(images[0].size(0), -1, -1)
            mod_text_last_hidden = self.blip_model.forward_fusion(attention_mask=attn_mask,
                                                                  query_tokens=mod_text_tokens,
                                                                  learned_embeds=learned_embeds,
                                                                  no_img=True)
            mod_cap_feat = F.normalize(self.blip_model.text_proj(mod_text_last_hidden[:, 0, :]), dim=-1)
            loss_dict = {
                "loss_fus2tar": self.blip_model.BBC_loss(fusion_feats.to(self.device),
                                                         tar_fusion_feats.to(self.device)),
                'loss_fus2cap': self.blip_model.BBC_loss(fusion_feats.to(self.device),
                                                         tar_cap_text_feat.to(self.device)),
                'loss_mod2fus': self.blip_model.BBC_loss(mod_cap_feat.to(self.device),
                                                         tar_fusion_feats.to(self.device)),
                'loss_mod2cap': self.blip_model.BBC_loss(mod_cap_feat.to(self.device),
                                                         tar_cap_text_feat.to(self.device))
            }
            loss_per_turn = sum(loss_dict.values())
            loss_per_sample += loss_per_turn

            mask = (n_turns == turn_i)
            filtered_loss = loss_per_sample[mask]
            loss_total += filtered_loss.sum() / turn_i

        return loss_total / images[0].size(0)

    @torch.no_grad()
    def inference(self, samples):
        self.text_decoder.clear_cache()

        target_feats = samples["target_feats"]
        n_turns = samples["n_turns"]  # (B,)
        images = samples["images"]  # (6, B, 3, 225, 225)
        cap_input_ids = samples["cap_input_ids"]  # (6, B, 32)
        cap_attention_mask = samples["cap_attention_mask"]  # (6, B, 32)
        mod_input_ids = samples["mod_input_ids"]  # (5, B, 40)
        mod_attention_mask = samples["mod_attention_mask"]  # (5, B, 40)

        cached_query_tokens = self.blip_model.query_tokens.expand(images[0].size(0), -1, -1)

        mod_attention_mask = [attn.to(self.device) for attn in mod_attention_mask]
        cap_attention_mask = [attn.to(self.device) for attn in cap_attention_mask]

        cap_input_ids = [input_id.to(self.device) for input_id in cap_input_ids]
        mod_input_ids = [input_id.to(self.device) for input_id in mod_input_ids]

        images = [img.to(self.device, non_blocking=True) for img in images]
        image_feats = [self.blip_model.ln_vision(self.blip_model.visual_encoder(img)).detach() for img in images]
        image_atts_list = [torch.ones(f.size()[:-1], dtype=torch.long).to(f.device) for f in image_feats]

        last_ref_fusion_feats_all = torch.zeros(
            images[0].size(0), self.blip_model.text_proj.out_features, device=self.device
        )
        first_fusion_feats_all = torch.zeros(
            images[0].size(0), self.blip_model.text_proj.out_features, device=self.device
        )
        second_fusion_feats_all = torch.zeros(
            images[0].size(0), self.blip_model.text_proj.out_features, device=self.device
        )
        query_tokens = None
        for turn_i in range(1, self.max_turn + 1):
            valid_mask = (n_turns >= turn_i)
            final_mask = (n_turns == turn_i)
            if valid_mask.sum() == 0:
                continue
            if query_tokens is None:
                query_tokens = cached_query_tokens.clone()
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            attn_mask = torch.cat([query_atts, query_atts], dim=1)
            ref_cap_attn_mask = torch.cat([query_atts, cap_attention_mask[turn_i - 1][:, :query_tokens.size(1)]], dim=1)

            fusion_last_hidden = self.blip_model.forward_fusion(
                input_ids=cap_input_ids[turn_i - 1][:, :query_tokens.size(1)],
                attention_mask=ref_cap_attn_mask,
                image_embeds=image_feats[turn_i - 1],
                image_atts=image_atts_list[turn_i - 1],
                query_tokens=query_tokens)
            fusion_processed = fusion_last_hidden[:, :query_tokens.size(1), :]
            learned_embeds, _ = self.text_decoder(input_ids=mod_input_ids[turn_i - 1],
                                                  attention_mask=mod_attention_mask[turn_i - 1],
                                                  current_position=(turn_i - 1) * self.len_turn,
                                                  use_cache=True)
            learned_embeds = self.pooling(learned_embeds)

            text_last_hidden = self.blip_model.forward_text(attention_mask=attn_mask,
                                                            learned_embeds=learned_embeds,
                                                            query_tokens=fusion_processed)
            text_processed = text_last_hidden[:, 1:query_tokens.size(1) + 1, :]
            query_tokens = self.blip_model.query_proj(fusion_processed + text_processed)

            if turn_i == 1:
                first_fusion_feats_all = F.normalize(self.blip_model.text_proj(query_tokens[:, -1, :]), dim=-1)
            elif turn_i == 2:
                second_fusion_feats_all = F.normalize(self.blip_model.text_proj(query_tokens[:, -1, :]), dim=-1)
            if final_mask.sum() > 0:
                selected_feats = self.blip_model.text_proj(query_tokens[:, -1, :])[final_mask]
                projected_feats = F.normalize(selected_feats, dim=-1)
                last_ref_fusion_feats_all[final_mask] = projected_feats

        first_sim_matrix = self.blip_model.compute_distance_matrix(first_fusion_feats_all.to(self.device),
                                                                   target_feats)
        second_sim_matrix = self.blip_model.compute_distance_matrix(second_fusion_feats_all.to(self.device),
                                                                    target_feats)
        last_sim_matrix = self.blip_model.compute_distance_matrix(last_ref_fusion_feats_all.to(self.device),
                                                                  target_feats)
        return first_sim_matrix, second_sim_matrix, last_sim_matrix


class RetrospectiveTextDecoder(nn.Module):
    def __init__(
            self,
            hidden_dim,
            vocab_size,
            len_turn,
            max_turn,
            num_layers=6,
            num_heads=8,
            dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = len_turn * max_turn
        self.len_turn = len_turn

        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(self.max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TextDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.kv_cache = None
        self.mask_cache = None

    def init_kv_cache(self, batch_size: int):
        self.kv_cache = []
        for _ in range(self.num_layers):
            k_cache = torch.zeros(
                batch_size, self.num_heads, 0, self.hidden_dim // self.num_heads,
                device=self.token_embeddings.weight.device
            )
            v_cache = torch.zeros(
                batch_size, self.num_heads, 0, self.hidden_dim // self.num_heads,
                device=self.token_embeddings.weight.device
            )
            mask_cache = None
            self.kv_cache.append((k_cache, v_cache, mask_cache))

    def clear_cache(self):
        self.kv_cache = None
        self.mask_cache = None

    def get_cache_memory_usage(self):
        if self.kv_cache is None:
            return 0

        total_size = 0
        for k_cache, v_cache, mask_cache in self.kv_cache:
            total_size += k_cache.element_size() * k_cache.nelement()
            total_size += v_cache.element_size() * v_cache.nelement()
            if mask_cache is not None:
                total_size += mask_cache.element_size() * mask_cache.nelement()

        return total_size

    def forward(
            self,
            input_ids,
            attention_mask,
            current_position,
            use_cache
    ):
        batch_size, seq_len = input_ids.shape
        assert seq_len == self.len_turn, f"Expected turn length {self.len_turn}, got {seq_len}"

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.arange(
            current_position, current_position + seq_len,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = self.dropout(token_embeds + position_embeds)

        if use_cache and (self.kv_cache is None or current_position == 0):
            self.init_kv_cache(batch_size)

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            if use_cache:
                layer_kv_cache = self.kv_cache[i] if self.kv_cache else None
            else:
                layer_kv_cache = None

            hidden_states, updated_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                kv_cache=layer_kv_cache,
                use_cache=use_cache
            )

            if use_cache:
                new_kv_cache.append(updated_kv)

        hidden_states = self.ln_f(hidden_states)
        output = self.output_proj(hidden_states)
        if use_cache:
            self.kv_cache = new_kv_cache

        return output, self.kv_cache


class TextDecoderLayer(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_heads,
            dropout
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.self_attn = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            hidden_states,
            attention_mask,
            kv_cache,
            use_cache
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, updated_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache
        )
        hidden_states = residual + self.dropout(attn_output)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, updated_kv


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_heads,
            dropout
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            kv_cache=None,
            use_cache=True
    ):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if attention_mask is not None:
            # [batch_size, 40] -> [batch_size, 1, 1, 40]
            current_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            current_mask = None

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if use_cache and kv_cache is not None:
            k_cache, v_cache, mask_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)  # [batch, heads, total_seq, head_dim]
            v = torch.cat([v_cache, v], dim=2)

            if current_mask is not None:
                if mask_cache is not None:
                    full_mask = torch.cat([mask_cache, current_mask], dim=-1)
                else:
                    full_mask = current_mask
            else:
                full_mask = mask_cache
        else:
            full_mask = current_mask

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if full_mask is not None:
            # full_mask: [batch, 1, 1, total_seq]
            # attn_weights: [batch, heads, 40, total_seq]
            mask_value = -1e4
            attn_weights = attn_weights.masked_fill(full_mask.eq(0), mask_value)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, head, 40, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        attn_output = self.o_proj(attn_output)

        if use_cache:
            updated_kv = (k, v, full_mask)
        else:
            updated_kv = None

        return attn_output, updated_kv


"""
class TextualHistoryEncoder(nn.Module):
    def __init__(self, len_turn):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.len_turn = len_turn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.linear_proj = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.max_turn = 5

    def forward(self, samples, n_turns):
        batch_size = len(n_turns)

        all_input_ids = []
        all_attention_mask = []

        for i in range(batch_size):
            n_turn = n_turns[i]
            n_pad = (self.max_turn - n_turn) * self.len_turn

            sample_mod_input_ids = []
            sample_mod_attention_mask = []
            for j in range(self.max_turn):
                sample_mod_input_ids.append(samples["mod_input_ids"][j][i])
                sample_mod_attention_mask.append(samples["mod_attention_mask"][j][i])

            sample_mod_input_ids = torch.cat(sample_mod_input_ids, dim=0).view(1, -1)
            sample_mod_attention_mask = torch.cat(sample_mod_attention_mask, dim=0).view(1, -1)

            if n_turn < self.max_turn:
                sample_mod_input_ids[:, -n_pad:] = 0
                sample_mod_attention_mask[:, -n_pad:] = 0

            all_input_ids.append(sample_mod_input_ids)
            all_attention_mask.append(sample_mod_attention_mask)

        all_input_ids = torch.cat(all_input_ids, dim=0).to(self.device)  # (B, 200)
        all_attention_mask = torch.cat(all_attention_mask, dim=0).to(self.device)  # (B, 200)

        text_output = self.bert(
            all_input_ids,
            attention_mask=all_attention_mask,
            return_dict=True
        )

        text_output_last_hidden = text_output.last_hidden_state

        text_output_last_hidden_processed = []
        for i in range(batch_size):
            n = n_turns[i]
            start = self.len_turn * (n - 1)
            end = self.len_turn * n
            text_output_last_hidden_processed.append(text_output_last_hidden[i, start:end, :])
        text_output_processed = torch.stack(text_output_last_hidden_processed, dim=0)
        return text_output_processed
"""


class PoolingProjector(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.pool_proj = nn.Linear(input_len, output_len)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, D, 40)
        x = self.pool_proj(x)  # (B, D, 31)
        x = x.transpose(1, 2)  # (B, 31, D)
        return x
