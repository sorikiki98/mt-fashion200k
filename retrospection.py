import torch.cuda
import torch.nn as nn
from lavis.models.blip2_models.Qformer import BertModel
from transformers import BertTokenizer


class RetrospectiveMultiTurnCirModel(nn.Module):
    def __init__(self, blip_model, len_turn=40):
        super().__init__()
        self.blip_model = blip_model
        self.sparse_bert = SparseBertModel(len_turn)
        self.pooling = PoolingProjector(len_turn, blip_model.max_txt_len - 1)
        self.len_turn = len_turn

    def forward(self, samples):
        n_turns = samples["n_turns"]  # (B,)
        mod1_inputs = samples["mod1"]  # (B,)
        mod2_inputs = samples["mod2"]
        mod3_inputs = samples["mod3"]
        mod4_inputs = samples["mod4"]
        mod5_inputs = samples["mod5"]

        history_dict = self.blip_model.forward_history(samples)

        last_ref_fusion_processed = history_dict["last_ref"][0]  # (B, 32, 768)
        last_tar_fusion_feats = history_dict["last_tar"][1]  # (B, 256)

        mod_concat = list(zip(mod1_inputs, mod2_inputs, mod3_inputs, mod4_inputs, mod5_inputs))
        text_output = self.sparse_bert(mod_concat, n_turns)

        learned_modifier_tokens = self.pooling(text_output)  # (B, 31, 768)
        output_dict = {
            "last_fusion_processed": last_ref_fusion_processed,
            "learned_modifier_tokens": learned_modifier_tokens,
            "last_tar_fusion_feats": last_tar_fusion_feats
        }

        loss = self.blip_model.fine_tune(output_dict)
        return loss


class SparseBertModel(nn.Module):
    def __init__(self, len_turn):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                                       truncation_side="right",
                                                       use_fast=False)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.len_turn = len_turn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.linear_proj = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.max_turn = 5

    def forward(self, mods, n_turns):
        """
        :param mods: (B, max_turn)
        :param n_turns: (B,),
        :return: (B, len_turn, 768)
        """
        all_input_ids = []
        all_attention_mask = []

        for i, sample in enumerate(mods):
            n_turn = n_turns[i]
            n_pad = (self.max_turn - n_turn) * self.len_turn

            try:
                tokenized = self.tokenizer(
                    [str(x) for x in sample],  # sample: List[str]
                    padding="max_length",
                    truncation=True,
                    max_length=self.len_turn,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
            except Exception as e:
                print("Tokenizer input (mod_concat):", sample)
                raise e
            input_ids = tokenized["input_ids"]
            input_ids_concat = input_ids.view(1, -1)  # (1, 200)

            attention_mask = tokenized["attention_mask"]
            attention_mask_concat = attention_mask.view(1, -1)  # (1, 200)

            if n_turn < self.max_turn:
                input_ids_concat[:, -n_pad:] = 0
                attention_mask_concat[:, -n_pad:] = 0

            all_input_ids.append(input_ids_concat)
            all_attention_mask.append(attention_mask_concat)

        all_input_ids = torch.cat(all_input_ids, dim=0).to(self.device)
        all_attention_mask = torch.cat(all_attention_mask, dim=0).to(self.device)

        text_output = self.bert(
            all_input_ids,
            attention_mask=all_attention_mask,
            return_dict=True
        )

        text_output_last_hidden = text_output.last_hidden_state

        text_output_last_hidden_processed = []
        for i in range(len(n_turns)):
            n = n_turns[i]
            start = self.len_turn * (n - 1)
            end = self.len_turn * n
            text_output_last_hidden_processed.append(text_output_last_hidden[i, start:end, :])
        text_output_processed = torch.stack(text_output_last_hidden_processed, dim=0)
        return text_output_processed


class PoolingProjector(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.pool_proj = nn.Linear(input_len, output_len)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, D, 40)
        x = self.pool_proj(x)  # (B, D, 31)
        x = x.transpose(1, 2)  # (B, 31, D)
        return x
