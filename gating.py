import torch.nn as nn


class TurnGatingModule(nn.Module):
    def __init__(self, vocab_size, max_turns, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.max_turns = max_turns
        self.mod_embedding = nn.Embedding(vocab_size, embed_dim)
        self.mod_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.gate_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, max_turns),
            nn.Sigmoid()
        )

    def forward(self, mod_input_ids, mod_attention_mask=None):
        # mod_input_ids: [B, 40]
        # mod_attention_mask: [B, 40]

        # Embed modification tokens
        mod_embeds = self.mod_embedding(mod_input_ids)  # [B, 40, embed_dim]

        # Pack sequences if attention mask is provided (for efficiency)
        if mod_attention_mask is not None:
            lengths = mod_attention_mask.sum(dim=1).cpu()
            packed_embeds = nn.utils.rnn.pack_padded_sequence(
                mod_embeds, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.mod_lstm(packed_embeds)
        else:
            _, (hidden, cell) = self.mod_lstm(mod_embeds)

        last_hidden = hidden[-1]  # [B, hidden_dim]
        turn_gates = self.gate_generator(last_hidden)  # [B, max_turns]

        return turn_gates
