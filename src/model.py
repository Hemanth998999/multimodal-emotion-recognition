import torch
import torch.nn as nn
from transformers import RobertaModel

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        attn_out, _ = self.attn(query, key_value, key_value)
        return self.norm(query + attn_out)


class MultimodalEmotionModel(nn.Module):
    def __init__(self, num_classes=4, text_hidden=768, audio_dim=40,
                 proj_dim=256, dropout=0.3):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base")

        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.text_attend_audio = CrossModalAttention(proj_dim)
        self.audio_attend_text = CrossModalAttention(proj_dim)

        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim * 2),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, audio_feat):
        text_out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output

        text_emb = self.text_proj(text_out)
        audio_emb = self.audio_proj(audio_feat)

        t = text_emb.unsqueeze(1)
        a = audio_emb.unsqueeze(1)

        t_attn = self.text_attend_audio(t, a).squeeze(1)
        a_attn = self.audio_attend_text(a, t).squeeze(1)

        fused = torch.cat([t_attn, a_attn], dim=-1)
        gate = self.gate(fused)
        gated = fused * gate

        logits = self.classifier(gated)
        return logits