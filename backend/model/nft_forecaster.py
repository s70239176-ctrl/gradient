"""
NFTFloorForecaster — Multimodal NFT Floor Price Prediction Model
Dual-tower architecture: PatchTST (time-series) + CLIP (visual) with Gated Fusion
"""
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Dict


class PatchEmbedding(nn.Module):
    def __init__(self, seq_len: int, patch_len: int, d_model: int, n_features: int):
        super().__init__()
        self.patch_len = patch_len
        n_patches = seq_len // patch_len
        self.projection = nn.Linear(patch_len * n_features, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

    def forward(self, x):
        B, T, F = x.shape
        n_patches = T // self.patch_len
        x = x[:, :n_patches * self.patch_len, :]
        x = x.reshape(B, n_patches, self.patch_len * F)
        return self.projection(x) + self.pos_enc


class SmallPatchTST(nn.Module):
    def __init__(self, seq_len=90, n_features=11, patch_len=9, d_model=128, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(seq_len, patch_len, d_model, n_features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_dim = d_model

    def forward(self, x):
        patches = self.patch_embed(x)
        encoded = self.encoder(patches)
        return encoded.mean(dim=1)


class GatedFusion(nn.Module):
    def __init__(self, ts_dim=128, vis_dim=512, hidden=256):
        super().__init__()
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, ts_dim), nn.LayerNorm(ts_dim), nn.GELU()
        )
        self.gate = nn.Sequential(nn.Linear(ts_dim * 2, ts_dim), nn.Sigmoid())
        self.fusion_mlp = nn.Sequential(
            nn.Linear(ts_dim * 2, hidden), nn.GELU(),
            nn.Dropout(0.15), nn.Linear(hidden, hidden // 2), nn.GELU(),
        )
        self.out_dim = hidden // 2

    def forward(self, ts_emb, vis_emb):
        vis = self.vis_proj(vis_emb)
        combined = torch.cat([ts_emb, vis], dim=-1)
        gate = self.gate(combined)
        gated_vis = gate * vis
        fused = torch.cat([ts_emb, gated_vis], dim=-1)
        return self.fusion_mlp(fused)


class NFTFloorForecaster(nn.Module):
    """
    Multimodal NFT floor price forecaster.
    Inputs:
        ts_input:  (B, 90, 11) — 90-day time-series features
        vis_input: (B, 512)    — CLIP ViT-B/32 collection embedding
    Outputs:
        direction_logit: (B, 1) — sigmoid → P(price UP >2%)
        magnitude:       (B, 1) — predicted % change in [-1, 1]
    """
    def __init__(self):
        super().__init__()
        self.ts_encoder = SmallPatchTST()
        self.fusion = GatedFusion()
        self.direction_head = nn.Linear(self.fusion.out_dim, 1)
        self.magnitude_head = nn.Sequential(nn.Linear(self.fusion.out_dim, 1), nn.Tanh())

    def forward(self, ts_input, vis_input):
        ts_emb = self.ts_encoder(ts_input)
        fused = self.fusion(ts_emb, vis_input)
        return self.direction_head(fused), self.magnitude_head(fused)

    @torch.no_grad()
    def predict(self, ts_input: np.ndarray, vis_input: np.ndarray) -> Dict:
        ts_t = torch.FloatTensor(ts_input)
        vis_t = torch.FloatTensor(vis_input)
        if ts_t.ndim == 2:
            ts_t = ts_t.unsqueeze(0)
        if vis_t.ndim == 1:
            vis_t = vis_t.unsqueeze(0)
        dir_logit, magnitude = self.forward(ts_t, vis_t)
        prob_up = torch.sigmoid(dir_logit).item()
        pct_change = magnitude.item() * 100
        confidence = max(prob_up, 1 - prob_up)
        return {
            "direction": "UP" if prob_up > 0.5 else "DOWN",
            "confidence": round(confidence, 4),
            "prob_up": round(prob_up, 4),
            "predicted_pct_change": round(pct_change, 2),
        }


def load_model(weights_path: str = None, device: str = "cpu") -> NFTFloorForecaster:
    model = NFTFloorForecaster().to(device)
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model
