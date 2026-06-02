"""
SleepMapper — Multimodal CNN: EfficientNet-B0 audio + SpO2 LSTM fusion
Fuses:
  - Audio branch : EfficientNet-B0 backbone (reused from resnet18.py) → (batch, 1280)
  - SpO2 branch  : 2-layer LSTM over 50-step SpO2 sequence → (batch, 32)
  - Fusion head  : Linear(1312, 256) → ReLU → Dropout(0.4) → Linear(256, 1)

Input:
  - mel  : (batch, 1, 128, T) — raw log-mel from TaoMultimodalDataset
            resized internally to (batch, 3, 224, 224) before EfficientNet
  - spo2 : (batch, 50) — normalised SpO2 sequence [0.0, 1.0]

Output:
  - (batch,) raw logit — pass through sigmoid for probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ─── SpO2 Encoder ─────────────────────────────────────────────────────────────

class SpO2Encoder(nn.Module):
    """
    Encodes a 50-step SpO2 sequence into a fixed-length vector.

    Input : (batch, 50) — normalised SpO2 signal at 1Hz
    Output: (batch, 32) — last hidden state of the final LSTM layer

    Using 2-layer LSTM with dropout as specified in the prompt.
    SpO2 is a 1-dimensional physiological signal, so input_size=1.
    """
    def __init__(self):
        super(SpO2Encoder, self).__init__()

        # input_size=1 because SpO2 is a scalar per timestep
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.2,    # as specified in prompt
        )

    def forward(self, spo2):
        # spo2: (batch, 50) → unsqueeze feature dim → (batch, 50, 1)
        x = spo2.unsqueeze(-1)

        # lstm_out: (batch, 50, 32) | h_n: (num_layers=2, batch, 32)
        _, (h_n, _) = self.lstm(x)

        # Take the last layer's hidden state as the sequence encoding
        # h_n[-1]: (batch, 32)
        return h_n[-1]


# ─── Multimodal Model ─────────────────────────────────────────────────────────

class MultimodalSleepMapper(nn.Module):
    """
    Multimodal sleep apnea classifier combining smartphone audio and SpO2.

    Audio branch:
        - EfficientNet-B0 backbone (pretrained ImageNet weights by default)
        - The original classifier head is REPLACED — we extract the 1280-dim
          feature vector from the backbone and fuse it with SpO2 features
        - mel input (1, 128, T) is resized to (3, 224, 224) here internally

    SpO2 branch:
        - SpO2Encoder: 2-layer LSTM → (batch, 32)

    Fusion:
        - Concatenate audio (1280) + spo2 (32) → (1312)
        - Linear(1312, 256) → ReLU → Dropout(0.4) → Linear(256, 1)

    NOTE: We reuse the EfficientNet-B0 backbone architecture (same as SleepResNet18
    in resnet18.py) but we DO NOT import SleepResNet18 because that model wraps
    the entire pipeline including the final 1-output head, which we need to replace
    with a fusion-aware head. We build the backbone directly to access the 1280-dim
    feature space before classification.
    """
    def __init__(self, pretrained=True):
        super(MultimodalSleepMapper, self).__init__()

        # ── Audio encoder: EfficientNet-B0 backbone ──
        base = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )

        # Keep only the feature extractor (everything except the final classifier)
        # EfficientNet-B0 structure: features → avgpool → classifier
        # We replace classifier with Identity so forward() returns (batch, 1280)
        self.audio_encoder = nn.Sequential(
            base.features,       # Conv + MBConv blocks
            base.avgpool,        # AdaptiveAvgPool2d → (batch, 1280, 1, 1)
            nn.Flatten(),        # → (batch, 1280)
        )

        # ── SpO2 encoder: 2-layer LSTM ──
        self.spo2_encoder = SpO2Encoder()

        # ── Fusion classifier ──
        # 1280 (audio) + 32 (spo2) = 1312
        self.fusion_head = nn.Sequential(
            nn.Linear(1280 + 32, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 1),
        )

    def _preprocess_mel(self, mel):
        """
        Resize (batch, 1, 128, T) → (batch, 3, 224, 224) for EfficientNet.

        Matches exactly what dataset.py does for SleepResNet18:
          1. Bilinear interpolation to 224×224
          2. Repeat single channel 3 times (grayscale → pseudo-RGB)
          3. ImageNet normalisation
        """
        # (batch, 1, 128, T) → (batch, 1, 224, 224)
        x = F.interpolate(mel, size=(224, 224), mode="bilinear", align_corners=False)

        # (batch, 1, 224, 224) → (batch, 3, 224, 224)
        x = x.repeat(1, 3, 1, 1)

        # ImageNet normalisation (same values as dataset.py)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        return x

    def forward(self, mel, spo2):
        """
        Args:
            mel  : (batch, 1, 128, T) — raw log-mel from TaoMultimodalDataset
            spo2 : (batch, 50)        — normalised SpO2 window

        Returns:
            (batch,) raw logit — squeeze the trailing dim for BCEWithLogitsLoss
        """
        # Audio branch
        mel_rgb    = self._preprocess_mel(mel)           # (batch, 3, 224, 224)
        audio_feat = self.audio_encoder(mel_rgb)          # (batch, 1280)

        # SpO2 branch
        spo2_feat  = self.spo2_encoder(spo2)              # (batch, 32)

        # Fuse
        fused  = torch.cat([audio_feat, spo2_feat], dim=1)  # (batch, 1312)
        logits = self.fusion_head(fused)                     # (batch, 1)

        return logits.squeeze(1)  # (batch,) — consistent with BCEWithLogitsLoss


# ─── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = MultimodalSleepMapper(pretrained=False)
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Dummy forward pass
    mel  = torch.randn(4, 1, 128, 938)   # batch=4
    spo2 = torch.rand(4, 50)
    out  = model(mel, spo2)
    print(f"Output shape    : {out.shape}")  # (4,)
