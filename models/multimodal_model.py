"""
Simple fusion classifier (video-only friendly). The training script infers input dims
from dataset sample.

Architecture:
 - small MLP encoder per-modality
 - concatenation
 - fused MLP -> logits
"""

import torch
import torch.nn as nn

class SmallEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.net(x)

class MultimodalClassifier(nn.Module):
    def __init__(self, video_dim, audio_dim, text_dim, hidden=256, num_classes=2):
        super().__init__()
        # encode to small fixed vectors
        self.venc = SmallEncoder(video_dim, 128)
        self.aenc = SmallEncoder(audio_dim, 64)
        self.tenc = SmallEncoder(text_dim, 128)

        fuse = 128 + 64 + 128
        self.classifier = nn.Sequential(
            nn.Linear(fuse, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, video, audio, text):
        v = self.venc(video)
        a = self.aenc(audio)
        t = self.tenc(text)
        x = torch.cat([v, a, t], dim=1)
        return self.classifier(x)
