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
import torch.nn.functional as F


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
    """Flexible multimodal classifier.

    Supports enabling/disabling audio/text encoders so the same class can be used
    for video-only, text-only or multimodal training/inference.

    Args:
      video_dim: int or None
      audio_dim: int or None
      text_dim: int or None
      use_audio: bool
      use_text: bool
    """

    def __init__(self, video_dim=None, audio_dim=None, text_dim=None,
                 use_audio=False, use_text=False, hidden=256, num_classes=2):
        super().__init__()
        self.use_audio = use_audio
        self.use_text = use_text

        # create encoders only for enabled modalities
        self.venc = None
        self.aenc = None
        self.tenc = None
        fuse_size = 0

        if video_dim is not None:
            self.venc = SmallEncoder(video_dim, 128)
            fuse_size += 128

        if self.use_audio:
            if audio_dim is None:
                raise ValueError("use_audio=True but audio_dim is None")
            self.aenc = SmallEncoder(audio_dim, 64)
            fuse_size += 64

        if self.use_text:
            if text_dim is None:
                raise ValueError("use_text=True but text_dim is None")
            self.tenc = SmallEncoder(text_dim, 128)
            fuse_size += 128

        if fuse_size == 0:
            raise ValueError("At least one modality must be enabled/provided")

        self.classifier = nn.Sequential(
            nn.Linear(fuse_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, video=None, audio=None, text=None):
        parts = []
        # video may be provided even if not using audio/text
        if self.venc is not None:
            if video is None:
                raise ValueError("video input is required by model but got None")
            parts.append(self.venc(video))

        if self.aenc is not None:
            if audio is None:
                # allow zeros but prefer explicit input
                raise ValueError("audio input is required by model but got None")
            parts.append(self.aenc(audio))

        if self.tenc is not None:
            if text is None:
                raise ValueError("text input is required by model but got None")
            parts.append(self.tenc(text))

        x = torch.cat(parts, dim=1)
        return self.classifier(x)

    def predict_proba(self, video=None, audio=None, text=None):
        """Return softmax probabilities for inputs."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(video=video, audio=audio, text=text)
            probs = F.softmax(logits, dim=1)
        return probs

    def predict(self, video=None, audio=None, text=None):
        probs = self.predict_proba(video=video, audio=audio, text=text)
        labels = torch.argmax(probs, dim=1)
        return labels, probs
