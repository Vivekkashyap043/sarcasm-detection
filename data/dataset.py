"""
Multimodal dataset loader (video-first). Designed for your transcripts.csv with columns:
  SCENE, KEY, SENTENCE, Sarcasm

Behavior:
- Loads cached features from dataset/features/video/context/<scene>.npy and
  dataset/features/video/utterance/<key>.npy
- Optionally loads audio from dataset/features/audio/utterance/<key>.npy
- Optionally loads text embeddings from dataset/features/text/<key>.npy
- If some modality is missing it supplies a zero vector of the expected dimension.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# --- CSV columns ---
CSV_PATH = os.path.join("dataset", "transcripts.csv")
SCENE_COL = "SCENE"
KEY_COL = "KEY"
TEXT_COL = "SENTENCE"
LABEL_COL = "Sarcasm"
# ---------------------------------------------------------------

# feature folders (created by preprocessing)
VIDEO_FEATURE_DIR = os.path.join("dataset", "features", "video")  # contains 'context' and 'utterance'
AUDIO_FEATURE_DIR = os.path.join("dataset", "features", "audio", "utterance")
TEXT_FEATURE_DIR  = os.path.join("dataset", "features", "text")

def _safe_load_npy(path):
    if not os.path.exists(path):
        return None
    return np.load(path)

def _try_paths(base_dir, name):
    """Try common filename patterns for `name`."""
    candidates = [
        os.path.join(base_dir, name + ".npy"),
        os.path.join(base_dir, name)  # if name already contains extension or path
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


class MultimodalDataset(Dataset):
    def __init__(self, csv_path=CSV_PATH, use_audio=False, use_text=False):
        self.df = pd.read_csv(csv_path)
        for col in [SCENE_COL, KEY_COL, TEXT_COL, LABEL_COL]:
            if col not in self.df.columns:
                raise KeyError(f"Expected column '{col}' in CSV")

        self.use_audio = use_audio
        self.use_text = use_text

        # detect dims from first available feature (otherwise will set fallback dims)
        self._video_context_dim = None
        self._video_utt_dim = None
        self._audio_dim = None
        self._text_dim = None

        # probe feature folders for any example to infer dims
        # video context
        ctx_dir = os.path.join(VIDEO_FEATURE_DIR, "context")
        utt_dir = os.path.join(VIDEO_FEATURE_DIR, "utterance")
        if os.path.isdir(ctx_dir):
            for f in os.listdir(ctx_dir):
                if f.endswith(".npy"):
                    v = np.load(os.path.join(ctx_dir, f))
                    self._video_context_dim = v.reshape(-1).shape[0]
                    break
        if os.path.isdir(utt_dir):
            for f in os.listdir(utt_dir):
                if f.endswith(".npy"):
                    v = np.load(os.path.join(utt_dir, f))
                    self._video_utt_dim = v.reshape(-1).shape[0]
                    break
        if self.use_audio and os.path.isdir(AUDIO_FEATURE_DIR):
            for f in os.listdir(AUDIO_FEATURE_DIR):
                if f.endswith(".npy"):
                    a = np.load(os.path.join(AUDIO_FEATURE_DIR, f))
                    self._audio_dim = a.reshape(-1).shape[0]
                    break
        if self.use_text and os.path.isdir(TEXT_FEATURE_DIR):
            for f in os.listdir(TEXT_FEATURE_DIR):
                if f.endswith(".npy"):
                    t = np.load(os.path.join(TEXT_FEATURE_DIR, f))
                    self._text_dim = t.reshape(-1).shape[0]
                    break

        # set default fallback dims if None (small vectors)
        if self._video_context_dim is None:
            self._video_context_dim = 2048  # default ResNet50 feature size
        if self._video_utt_dim is None:
            self._video_utt_dim = 2048
        if self._audio_dim is None:
            self._audio_dim = 64
        if self._text_dim is None:
            self._text_dim = 384

    @property
    def video_dim(self):
        """Dimension of the concatenated video feature (context + utterance)"""
        return int(self._video_context_dim + self._video_utt_dim)

    @property
    def audio_dim(self):
        return int(self._audio_dim)

    @property
    def text_dim(self):
        return int(self._text_dim)

    def __len__(self):
        return len(self.df)

    def _load_video_feature(self, kind, name):
        base = os.path.join(VIDEO_FEATURE_DIR, kind)  # kind = "context" or "utterance"
        candidate = _try_paths(base, name)
        if candidate:
            return np.load(candidate)
        # fallback: try without any suffix if name includes an indicator
        # (this is a last resort and returns zeros if nothing found)
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        scene = str(row[SCENE_COL])
        key   = str(row[KEY_COL])
        label = int(row[LABEL_COL]) if str(row[LABEL_COL]).strip() != "" else -1

        # 1) load context video feature (try SCENE)
        v_ctx = self._load_video_feature("context", scene)
        # 2) load utterance video feature (try KEY)
        v_utt = self._load_video_feature("utterance", key)

        # audio and text (optional)
        a_utt = None
        t_emb = None
        if self.use_audio:
            a_path = _try_paths(AUDIO_FEATURE_DIR, key)
            if a_path:
                a_utt = np.load(a_path)
        if self.use_text:
            t_path = _try_paths(TEXT_FEATURE_DIR, key)
            if t_path:
                t_emb = np.load(t_path)

        # If any modality missing -> fill zeros to preserve shape
        if v_ctx is None:
            v_ctx = np.zeros(self._video_context_dim, dtype=np.float32)
        else:
            v_ctx = v_ctx.reshape(-1).astype(np.float32)
        if v_utt is None:
            v_utt = np.zeros(self._video_utt_dim, dtype=np.float32)
        else:
            v_utt = v_utt.reshape(-1).astype(np.float32)
        if a_utt is None:
            a_utt = np.zeros(self._audio_dim, dtype=np.float32)
        else:
            a_utt = a_utt.reshape(-1).astype(np.float32)
        if t_emb is None:
            t_emb = np.zeros(self._text_dim, dtype=np.float32)
        else:
            t_emb = t_emb.reshape(-1).astype(np.float32)

        # combine video context + utterance (concatenate)
        video_feat = np.concatenate([v_ctx, v_utt], axis=-1).astype(np.float32)

        return {
            "video": torch.from_numpy(video_feat),
            "audio": torch.from_numpy(a_utt),
            "text":  torch.from_numpy(t_emb),
            "label": torch.tensor(label, dtype=torch.long)
        }
