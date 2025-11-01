"""
Compute log-mel mean vector per utterance WAV and save to:
  dataset/features/audio/utterance/<basename>.npy

Usage:
  python extract_audio_feats.py --audio_root ../../dataset/audio --out_root ../../dataset/features/audio
Requires: librosa
"""

import os
import glob
import argparse
import numpy as np
import librosa

def compute_logmelspec_mean(wav_path, sr=16000, n_mels=64, hop_length=256):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    if y.size == 0:
        return None
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # mean across time -> (n_mels,)
    return log_mel.mean(axis=1).astype(np.float32)

def main(audio_root, out_root):
    src_dir = os.path.join(audio_root, "utterance_audio")
    dst_dir = os.path.join(out_root, "utterance")
    os.makedirs(dst_dir, exist_ok=True)
    for wav in sorted(glob.glob(os.path.join(src_dir, "*.wav"))):
        base = os.path.splitext(os.path.basename(wav))[0]
        outp = os.path.join(dst_dir, base + ".npy")
        if os.path.exists(outp):
            continue
        feat = compute_logmelspec_mean(wav)
        if feat is None:
            print("Empty:", wav); continue
        np.save(outp, feat)
        print("Saved", outp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root", default=os.path.join("..","..","dataset","audio"))
    parser.add_argument("--out_root", default=os.path.join("..","..","dataset","features","audio"))
    args = parser.parse_args()
    main(args.audio_root, args.out_root)
