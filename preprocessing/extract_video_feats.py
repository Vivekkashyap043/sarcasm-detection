"""
Extract compact video features (ResNet50) for all videos in:
  dataset/video/context_videos
  dataset/video/utterance_videos

Saves one .npy per input video in:
  dataset/features/video/context/<basename>.npy
  dataset/features/video/utterance/<basename>.npy

Usage:
  cd MPP_Code/preprocessing
  python extract_video_feats.py --video_root ../../dataset/video --out_root ../../dataset/features/video
"""

import os
import glob
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm

def sample_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, max(0,total-1), num_frames).astype(int)
    idx_set = set(indices.tolist())
    cur = 0
    grabbed = 0
    while cap.isOpened() and grabbed < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if cur in idx_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            grabbed += 1
        cur += 1
    cap.release()
    return frames

def extract_features(video_path, model, device, num_frames=8):
    frames = sample_frames(video_path, num_frames=num_frames)
    if len(frames) == 0:
        return None
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor()
    ])
    batch = torch.stack([transform(f) for f in frames]).to(device)
    with torch.no_grad():
        feats = model(batch).cpu().numpy()  # (num_frames, feat_dim)
    # mean pool across frames -> (feat_dim,)
    return feats.mean(axis=0).astype(np.float32)

def process_kind(kind_dir, out_dir, model, device):
    os.makedirs(out_dir, exist_ok=True)
    for p in tqdm(sorted(glob.glob(os.path.join(kind_dir, "*"))), desc=f"Processing {os.path.basename(kind_dir)}"):
        if not p.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue
        basename = os.path.splitext(os.path.basename(p))[0]
        out_file = os.path.join(out_dir, basename + ".npy")
        if os.path.exists(out_file):
            continue
        feat = extract_features(p, model, device)
        if feat is None:
            print("Warning empty", p)
            continue
        np.save(out_file, feat)

def main(video_root, out_root):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ResNet50 backbone, remove final fc
    resnet = models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device).eval()

    # map of source subfolders -> destination subfolders
    mapping = {
        "context_videos": os.path.join(out_root, "context"),
        "utterance_videos": os.path.join(out_root, "utterance")
    }

    for src_sub, dst_sub in mapping.items():
        src_dir = os.path.join(video_root, src_sub)
        if not os.path.isdir(src_dir):
            print("Skipping missing dir:", src_dir)
            continue
        process_kind(src_dir, dst_sub, resnet, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", default=os.path.join("..","..","dataset","video"))
    parser.add_argument("--out_root", default=os.path.join("..","..","dataset","features","video"))
    args = parser.parse_args()
    main(args.video_root, args.out_root)
