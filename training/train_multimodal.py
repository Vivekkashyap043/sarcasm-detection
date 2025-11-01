"""
Training script for the multimodal classifier.

Works with video-only (audio/text optional). It infers dims from dataset sample.
Run:
  python train_multimodal.py --csv ../../dataset/transcripts.csv --epochs 10 --batch_size 16
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score

from data.dataset import MultimodalDataset
from models.multimodal_model import MultimodalClassifier

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def collate_batch(batch):
    vids = torch.stack([b["video"] for b in batch])
    auds = torch.stack([b["audio"] for b in batch])
    txts = torch.stack([b["text"] for b in batch])
    labs = torch.stack([b["label"] for b in batch])
    return vids, auds, txts, labs

def train(args):
    set_seed(args.seed)
    ds = MultimodalDataset(csv_path=args.csv, use_audio=args.use_audio, use_text=args.use_text)
    # remove rows with label -1 (if any)
    indices = [i for i in range(len(ds)) if int(ds.df.iloc[i][args.label_col]) in [0,1]]
    if len(indices) < len(ds):
        from torch.utils.data import Subset
        ds = Subset(ds, indices)

    n = len(ds)
    val_n = max(1, int(n * args.val_split))
    train_n = n - val_n
    train_set, val_set = random_split(ds, [train_n, val_n])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=2)

    # infer dims from one example
    sample = ds[0] if not isinstance(ds, torch.utils.data.Subset) else ds.dataset[ds.indices[0]]
    video_dim = sample["video"].shape[0]
    audio_dim = sample["audio"].shape[0]
    text_dim  = sample["text"].shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalClassifier(video_dim, audio_dim, text_dim, hidden=args.hidden, num_classes=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for vids, auds, txts, labs in train_loader:
            vids, auds, txts, labs = vids.to(device), auds.to(device), txts.to(device), labs.to(device)
            logits = model(vids, auds, txts)
            loss = criterion(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for vids, auds, txts, labs in val_loader:
                vids, auds, txts, labs = vids.to(device), auds.to(device), txts.to(device), labs.to(device)
                logits = model(vids, auds, txts)
                p = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(p.tolist()); trues.extend(labs.cpu().numpy().tolist())

        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average="macro")
        print(f"Epoch {epoch} loss={np.mean(losses):.4f} val_acc={acc:.4f} val_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
            torch.save(model.state_dict(), args.ckpt)
            print("Saved best:", args.ckpt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join("..","..","dataset","transcripts.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--use_audio", action="store_true")
    parser.add_argument("--use_text", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", default=os.path.join("..","..","checkpoints","best.pth"))
    parser.add_argument("--label_col", default="Sarcasm")
    args = parser.parse_args()
    train(args)
