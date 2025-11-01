"""
Reads dataset/transcripts.csv and stores one .npy per utterance KEY in:
  dataset/features/text/<KEY>.npy

Uses a small sentence transformer (HF). Default model: all-MiniLM-L6-v2
Usage:
  python text_to_embeddings.py --csv ../../dataset/transcripts.csv --out ../../dataset/features/text
Requires: transformers
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

def embed_batch(texts, tokenizer, model, device):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, return_dict=True)
        last = out.last_hidden_state  # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        summed = (last * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = (summed / counts).cpu().numpy()
    return pooled

def main(csv_path, out_dir, id_col="KEY", text_col="SENTENCE", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if id_col not in df.columns or text_col not in df.columns:
        raise KeyError("CSV columns not found; edit parameters")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    batch = []
    ids  = []
    bs = 32
    for i, row in df.iterrows():
        uid = str(row[id_col])
        text = str(row[text_col]) if not pd.isna(row[text_col]) else ""
        ids.append(uid)
        batch.append(text)
        if len(batch) == bs:
            embs = embed_batch(batch, tokenizer, model, device)
            for u, e in zip(ids, embs):
                np.save(os.path.join(out_dir, f"{u}.npy"), e.astype(np.float32))
            batch, ids = [], []
    if batch:
        embs = embed_batch(batch, tokenizer, model, device)
        for u, e in zip(ids, embs):
            np.save(os.path.join(out_dir, f"{u}.npy"), e.astype(np.float32))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join("..","..","dataset","transcripts.csv"))
    parser.add_argument("--out", default=os.path.join("..","..","dataset","features","text"))
    parser.add_argument("--id_col", default="KEY")
    parser.add_argument("--text_col", default="SENTENCE")
    args = parser.parse_args()
    main(args.csv, args.out, id_col=args.id_col, text_col=args.text_col)
