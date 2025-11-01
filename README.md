# Sarcasm Detection (multimodal)

This repo contains a simple multimodal sarcasm detection pipeline. It supports:
- Video-only inference/training (precomputed video features)
- Text-only inference/training (text embeddings)
- Multimodal training combining video/audio/text features

Quick start
1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Precompute features:
- Video features: `preprocessing/extract_video_feats.py` -> writes to `dataset/features/video`
- Text embeddings: `preprocessing/text_to_embeddings.py` -> writes to `dataset/features/text`

3. Train

Video-only (default):

```powershell
python training/train_multimodal.py --csv dataset/video/transcripts.csv --epochs 10 --batch_size 16
```

Text-only (disable video encoder + enable text):

```powershell
python training/train_multimodal.py --csv dataset/video/transcripts.csv --no_video --use_text --epochs 10 --batch_size 32
```

Multimodal (video + text, optionally audio):

```powershell
python training/train_multimodal.py --csv dataset/video/transcripts.csv --use_text --use_audio --epochs 10 --batch_size 16
```

4. Inference / Test

Video-only inference (provide concatenated video .npy feature):

```powershell
python training/run_inference.py --video dataset/features/video/utterance/1_10004_u.npy --ckpt checkpoints/best.pth
```

Text-only inference:

```powershell
python training/run_inference.py --text dataset/features/text/1_10004_u.npy --no_video --use_text --ckpt checkpoints/best.pth
```

If you want to run a quick evaluation loop on a test CSV, you can load the dataset with the same `--no_video/--use_text` flags used during training and iterate through the samples calling the model's `predict` to collect metrics.

Notes
- `training/train_multimodal.py` now supports `--use_audio` and `--use_text` flags to enable audio/text modalities.
- `models/multimodal_model.py` is flexible and can be used for single or multi-modal inference.

If you want a web UI to accept raw videos, you'll need to connect an endpoint that runs `preprocessing/extract_video_feats.extract_features` on the uploaded video and then call `training/run_inference.py` (or load the model in-process).