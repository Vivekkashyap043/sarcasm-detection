"""Simple inference runner for a saved checkpoint.

This script expects precomputed .npy features for each modality. It supports
video-only, text-only or multimodal inputs depending on flags.

Example (video-only):
  python run_inference.py --video features/video/utterance/123.npy --ckpt ../checkpoints/best.pth

Example (text-only):
  python run_inference.py --text features/text/123.npy --use_text --ckpt ../checkpoints/best.pth
"""

import argparse
import os
import numpy as np
import torch

from models.multimodal_model import MultimodalClassifier


def load_npy(path):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.load(path).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)  # add batch dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help=".npy file for video feature (concatenated context+utt)")
    parser.add_argument("--audio", default=None, help=".npy file for audio feature")
    parser.add_argument("--text", default=None, help=".npy file for text embedding")
    parser.add_argument("--use_audio", action="store_true")
    parser.add_argument("--use_text", action="store_true")
    # allow disabling video encoder for pure-text or pure-audio inference
    parser.add_argument("--no_video", dest="use_video", action="store_false",
                        help="Disable video input / encoder (use when running text-only inference)")
    parser.set_defaults(use_video=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # load only the modalities the user requested
    v = load_npy(args.video) if args.use_video else None
    a = load_npy(args.audio) if args.use_audio else None
    t = load_npy(args.text) if args.use_text else None

    # if user expects video but didn't provide file -> error
    if args.use_video and v is None:
        raise ValueError("video input is required for --use_video (provide --video path or pass --no_video to disable)")

    # infer dims
    video_dim = v.shape[1] if v is not None else None
    audio_dim = a.shape[1] if a is not None else None
    text_dim  = t.shape[1] if t is not None else None

    device = torch.device(args.device)

    model = MultimodalClassifier(video_dim=video_dim if args.use_video else None,
                                 audio_dim=audio_dim if args.use_audio else None,
                                 text_dim=text_dim if args.use_text else None,
                                 use_audio=args.use_audio,
                                 use_text=args.use_text)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    v = v.to(device) if v is not None else None
    a = a.to(device) if a is not None else None
    t = t.to(device) if t is not None else None

    labels, probs = model.predict(video=v if args.use_video else None,
                                   audio=a if args.use_audio else None,
                                   text=t if args.use_text else None)
    label = int(labels.cpu().numpy()[0])
    prob = float(probs.cpu().numpy()[0, label])
    print(f"predicted={label} prob={prob:.4f}")


if __name__ == "__main__":
    main()
