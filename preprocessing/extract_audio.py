
"""
Extract .wav audio tracks from videos using ffmpeg.
Produces:
  dataset/audio/context_audio/<basename>.wav
  dataset/audio/utterance_audio/<basename>.wav
Usage:
  python extract_audio.py --video_root ../../dataset/video --out_root ../../dataset/audio
"""

import os
import subprocess
import argparse

def extract(video_path, out_wav, sr=16000):
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
        out_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def walk_and_extract(video_root, out_root, sr=16000):
    for root, _, files in os.walk(video_root):
        for f in files:
            if not f.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
                continue
            inpath = os.path.join(root, f)
            rel = os.path.relpath(root, video_root)
            outdir = os.path.join(out_root, rel.replace("videos", "audio"))
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(outdir, os.path.splitext(f)[0] + ".wav")
            print(f"Extracting {inpath} -> {outpath}")
            extract(inpath, outpath, sr=sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", default=os.path.join("..","..","dataset","video"))
    parser.add_argument("--out_root", default=os.path.join("..","..","dataset","audio"))
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()
    walk_and_extract(args.video_root, args.out_root, sr=args.sr)
