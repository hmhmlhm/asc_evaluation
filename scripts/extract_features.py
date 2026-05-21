import librosa
import numpy as np
import pandas as pd
import os, glob, json

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

audio_files = glob.glob(os.path.join(RAW_DIR, "*.wav"))
print(f"发现 {len(audio_files)} 个音频文件")

all_features = []
for file in audio_files:
    try:
        y, sr = librosa.load(file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        all_features.append({
            "filename": os.path.basename(file),
            "features": mfcc_mean.tolist()
        })
    except Exception as e:
        print(f"处理 {file} 时出错: {e}")

with open(os.path.join(PROCESSED_DIR, "features.json"), "w") as f:
    json.dump(all_features, f, indent=2)
print(f"特征已保存到 {PROCESSED_DIR}/features.json")
