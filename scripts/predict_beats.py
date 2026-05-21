import sys, os
sys.path.insert(0, '/root/asc_evaluation')

import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from beats.BEATs import BEATs, BEATsConfig

DEVICE = torch.device("cpu")
CHECKPOINT_PRETRAIN = "models/BEATs_iter3_plus_AS2M.pt"
CHECKPOINT_FINETUNED = "models/beats_finetuned_best.pt"
SAMPLING_RATE = 16000
MAX_DURATION = 5.0
DATA_DIR = "data/raw"
OUTPUT_CSV = "results/predictions_beats.csv"

# 加载预训练权重构建模型结构
pretrain = torch.load(CHECKPOINT_PRETRAIN, map_location='cpu')
cfg = BEATsConfig(pretrain['cfg'])
model = BEATs(cfg)
model.load_state_dict(pretrain['model'], strict=False)

# 加载微调后的最佳模型
finetuned = torch.load(CHECKPOINT_FINETUNED, map_location='cpu')
model.load_state_dict(finetuned['model_state'])
model.to(DEVICE)
model.eval()

# 分类器
in_features = 768  # 已知
num_labels = len(finetuned['id2label'])
classifier = nn.Linear(in_features, num_labels).to(DEVICE)
classifier.load_state_dict(finetuned['classifier_state'])
classifier.eval()

id2label = finetuned['id2label']

def get_features(model, x):
    out = model.extract_features(x)
    if isinstance(out, tuple):
        feat = out[0]
    elif isinstance(out, dict):
        feat = list(out.values())[0]
    else:
        feat = out
    if feat.dim() == 3:
        feat = feat.mean(dim=1)
    return feat

# 读取测试文件列表（从标注文件获取所有测试音频文件名）
test_df = pd.read_csv('data/test.csv')  # 之前划分时生成的
results = []
for fname in tqdm(test_df['filename'], desc="推理中"):
    file_path = os.path.join(DATA_DIR, fname)
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)
        target_len = int(SAMPLING_RATE * MAX_DURATION)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        input_tensor = torch.from_numpy(audio).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            feat = get_features(model, input_tensor)
            logits = classifier(feat)
        pred_id = torch.argmax(logits, dim=1).item()
        pred_label = id2label[pred_id]
        results.append({'filename': fname, 'pred_label': pred_label})
    except Exception as e:
        print(f"Error {fname}: {e}")
        results.append({'filename': fname, 'pred_label': 'unknown'})

df_pred = pd.DataFrame(results)
df_pred.to_csv(OUTPUT_CSV, index=False)
print(f"预测完成，保存至 {OUTPUT_CSV}")
