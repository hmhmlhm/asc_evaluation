import sys, os
sys.path.insert(0, '/root/asc_evaluation')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa, numpy as np, pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from beats.BEATs import BEATs, BEATsConfig

# ---------- 配置 ----------
DEVICE = torch.device("cpu")
CHECKPOINT_PATH = "models/BEATs_iter3_plus_AS2M.pt"
SAMPLING_RATE = 16000
MAX_DURATION = 5.0
BATCH_SIZE = 2
EPOCHS_STAGE1 = 2
EPOCHS_STAGE2 = 3
LR_STAGE1 = 1e-4
LR_STAGE2 = 1e-5
DATA_DIR = "data/raw"

# ---------- 加载本地 BEATs 模型 ----------
print("加载本地BEATs模型...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
cfg = BEATsConfig(checkpoint['cfg'])
model = BEATs(cfg)
model.load_state_dict(checkpoint['model'], strict=False)
model.to(DEVICE)
print("模型加载成功。")

# ---------- 自定义特征提取器（实际上 BEATs 内部已包含预处理，这里不用） ----------
class SimpleFeatureExtractor:
    def __call__(self, audio, sampling_rate, return_tensors="pt"):
        input_values = torch.from_numpy(audio).unsqueeze(0).float()
        return {"input_values": input_values}

processor = SimpleFeatureExtractor()

# ---------- 数据集 ----------
class AudioDataset(Dataset):
    def __init__(self, csv_path, label2id, processor):
        self.df = pd.read_csv(csv_path)
        self.label2id = label2id
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(DATA_DIR, row['filename'])
        try:
            audio, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)
            target_len = int(SAMPLING_RATE * MAX_DURATION)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
            inputs = self.processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
            input_values = inputs["input_values"].squeeze(0)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            input_values = torch.zeros(int(SAMPLING_RATE * MAX_DURATION))
        label = self.label2id[row['label']]
        return input_values, label

# ---------- 标签映射 ----------
all_labels = sorted(pd.read_csv('data/train.csv')['label'].unique())
label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(label2id)
print("类别数:", num_labels)

train_dataset = AudioDataset('data/train.csv', label2id, processor)
val_dataset = AudioDataset('data/val.csv', label2id, processor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- 外部分类头 ----------
print("探测模型输出维度...")
model.eval()
with torch.no_grad():
    dummy_input = torch.randn(1, int(SAMPLING_RATE * MAX_DURATION)).to(DEVICE)
    # 关键修改：使用 extract_features
    dummy_output = model.extract_features(dummy_input)
    # 处理返回值（通常是 (features, padding_mask) 或 (features, layer_results)）
    if isinstance(dummy_output, tuple):
        dummy_feat = dummy_output[0]
    elif isinstance(dummy_output, dict):
        if 'x' in dummy_output:
            dummy_feat = dummy_output['x']
        elif 'features' in dummy_output:
            dummy_feat = dummy_output['features']
        else:
            dummy_feat = list(dummy_output.values())[0]
    else:
        dummy_feat = dummy_output
    in_features = dummy_feat.shape[-1]
    print(f"模型输出特征维度: {in_features}")

classifier = nn.Linear(in_features, num_labels).to(DEVICE)

# ---------- 提取特征的通用函数 ----------
def get_features(model, x):
    out = model.extract_features(x)
    if isinstance(out, tuple):
        feat = out[0]
    elif isinstance(out, dict):
        if 'x' in out:
            feat = out['x']
        elif 'features' in out:
            feat = out['features']
        else:
            feat = list(out.values())[0]
    else:
        feat = out
    # 如果特征是 3D (batch, time, feature_dim)，进行时间平均池化
    if feat.dim() == 3:
        feat = feat.mean(dim=1)   # 得到 (batch, feature_dim)
    return feat

# ---------- 阶段1：仅训练分类头 ----------
print("阶段1：训练新分类头...")
for param in model.parameters():
    param.requires_grad = False

optimizer = optim.Adam(classifier.parameters(), lr=LR_STAGE1)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS_STAGE1):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_STAGE1}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        feat = get_features(model, inputs)
        logits = classifier(feat)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    print(f"Train Loss: {total_loss/len(train_loader):.4f}")

# ---------- 阶段2：端到端微调 ----------
print("阶段2：端到端微调...")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr=LR_STAGE2
)

best_val_acc = 0.0
for epoch in range(EPOCHS_STAGE2):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_STAGE2}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        feat = get_features(model, inputs)
        logits = classifier(feat)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            feat = get_features(model, inputs)
            logits = classifier(feat)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state': model.state_dict(),
            'classifier_state': classifier.state_dict(),
            'label2id': label2id,
            'id2label': id2label
        }, "models/beats_finetuned_best.pt")
        print("最佳模型已保存")

print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")
