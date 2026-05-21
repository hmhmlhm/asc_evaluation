import os, glob, librosa, numpy as np, pandas as pd
from panns_inference import AudioTagging

print("正在加载 PANNs CNN14 模型...")
at = AudioTagging(checkpoint_path=None, device='cpu')
print("模型加载完成")

# 修正后的10类映射（与你数据集真实标签匹配）
label_map = {
    'alarm': 'Alarm',
    'baby_cry': 'Baby cry, infant cry',
    'dog_bark': 'Dog',
    'engine': 'Engine',
    'fire': 'Fire',
    'footsteps': 'Footsteps',
    'knocking': 'Knock',
    'piano': 'Piano',
    'speech': 'Speech',
    'telephone_ringing': 'Telephone bell ringing'
}

inv_map = {v: k for k, v in label_map.items()}

results = []
audio_files = glob.glob('data/raw/*.wav')
total = len(audio_files)
print(f"找到 {total} 个音频文件，开始预测...")

for idx, file in enumerate(audio_files):
    try:
        audio, _ = librosa.load(file, sr=32000, mono=True, res_type='kaiser_fast')
        audio = audio[None, :]
        clipwise_out, _ = at.inference(audio)
        best_idx = np.argmax(clipwise_out[0])
        best_label_name = at.labels[best_idx]
        if best_label_name in inv_map:
            pred_label = inv_map[best_label_name]
        else:
            pred_label = 'unknown'
        results.append({
            'filename': os.path.basename(file),
            'pred_label': pred_label,
            'confidence': float(clipwise_out[0, best_idx])
        })
        if (idx+1) % 500 == 0:
            print(f"已处理 {idx+1}/{total}")
    except Exception as e:
        print(f"处理 {file} 出错: {e}")
        continue

df_pred = pd.DataFrame(results)
df_pred.to_csv('results/predictions_panns.csv', index=False)
print(f"预测完成，共生成 {len(df_pred)} 条记录，保存在 results/predictions_panns.csv")
