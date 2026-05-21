import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 读取官方标注（制表符分隔，无表头）
cols = ['filename', 'label', 'device', 'unknown']
train_full = pd.read_csv('data/evaluation_setup/development_train.txt', sep='\t', header=None, names=cols)
test = pd.read_csv('data/evaluation_setup/development_test.txt', sep='\t', header=None, names=cols)

# 提取纯文件名（去掉 audio/ 前缀），方便后续加载
train_full['filename'] = train_full['filename'].apply(lambda x: os.path.basename(x))
test['filename'] = test['filename'].apply(lambda x: os.path.basename(x))

# 划分训练集和验证集（按类别分层抽样）
train, val = train_test_split(
    train_full,
    test_size=0.2,
    random_state=42,
    stratify=train_full['label']
)

# 保存为 CSV（只需 filename 和 label 两列）
train[['filename', 'label']].to_csv('data/train.csv', index=False)
val[['filename', 'label']].to_csv('data/val.csv', index=False)
test[['filename', 'label']].to_csv('data/test.csv', index=False)

print(f"训练集: {len(train)} 条")
print(f"验证集: {len(val)} 条")
print(f"测试集: {len(test)} 条")
