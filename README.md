# 自动化声学场景分类评估系统

> 基于 BEATs 预训练模型，针对 DCASE 2026 声音事件分类任务，设计两阶段微调策略，实现高精度分类，并构建一键式自动化评估流水线。
>
> 本文采用的模型为:
> [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)
> [unilm/beats at master · microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beats)

## 🔥 项目亮点

- **零样本基线**：使用 PANNs 通用音频分类模型直接推理，准确率 **29.6%**，证明任务特殊性
- **迁移学习**：BEATs 两阶段微调（冻结主干 + 端到端微调），测试准确率 **88.1%**，提升幅度 **58.5 个百分点**
- **自动化评估**：Shell 脚本一键完成特征提取、模型评估、混淆矩阵可视化
- **数据库对比**：结果存入 MySQL，支持多模型指标横向对比
- **工程化实践**：规范的目录结构、版本控制、模型管理

## 📁 项目结构

```
asc_evaluation/
├── data/
│ ├── raw/ # 原始音频文件（需自行下载）
│ ├── processed/ # 提取的特征（features.json）
│ └── evaluation_setup/ # 官方数据划分与标注
├── models/ # 预训练权重与微调模型（需自行下载）
├── results/
│ ├── predictions.csv # 最终预测结果
│ ├── classification_report.csv
│ └── figures/ # 可视化图表
├── scripts/
│ ├── extract_features.py # 特征提取（MFCC）
│ ├── evaluate.py # 模型评估
│ ├── visualize.py # 混淆矩阵绘制
│ ├── predict_with_panns.py # PANNs 零样本推理
│ ├── train_beats.py # BEATs 微调训练
│ └── predict_beats.py # BEATs 推理
├── run_all.sh # 一键运行全流程
└── README.md
```

## ⚙️ 环境依赖

```
- Python 3.6+
- PyTorch 1.10+
- librosa, pandas, numpy, scikit-learn, matplotlib
- panns-inference (可选，用于 PANNs 基线)
- MySQL 8.0（用于多模型对比存储）
```

## 📊 数据准备

1. 下载 DCASE 2026 Task7 官方数据集，将 `.wav` 音频文件放入 `data/raw/`
2. `data/evaluation_setup/` 中已包含官方训练/测试划分文件

## 🚀 快速开始

1. **提取特征 + 评估 + 可视化（一键运行）**

   bash

   ```
   chmod +x run_all.sh
   ./run_all.sh
   ```

   输出：

   - 特征文件：`data/processed/features.json`
   - 评估报告：`results/classification_report.csv`
   - 混淆矩阵：`results/figures/confusion_matrix.png`

2. **使用 BEATs 微调模型进行预测**

   bash

   ```
   python scripts/predict_beats.py
   ```

   生成 `results/predictions_beats.csv`，然后合并真实标签并重新运行 `./run_all.sh` 获得最终测试准确率。

## 📈 实验结果

![confusion_matrix](C:\Users\30569\Desktop\asc_evaluation\results\figures\confusion_matrix.png)

| 模型             | 方式                   | 测试准确率 |
| :--------------- | :--------------------- | :--------- |
| PANNs            | 零样本推理（直接预测） | 29.64%     |
| **BEATs (ours)** | **两阶段微调**         | **88.10%** |

*注：BEATs 基准为随机分类头（≈10%），微调带来约 78 个百分点提升。*

## 👤 作者

- GitHub: [hmhmlhm](https://github.com/hmhmlhm)
- 项目链接: https://github.com/hmhmlhm/asc_evaluation
