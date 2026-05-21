#!/bin/bash
echo "===== 1. 提取音频特征 ====="
python3 scripts/extract_features.py

echo "===== 2. 评估模型 ====="
python3 scripts/evaluate.py

echo "===== 3. 生成可视化报告 ====="
python3 scripts/visualize.py

echo "===== 完成！ ====="
