import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("results/predictions.csv")
print(f"共 {len(df)} 条预测记录")

acc = accuracy_score(df['true_label'], df['pred_label'])
print(f"整体准确率: {acc:.2%}")

report = classification_report(df['true_label'], df['pred_label'], output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("results/classification_report.csv")
print("分类报告已保存到 results/classification_report.csv")
