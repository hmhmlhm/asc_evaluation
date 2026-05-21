import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

df = pd.read_csv("results/predictions.csv")
cm = confusion_matrix(df['true_label'], df['pred_label'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Sound Event Classification Confusion Matrix")
plt.savefig("results/figures/confusion_matrix.png", dpi=150)
print("混淆矩阵已保存到 results/figures/confusion_matrix.png")
