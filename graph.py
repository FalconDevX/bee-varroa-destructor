import pandas as pd
import matplotlib.pyplot as plt

df_full = pd.read_csv("results.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(df_full["epoch"], df_full["train/box_loss"], label="Box loss")
axes[0, 0].plot(df_full["epoch"], df_full["train/cls_loss"], label="Cls loss")
axes[0, 0].plot(df_full["epoch"], df_full["train/dfl_loss"], label="DFL loss")
axes[0, 0].set_title("Training losses")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(df_full["epoch"], df_full["val/box_loss"], label="Val Box loss")
axes[0, 1].plot(df_full["epoch"], df_full["val/cls_loss"], label="Val Cls loss")
axes[0, 1].plot(df_full["epoch"], df_full["val/dfl_loss"], label="Val DFL loss")
axes[0, 1].set_title("Validation losses")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(df_full["epoch"], df_full["metrics/precision(B)"], label="Precision")
axes[1, 0].plot(df_full["epoch"], df_full["metrics/recall(B)"], label="Recall")
axes[1, 0].set_title("Precision & Recall")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Score")
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(df_full["epoch"], df_full["metrics/mAP50(B)"], label="mAP50")
axes[1, 1].plot(df_full["epoch"], df_full["metrics/mAP50-95(B)"], label="mAP50-95")
axes[1, 1].set_title("mAP metrics")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Score")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
