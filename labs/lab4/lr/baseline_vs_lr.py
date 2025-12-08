import os
import matplotlib.pyplot as plt

# Data
labels = ["Baseline", "LR"]
values = [0.82, 0.82]

plt.figure(figsize=(6, 4))
plt.bar(labels, values)

plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

# Ensure images directory exists and save the figure
os.makedirs("images", exist_ok=True)
plt.savefig("images/baseline_vs_lr_accuracy.png", bbox_inches="tight")
plt.show()
