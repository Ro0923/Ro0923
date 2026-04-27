import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import RWF2000Dataset
from model import ViolenceNet

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
DATA_DIR = "Processed_Clean"
BATCH_SIZE = 8
MODEL_PATH = "best_model.pt"
THRESHOLD = 0.7
OUT_DIR = "eval_plots"
# ---------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- DATA ----------------
val_dataset = RWF2000Dataset(DATA_DIR, "val")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
model = ViolenceNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

y_true = []
y_score = []

# ---------------- INFERENCE ----------------
with torch.no_grad():
    for frames, flow, labels in val_loader:
        frames = frames.to(device)
        flow = flow.to(device)

        outputs = model(frames, flow)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # violence prob

        y_true.extend(labels.numpy())
        y_score.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_score = np.array(y_score)

# threshold-based prediction (better than argmax for analysis)
y_pred = (y_score >= THRESHOLD).astype(int)

# =============================
# CONFUSION MATRIX PNG
# =============================
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NonFight", "Fight"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved confusion_matrix.png")

# =============================
# ROC CURVE PNG
# =============================
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved roc_curve.png")
print("ROC AUC:", roc_auc)

# =============================
# PRECISION–RECALL PNG
# =============================
precision, recall, _ = precision_recall_curve(y_true, y_score)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")

plt.savefig(f"{OUT_DIR}/pr_curve.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved pr_curve.png")

print("\n✅ All evaluation plots generated in:", OUT_DIR)
