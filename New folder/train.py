import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RWF2000Dataset
from model import ViolenceNet
from tqdm import tqdm

# ---------------- CONFIG ----------------
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
DATA_DIR = "Processed_Clean"
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_dataset = RWF2000Dataset(DATA_DIR, "train")
val_dataset = RWF2000Dataset(DATA_DIR, "val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = ViolenceNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for frames, flow, labels in train_bar:
        frames = frames.to(device)
        flow = flow.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames, flow)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")
    # ---------- VALIDATION ----------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for frames, flow, labels in val_loader:
            frames = frames.to(device)
            flow = flow.to(device)
            labels = labels.to(device)

            outputs = model(frames, flow)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pt")
        print("✅ Best model saved")

print("Training complete")
