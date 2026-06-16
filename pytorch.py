"""
PyTorch End-to-End Computer Vision Workflow
Synthetic tensor data -> Dataset -> DataLoader -> CNN -> Train -> Evaluate -> Export
"""

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 1. Synthetic Image Data ───────────────────────────────────────────────────
N_SAMPLES = 2000
IMG_SIZE = 16

X = torch.randn(N_SAMPLES, 1, IMG_SIZE, IMG_SIZE)          # (N, C, H, W)
y = (X.mean(dim=(1, 2, 3)) > 0).long()                     # binary label
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ── 2. Train / Test Split ─────────────────────────────────────────────────────
SPLIT = int(0.8 * N_SAMPLES)

train_ds = TensorDataset(X[:SPLIT], y[:SPLIT])
test_ds  = TensorDataset(X[SPLIT:], y[SPLIT:])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# ── 3. CNN Model ──────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 1x16x16 -> 8x8x8
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 8x8x8 -> 16x4x4
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Classifier
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


model = SimpleCNN().to(device)
print(model)

# ── 4. Loss & Optimizer ───────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── 5. Training Loop ──────────────────────────────────────────────────────────
EPOCHS = 5

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch}/{EPOCHS}]  Loss: {avg_loss:.4f}")

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds   = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += len(yb)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")

# ── 7. Save Model Weights ─────────────────────────────────────────────────────
torch.save(model.state_dict(), "cnn.pth")
print("Model saved to cnn.pth")

# ── 8. Export to ONNX ─────────────────────────────────────────────────────────
dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "cnn.onnx",
    opset_version=11,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
)
print("ONNX model exported to cnn.onnx")

# ── 9. Inference on a Single Sample ──────────────────────────────────────────
sample = X[0:1].to(device)

model.eval()
with torch.no_grad():
    probs = model(sample).softmax(dim=1)
    pred_class = probs.argmax(dim=1).item()

print(f"Sample probabilities: {probs}")
print(f"Predicted class: {pred_class}")

