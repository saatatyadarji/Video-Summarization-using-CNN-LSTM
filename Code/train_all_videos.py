import torch
from torch.utils.data import DataLoader
from dataset_multi import SumMeMultiVideoDataset
from model import LSTMVideoSummarizer
import os

# -----------------------------
# CONFIG
# -----------------------------
SEQ_LEN = 10
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

FEATURES_DIR = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\features"
GT_DIR = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\GT"
MODEL_SAVE_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\outputs\\model.pth"

os.makedirs("outputs", exist_ok=True)

# -----------------------------
# DATASET
# -----------------------------
dataset = SumMeMultiVideoDataset(
    FEATURES_DIR,
    GT_DIR,
    seq_len=SEQ_LEN
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# -----------------------------
# MODEL
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMVideoSummarizer().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("✅ Model trained on ALL videos and saved.")
