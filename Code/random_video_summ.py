import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from model import LSTMVideoSummarizer

# =====================================================
# CONFIG
# =====================================================
VIDEO_PATH = "C:\\Users\\saata\\OneDrive\\Pictures\\Malav_pg.mp4"
OUTPUT_DIR = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\outputs\\summary"
MODEL_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\outputs\\model.pth"

TARGET_FPS = 2
SEQ_LEN = 10
SUMMARY_RATIO = 0.35   # 65% of frames

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD TRAINED LSTM MODEL
# =====================================================
model = LSTMVideoSummarizer().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded")

# =====================================================
# LOAD RESNET FEATURE EXTRACTOR
# =====================================================
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# READ VIDEO + EXTRACT FRAMES & FEATURES
# =====================================================
cap = cv2.VideoCapture(VIDEO_PATH)
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, int(video_fps // TARGET_FPS))

frames = []
features = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        frames.append(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = resnet(tensor)
            feat = feat.squeeze().cpu().numpy()

        features.append(feat)

    frame_count += 1

cap.release()

frames = np.array(frames)
features = np.array(features)

print(f"Extracted {len(frames)} frames")

# =====================================================
# SAFETY CHECK
# =====================================================
if len(features) <= SEQ_LEN:
    raise ValueError("Video too short for sequence length")

# =====================================================
# CREATE SEQUENCES (FAST, NO WARNING)
# =====================================================
X = []
for i in range(len(features) - SEQ_LEN):
    X.append(features[i:i + SEQ_LEN])

X = torch.from_numpy(np.array(X)).float().to(device)

# =====================================================
# PREDICT SEQUENCE IMPORTANCE
# =====================================================
with torch.no_grad():
    seq_scores = model(X).cpu().numpy().flatten()

# =====================================================
# MAP SEQUENCE SCORES → FRAME INDICES (FIXED)
# =====================================================
frame_indices = np.arange(len(seq_scores)) + SEQ_LEN // 2
frame_indices = np.clip(frame_indices, 0, len(frames) - 1)

pairs = list(zip(frame_indices, seq_scores))

num_selected = max(1, int(len(frames) * SUMMARY_RATIO))

# Sort by importance
pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

# Keep only UNIQUE frame indices
unique_pairs = []
seen = set()

for idx, score in pairs:
    if idx not in seen:
        unique_pairs.append((idx, score))
        seen.add(idx)
    if len(unique_pairs) >= num_selected:
        break

# Sort temporally
unique_pairs = sorted(unique_pairs, key=lambda x: x[0])

selected_frames = [frames[idx] for idx, _ in unique_pairs]

print(f"Selected {len(selected_frames)} important frames")

# =====================================================
# CREATE SUMMARY VIDEO
# =====================================================
clip = ImageSequenceClip(
    [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in selected_frames],
    fps=TARGET_FPS
)

output_path = os.path.join(OUTPUT_DIR, "summary.mp4")
clip.write_videofile(output_path, codec="libx264", audio=False)

print("🎉 Summary video saved at:", output_path)