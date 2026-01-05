import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# -----------------------------
# PATHS
# -----------------------------
VIDEOS_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\videos"
FRAMES_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\outputsframes"
FEATURES_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\features"
os.makedirs(FEATURES_PATH, exist_ok=True)

# =====================================================
# MODEL
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

# =====================================================
# TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

TARGET_FPS = 2  # 👈 THIS IS THE KEY

# =====================================================
# FEATURE EXTRACTION
# =====================================================
for video_file in os.listdir(VIDEOS_PATH):
    if not video_file.endswith(".mp4"):
        continue

    video_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(VIDEOS_PATH, video_file)

    print(f"\nExtracting features for: {video_name}")

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps // TARGET_FPS))

    features = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model(frame)
                feat = feat.squeeze().cpu().numpy()

            features.append(feat)

        frame_count += 1

    cap.release()

    features = np.array(features)

    np.save(
        os.path.join(FEATURES_PATH, f"{video_name}.npy"),
        features
    )

    print(f"Saved features: {video_name}.npy | Shape: {features.shape}")

print("\n✅ Feature extraction completed at ~2 FPS for all videos.")