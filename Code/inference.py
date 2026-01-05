import os
import torch
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from model import LSTMVideoSummarizer

# =====================================================
# CONFIG
# =====================================================
SEQ_LEN = 10                 # sequence length used during training
K_PERCENT = 0.25             # keep top 15% frames or 25%
FPS = 2                      # FPS of extracted frames

VIDEO_NAME = "Cooking" # name of the video to summarize

FRAMES_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\outputsframes"
FEATURES_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\features\\Cooking.npy"
MODEL_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\outputs\\model.pth"
SUMMARY_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\outputs\\summary"

os.makedirs(SUMMARY_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD MODEL (RUNS ONCE)
# =====================================================
model = LSTMVideoSummarizer()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully")

features = np.load(FEATURES_PATH)
features = torch.tensor(features, dtype=torch.float32).to(device)

print("Loaded features shape:", features.shape)

# =====================================================
# CREATE SEQUENCES
# =====================================================
sequences = []

for i in range(len(features) - SEQ_LEN):
    sequences.append(features[i:i + SEQ_LEN])

if len(sequences) == 0:
    raise RuntimeError("Not enough frames to create sequences")

sequences = torch.stack(sequences)

print("Total sequences:", sequences.shape)

# =====================================================
# RUN INFERENCE
# =====================================================
with torch.no_grad():
    predictions = model(sequences)

# take last prediction of each sequence
frame_scores = predictions[:, -1].cpu().numpy()

print("Predicted frame scores shape:", frame_scores.shape)

# =====================================================
# SELECT TOP-K IMPORTANT FRAMES
# =====================================================
num_frames = len(frame_scores)
k = int(num_frames * K_PERCENT)

important_indices = np.argsort(frame_scores)[-k:]
important_indices = np.sort(important_indices)

print(f"Selected {len(important_indices)} important frames")

# =====================================================
# LOAD FRAME FILES (NUMERIC SORT – VERY IMPORTANT)
# =====================================================
def extract_frame_index(filename):
    # Example: Air_Force_One.mp4_frame_123.jpg
    return int(filename.split("_frame_")[1].split(".jpg")[0])

frame_files = sorted(
    [f for f in os.listdir(FRAMES_PATH) if f.startswith(VIDEO_NAME)],
    key=extract_frame_index
)

offset = SEQ_LEN - 1
max_frame_index = len(frame_files) - 1

selected_frames = []

for idx in important_indices:
    frame_idx = min(idx + offset, max_frame_index)
    selected_frames.append(
        os.path.join(FRAMES_PATH, frame_files[frame_idx])
    )

print("Selected frame files:", len(selected_frames))

if len(selected_frames) == 0:
    raise RuntimeError("No frames selected for summary.")


# =====================================================
# GENERATE SUMMARY VIDEO
# =====================================================
summary_video_path = os.path.join(
    SUMMARY_PATH, f"{VIDEO_NAME}_summary.mp4"
)

clip = ImageSequenceClip(selected_frames, fps=FPS)
clip.write_videofile(summary_video_path, codec="libx264", audio=False)

print("====================================")
print("SUMMARY VIDEO GENERATED SUCCESSFULLY")
print("Saved at:", summary_video_path)
print("====================================")