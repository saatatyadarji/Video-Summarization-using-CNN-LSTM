import os
import cv2
import numpy as np
from scipy.io import loadmat


# PATHS

VIDEOS_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\videos"
GT_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\GT"
FRAMES_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\outputsframes"


# PICK ONE VIDEO (for now)

video_name = sorted([v for v in os.listdir(VIDEOS_PATH) if v.endswith(".mp4")])[0]
video_base = os.path.splitext(video_name)[0]

print("Using video:", video_name)


# LOAD GT FILE

gt_file = video_base + ".mat"
gt_data = loadmat(os.path.join(GT_PATH, gt_file))

gt_scores = gt_data["gt_score"].flatten()
original_fps = int(gt_data["FPS"][0][0])

print("Original FPS:", original_fps)
print("GT score length:", len(gt_scores))


# LOAD VIDEO FPS (for safety)

cap = cv2.VideoCapture(os.path.join(VIDEOS_PATH, video_name))
video_fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.release()

print("Video FPS:", video_fps)


# FRAME SAMPLING SETUP

TARGET_FPS = 2
frame_interval = max(1, original_fps // TARGET_FPS)

print("Frame interval:", frame_interval)


# ALIGN FRAMES WITH GT SCORES

frame_files = sorted([
    f for f in os.listdir(FRAMES_PATH)
    if f.startswith(video_name) and f.endswith(".jpg")
])

aligned_data = []

for idx, frame_file in enumerate(frame_files):
    original_frame_index = idx * frame_interval

    if original_frame_index >= len(gt_scores):
        break

    importance_score = gt_scores[original_frame_index]

    aligned_data.append(
        (os.path.join(FRAMES_PATH, frame_file), importance_score)
    )


# DISPLAY RESULTS

print("\nAligned samples:")
for i in range(min(10, len(aligned_data))):
    print(aligned_data[i])

print("\nTotal aligned frames:", len(aligned_data))
