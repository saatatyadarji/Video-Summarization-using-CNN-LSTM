import os
import cv2

# paths
VIDEOS_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\videos"
OUTPUT_FRAMES_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\outputsframes"

# create output folder if not exists
os.makedirs(OUTPUT_FRAMES_PATH, exist_ok=True)

# pick one video
video_name = "Cooking.mp4"  # sorted([v for v in os.listdir(VIDEOS_PATH) if v.endswith(".mp4")])[0]
video_path = os.path.join(VIDEOS_PATH, video_name)

print("Extracting frames from:", video_name)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# get video FPS
video_fps = int(cap.get(cv2.CAP_PROP_FPS))
print("Video FPS:", video_fps)

# extract 2 frames per second
target_fps = 2
frame_interval = max(1, video_fps // target_fps)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        frame_filename = f"{video_name}_frame_{saved_count}.jpg"
        frame_path = os.path.join(OUTPUT_FRAMES_PATH, frame_filename)
        cv2.imwrite(frame_path, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"Saved {saved_count} frames to {OUTPUT_FRAMES_PATH}")
