import os
import cv2

# path to SumMe videos folder
VIDEOS_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\videos"

# list all video files
videos = [v for v in os.listdir(VIDEOS_PATH) if v.endswith(".mp4")]

print("Number of video files:", len(videos))
print("First 5 videos:", videos[:5])

# open the first video
video_path = os.path.join(VIDEOS_PATH, videos[0])
print("Opening video:", video_path)

cap = cv2.VideoCapture(video_path)

# check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# read first 5 frames
frame_count = 0
while frame_count < 5:
    ret, frame = cap.read()
    if not ret:
        break

    print(f"Read frame {frame_count}, shape:", frame.shape)
    frame_count += 1

cap.release()
print("Done reading video")
