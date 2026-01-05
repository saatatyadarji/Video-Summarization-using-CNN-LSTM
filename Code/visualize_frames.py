import os
import cv2
import matplotlib.pyplot as plt

FRAMES_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\outputsframes"

# list frame images
frames = sorted([f for f in os.listdir(FRAMES_PATH) if f.endswith(".jpg")])

print("Total frames found:", len(frames))

# display first 5 frames
num_frames_to_show = 5

for i in range(min(num_frames_to_show, len(frames))):
    frame_path = os.path.join(FRAMES_PATH, frames[i])
    
    # read image using OpenCV
    image = cv2.imread(frame_path)
    
    # convert BGR to RGB (important!)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(4, 3))
    plt.imshow(image)
    plt.title(frames[i])
    plt.axis("off")
    plt.show()
