import os
from scipy.io import loadmat

GT_PATH = "C:\\Users\\saata\\OneDrive\\Documents\\ai-ml project\\Video summarizer SumMe\\SumMe Dataset\\GT"

# list gt files
gt_files = sorted([f for f in os.listdir(GT_PATH) if f.endswith(".mat")])

print("Number of GT files:", len(gt_files))
print("First GT file:", gt_files[0])

# load one gt file
gt_file_path = os.path.join(GT_PATH, gt_files[0])
gt_data = loadmat(gt_file_path)

# print all keys inside the .mat file
print("\nKeys inside GT file:")
for key in gt_data.keys():
    print(key)

# access ground truth importance scores
if "gt_score" in gt_data:
    scores = gt_data["gt_score"]
    print("\nGT score shape:", scores.shape)
    print("First 10 scores:", scores[:10])
else:
    print("gt_score not found!")
