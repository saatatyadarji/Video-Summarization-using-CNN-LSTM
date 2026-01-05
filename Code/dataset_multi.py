import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class SumMeMultiVideoDataset(Dataset):
    def __init__(self, features_dir, gt_dir, seq_len=10):
        self.seq_len = seq_len
        self.samples = []

        for file in os.listdir(features_dir):
            if not file.endswith(".npy"):
                continue

            video_name = os.path.splitext(file)[0]

            features = np.load(os.path.join(features_dir, file))
            gt = loadmat(os.path.join(gt_dir, video_name + ".mat"))["gt_score"].flatten()

            min_len = min(len(features), len(gt))
            features = features[:min_len]
            gt = gt[:min_len]

            for i in range(min_len - seq_len):
                x = features[i:i + seq_len]
                y = gt[i:i + seq_len]
                self.samples.append((x, y))

        print("Total training sequences:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
