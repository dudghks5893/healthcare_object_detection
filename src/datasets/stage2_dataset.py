from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PillCropDataset(Dataset):
    def __init__(self, csv_path, class_to_idx, transform=None):
        self.df = pd.read_csv(csv_path).copy()
        self.transform = transform
        self.class_to_idx = class_to_idx

        self.df["class_id"] = self.df["class_id"].astype(str)
        self.df = self.df[self.df["class_id"].isin(self.class_to_idx.keys())].reset_index(drop=True)
        self.df["target"] = self.df["class_id"].map(self.class_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["crop_path"]).convert("RGB")
        target = int(row["target"])

        if self.transform:
            img = self.transform(img)

        return img, target