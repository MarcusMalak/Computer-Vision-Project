import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class ImageSet(Dataset):
    def __init__(self, csv_path, img_dir, transform = None):
        self.img_labels = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = image.resize((256, 256))
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

cwd = os.getcwd()
csv_path = cwd + "/printfailure/data/dataset/CV_Images/output/assigned_classes.csv"
img_dir = cwd + "/printfailure/data/dataset/CV_Images"

dataset = ImageSet(csv_path, img_dir)
train_loader = DataLoader(dataset, shuffle=True)
