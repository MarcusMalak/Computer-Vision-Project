import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageSet(Dataset):
    def __init__(self, csv_path, img_dir, transform1 = transforms.Grayscale(),  transform3 = transforms.ToTensor(), transform2 = transforms.ToPILImage()):
        self.img_labels = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3 
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        # if self.transform1:
        #     image = self.transform1(image)
        if self.transform2:
            image = self.transform2(image)
        if self.transform3:
            image = self.transform3(image)

        return image, label

class TrainLoader(DataLoader):
    def __init__(self, dataset):
        self.dataset = dataset

    def split_data(self):
        data_loader = DataLoader(self.dataset)

        return data_loader
