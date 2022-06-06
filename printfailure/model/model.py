from random import triangular
import torch
import torchvision
import torchvision.transforms as transforms


## USE 
## brightness, contrast, flipped, shift

if __name__ == '__main__':
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # batch_size = 4


    # dataset_train = torchvision.datasets.FashionMNIST("./data", train = True, download = True, transform = transforms.ToTensor())
    # idx = (dataset_train.targets==1) | (dataset_train.targets==0)
    # dataset_train.targets = dataset_train.targets[idx]
    # dataset_train.data = dataset_train.data[idx]

    # trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
    #                                         shuffle=True, num_workers=2)

    # dataset_test = torchvision.datasets.FashionMNIST("./data", train = False, download = True, transform = transforms.ToTensor())
    # idx = (dataset_test.targets==1) | (dataset_test.targets==0)
    # dataset_test.targets = dataset_test.targets[idx]
    # dataset_test.data = dataset_test.data[idx]

    # testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
    #                                         shuffle=False, num_workers=2)

    # import matplotlib.pyplot as plt
    # import numpy as np

    # # functions to show an image


    # def imshow(img):
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()


    import torch.nn as nn
    import torch.nn.functional as F
    import os
    import pandas as pd
    from torchvision.io import read_image
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    import numpy as np

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

            if self.transform1:
                image = self.transform1(image)
            if self.transform2:
                image = self.transform2(image)
            if self.transform3:
                image = self.transform3(image)    
            return image, label

    cwd = os.getcwd()
    csv_path = cwd + "/printfailure/data/dataset/CV_Images/output/assigned_classes.csv"
    img_dir = cwd + "/printfailure/data/dataset/CV_Images"

    dataset = ImageSet(csv_path, img_dir)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=True)
    

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2) ## OUTPUT = 6 * 256 * 256
            # self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2) ## 16 * 256 * 256
            self.fc1 = nn.Linear(16* 256 * 256, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 2)

        def forward(self, x):
            # x = self.pool(F.relu(self.conv1(x)))
            # x = self.pool(F.relu(self.conv2(x)))
            # x = torch.flatten(x, 1) # flatten all dimensions except batch
            # x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            # x = self.fc3(x)
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
        
            return x


    net = Net()

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    def train_model():
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:    # print every 10 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        torch.save(net.state_dict(), "train_model.pth")
    

    def test_model():
        correct = 0
        total = 0
        tn = 0
        fp = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if labels == 0:
                    tn = tn + 1
                    if predicted == 1:
                        fp = fp + 1

        recall = tn / (tn + fp)
                

        print(f'Accuracy of the network on the test images: {100 * correct // total} %')
        print(tn)
        print(fp)
        print(recall)
        # print(total)


    train_model()

    test_model()



    # for batch in train_loader:
    #     for data in test_loader:
    #         images, labels = data
    #         print(labels)

    # dataiter = iter(test_loader)
    # dataiter.next()
    # image, label = dataiter.next()

 



    # imshow(torchvision.utils.make_grid(images))
    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)

    # print(predicted)


    