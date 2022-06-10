import os
from pytest import fail
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from data import data_process
from model import model_2
from model import model_resnet
from model import model_pretrained_resnet
import torchvision.transforms as transforms
## Specify image paths
cwd = os.getcwd()
csv_path_train = cwd + "/data/dataset/CV_Images_12/training/output/assigned_classes.csv"
img_dir_train = cwd + "/data/dataset/CV_Images_12/training"
#
# csv_path_train = cwd + "/augmented_datset_train/output/out.csv"
# img_dir_train = cwd + "/augmented_datset_train"

csv_path_test = cwd + "/data/dataset/CV_Images_12/testing/test2/output/assigned_classes.csv"
img_dir_test = cwd + "/data/dataset/CV_Images_12/testing/test2"

## Generate train and test set
transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ])


# dataset_train = data_process.ImageSet(csv_path_train, img_dir_train)
dataset_train = data_process.ImageSet(csv_path_train, img_dir_train, transform1=transform, transform2=None,
                                      transform3=None)

# train_loader = data_process.TrainLoader(dataset_train)
train_loader = DataLoader(dataset_train, shuffle=False)

dataset_test = data_process.ImageSet(csv_path_test, img_dir_test)
# test_loader = data_process.TrainLoader(dataset_test)
test_loader = DataLoader(dataset_test, shuffle=False)


## Load CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

## Load Model

# model = model_resnet.ResNet18()
# model = model_2.Net()
model = model_pretrained_resnet.get_pretrained_resnet()

model.to(device)

## Loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.00001)

def evaluate_accuracy(data_loader, net, device=device):
    net.eval()  #make sure network is in evaluation mode

    #init
    acc_sum = torch.tensor([0], dtype=torch.float, device=device)
    n = 0

    for X, _, _, y in data_loader:
        # Copy the data to device.
        X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))
            n += y.shape[0] #increases with the number of samples in the batch
    return acc_sum.item()/n


epochs = 5

## Training loop
def train_model():
    correct = 0
    total = 0
    failure = 0
    success = 0
    correct_fail=0
    correct_success=0
    for epoch in range(epochs):  # loop over the dataset multiple times
        # running_loss = 0.0
        running_loss = []
        # running_loss_v=0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            # print(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())

            ## Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print(_)
            # print(predicted)
            # print(outputs)

            if labels == 1:
                failure = failure +1
                if predicted == 1:
                    correct_fail += 1

            if labels == 0:
                success = success +1
                if predicted == 0:
                    correct_success += 1

            # print(labels)
            # print(outputs)

            # print statistics
            # running_loss_v += loss.item()
            # if i % 10 == 9:    # print every 10 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss_v / 10:.3f}')
            #     running_loss_v = 0.0

        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
        print(f'Accuracy of the network on the train images: {100 * correct // total} %')
        print(f'Accuracy of the network failure: {100 * correct_fail // failure} %')
        print(f'Accuracy of the network success: {100 * correct_success // success} %')

    print('Finished Training')

    torch.save(model.state_dict(), "train_model.pth")


## Load model
# model.load_state_dict(torch.load("train_model.pth"))

## Testing loop
def test_model():
        correct = 0
        total = 0
        failure = 0
        success = 0
        correct_fail=0
        correct_success=0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)


                # calculate outputs by running images through the network
                outputs = model(images)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)


                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if labels == 1:
                    failure = failure +1
                    if predicted == 1:
                        correct_fail += 1

                if labels == 0:
                    success = success +1
                    if predicted == 0:
                        correct_success += 1


        print(f'Accuracy of the network on the test images: {100 * correct // total} %')
        print(f'Accuracy of the network failure: {100 * correct_fail // failure} %')
        print(f'Accuracy of the network success: {100 * correct_success // success} %')


train_model()

test_model()