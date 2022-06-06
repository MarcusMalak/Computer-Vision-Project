import os
from pytest import fail
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from data import data_process
from model import model_2 


## Specify image paths
cwd = os.getcwd()
csv_path_train = cwd + "/printfailure/data/dataset/CV_Images_12/training/output/assigned_classes.csv"
img_dir_train = cwd + "/printfailure/data/dataset/CV_Images_12/training"

csv_path_test = cwd + "/printfailure/data/dataset/CV_Images_12/testing/test2/output/assigned_classes.csv"
img_dir_test = cwd + "/printfailure/data/dataset/CV_Images_12/testing/test2"

## Generate train and test set
dataset_train = data_process.ImageSet(csv_path_train, img_dir_train)
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
model = model_2.Net()
model.to(device)

## Loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.000005, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)



## Training loop
def train_model():
    correct = 0
    total = 0
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
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

            ## Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0


    print(f'Accuracy of the network on the test images: {100 * correct // total} %')       
    print('Finished Training')
    torch.save(model.state_dict(), "train_model.pth")


## Load model
model.load_state_dict(torch.load("train_model.pth"))

## Testing loop
def test_model():
        correct = 0
        total = 0
        tp = 0
        fn = 0
        failure = 0
        success = 0
        correct_fail=0
        correct_succes=0
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
                    tp = tp + 1
                    if predicted == 0:
                        fn = fn + 1
                    if predicted == 1:
                        correct_fail += 1

                if labels == 0:
                    success = success +1
                    if predicted == 0:
                        correct_succes += 1

        # recall = tp / (tp + fn)
                
        print(f'Accuracy of the network on the test images: {100 * correct // total} %')
        print(f'Accuracy of the network failure: {100 * correct_fail // failure} %')
        print(f'Accuracy of the network success: {100 * correct_succes // success} %')


        # print(tp)
        # print(fn)
        # print(recall)
        # print(total)
        # print(success)
        # print(failure)


# train_model()

test_model()