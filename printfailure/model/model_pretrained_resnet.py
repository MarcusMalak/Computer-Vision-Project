import torch.nn as nn
import torchvision


def get_pretrained_resnet():
    model = torchvision.models.resnet18(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.Softmax(dim=1))

    return model

