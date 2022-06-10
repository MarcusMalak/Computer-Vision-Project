import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms


# Load data: transforms.RandomResizedCrop(224)
# after ToTensor() : transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_pretrained_resnet():
    model = torchvision.models.resnet18(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.Softmax(dim=1))

    return model

# model_conv = model_conv.to(device)
#
# criterion = nn.CrossEntropyLoss()
#
# # Observe that only parameters of final layer are being optimized as
# # opposed to before.
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
#
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
