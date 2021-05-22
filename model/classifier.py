'''
A simple modified version of ResNet-18. 
We change the output of the last fc layer to 2 classes. 
'''


import torch
from torchvision.models import resnet18
import torch.nn as nn


def get_resnet():
    net = resnet18()
    modified_net = nn.Sequential(*list(net.children())[:-1])  # fetch all of the layers before the last fc.
    return modified_net


class ClassifyModel(nn.Module):
    def __init__(self, n_class=2):
        super(ClassifyModel, self).__init__()
        self.backbone = get_resnet()
        self.extra_layer = nn.Linear(512, n_class)

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        out = self.extra_layer(out)
        return out


def get_classifier(pretrain_path, device):
    classifier = ClassifyModel().to(device)
    classifier.load_state_dict(torch.load(pretrain_path))    
    return classifier
