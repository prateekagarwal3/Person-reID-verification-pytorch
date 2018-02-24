import os
import time
import torch
import timeit
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from sklearn.decomposition import PCA
import torchvision.transforms as transforms

def preprocessImage(imgPath):
    img = Image.open(imgPath)
    img = transforms.Pad((80, 48), fill=0)(img)
    img = transforms.ToTensor()(img)
    return Variable(img.unsqueeze_(0))

def cnn(img):
    vgg = models.alexnet(pretrained=True)
    mod = list(vgg.classifier.children())
    new_classifier = torch.nn.Sequential(*mod[:2])
    vgg.classifier = new_classifier
    return vgg(img)

# print cnn(preprocessImage('/Users/prateek/8thSem/rl-person-verification/dataset/iLIDS-VID/i-LIDS-VID/sequences/cam1/person001/cam1_person001_00317.png'))
