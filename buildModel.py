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

class tripletRNN(nn.Module):
    def __init__(self, rnnOutput):
        super(tripletRNN, self).__init__()
        self.featureVector = rnnOutput

    def forward(self, x, y, z):
        featureVectorH = self.featureVector(Variable(x.float()))
        featureVectorHp = self.featureVector(Variable(y.float()))
        featureVectorHn = self.featureVector(Variable(z.float()))
        return featureVectorH, featureVectorHp, featureVectorHn

def cnn(img):
    vgg = models.alexnet(pretrained=True)
    mod = list(vgg.classifier.children())
    new_classifier = torch.nn.Sequential(*mod[:2])
    vgg.classifier = new_classifier
    return vgg(img)
