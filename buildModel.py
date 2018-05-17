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

def cnn(img):
    vgg = models.vgg16(pretrained=True)
    if torch.cuda.is_available():
        vgg = vgg.cuda()
    mod = list(vgg.classifier.children())
    new_classifier = torch.nn.Sequential(*mod[:2])
    vgg.classifier = new_classifier
    return vgg(img)

class TripletNet(nn.Module):
    def __init__(self, rnnOutput):
        super(TripletNet, self).__init__()
        self.featureVector = rnnOutput

    def forward(self, x, y, z):
        x = x.float()
        y = y.float()
        z = z.float()
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
        x = Variable(x)
        y = Variable(y)
        z = Variable(z)
        featureVectorH = self.featureVector(x)
        featureVectorHp = self.featureVector(y)
        featureVectorHn = self.featureVector(z)
        return featureVectorH, featureVectorHp, featureVectorHn
