import prepareDataset

import os
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

testTrainSplit = 0.8
frameDropThreshold = 0.7

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=9)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.mp = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Net()
# for (name, para) in model.named_parameters():
#     print name, para.size()

seqRootRGB = '/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
nTotalPersons = len(personFramesDict)
trainTriplets, testTriplets = prepareDataset.generateTriplets(nTotalPersons, testTrainSplit)

for i in range(1):
    for miniBatchIdx in sorted(random.sample(range(len(trainTriplets)), 1)) :
        triplet = trainTriplets[miniBatchIdx]
        APid = personIdxDict[triplet[0]]
        BPid = personIdxDict[triplet[1]]
        CPid = personIdxDict[triplet[2]]
        framesCountA = personFramesDict[APid][0]
        framesCountB = personFramesDict[BPid][1]
        framesCountC = personFramesDict[CPid][1]
        thresholdA = triplet[0] * frameDropThreshold
        thresholdB = triplet[1] * frameDropThreshold
        thresholdC = triplet[2] * frameDropThreshold
        initialStateA = [[1]*framesCountA]
        initialStateB = [[1]*framesCountA]
        initialStateC = [[1]*framesCountA]
        print framesCountA, framesCountB, framesCountC
