import prepareDataset

import os
import copy
import random
from itertools import count
from collections import namedtuple

import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

frameDropThreshold = 0.625

Transition = namedtuple('Transition', ('pid', 'framesCount', 'state', 'action', 'nextState', 'reward', 'framesDropInfo'))

seqRootRGB = '/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
personNoDict = dict([v,k] for k,v in personIdxDict.items())

def generatePairs(testTriplets):
    print testTriplets
    testPairs = []
    for i in range(testTriplets.size(0)):
        tempPair = [[testTriplets[i][0], 0], [testTriplets[i][1], 1]]
        testPairs.append(tempPair)
        tempPair = [[testTriplets[i][1], 1], [testTriplets[i][2], 1]]
        testPairs.append(tempPair)
        tempPair = [[testTriplets[i][0], 0], [testTriplets[i][2], 1]]
        testPairs.append(tempPair)
    return testPairs
