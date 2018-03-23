import utilsRL
import buildModel
import prepareDataset

import copy
import math
import random
from PIL import Image
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

frameDropThreshold = 0.625
trainPairs = utilsRL.createTestPairs()

Transition = namedtuple('Transition', ('pid', 'framesCount', 'state', 'action', 'nextState', 'reward', 'framesDropInfo'))
seqRootRGB = '/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
personNoDict = dict([v,k] for k,v in personIdxDict.items())

model = utilsRL.DQN()
if torch.cuda.is_available():
    model = model.cuda()

model.load_state_dict(torch.load('mytraining.pt'))
testTriplets = torch.load('testTriplets.pt')
testPairs = utilsVer.generatePairs(testTriplets)

'''
for pair in trainPairs:
    if pair[0] == pair[1]:
        label = 1
    else:
        label = 0

    framesDropInfo, threshold, initialState = utilsRL.getTripletInfo(triplet, personIdxDict, personFramesDict)
    pid = {}
    pid['A'] = personIdxDict[triplet[0]]
    pid['B'] = personIdxDict[triplet[1]]
    state = copy.deepcopy(initialState)

    while True:
        pid, framesCount, state, framesDropInfo = utilsVer.dictToTensor(pid, state, framesDropInfo)
        pid_batch, framesCount_batch, state_batch, action_batch = utilsRL.generateAllAction(pid, framesCount, nextState, framesDropInfo)
        # print pid_batch, framesCount_batch, state_batch, action_batch
        input = pid_batch, framesCount_batch, state_batch, action_batch
        bestAction = model(*input).max(0)[1].data

        nextState, done = utilsVer.performAction(state, bestAction, threshold, pid, framesDropInfo)
'''
