import utilsRL
import utilsVer
import buildModel
import prepareDataset

import sys
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

if sys.platform.startswith('linux'):
    dirPath = '/data/home/prateeka/'
elif sys.platform.startswith('darwin'):
    dirPath = '/Users/prateek/8thSem/'

frameDropThreshold = 0.625

Transition = namedtuple('Transition', ('pid', 'framesCount', 'state', 'action', 'nextState', 'reward', 'framesDropInfo'))
seqRootRGB = '/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
personNoDict = dict([v,k] for k,v in personIdxDict.items())

model = utilsRL.DQN()
if torch.cuda.is_available():
    model = model.cuda()

model.load_state_dict(torch.load(dirPath + 'rl-person-verification/runs/model_run_dqn.pt'))
testTriplets = torch.load('testTriplets.pt')
testPairs = utilsVer.generatePairs(testTriplets)
print testPairs

for pair in trainPairs:
    if pair[0] == pair[1]:
        label = 1
    else:
        label = 0

    pid, framesDropInfo, framesCount, threshold, initialState = utilsRL.getTripletInfo(triplet, personIdxDict, personFramesDict)
    state = initialState.clone()
    print framesDropInfo
    break
    for t in count():
        print("T Loop Running current t=", t)
        action = utilsRL.getAction(pid.clone(), framesCount.clone(), state.clone(), framesDropInfo.clone(), model)
        nextState, framesDropInfo, reward, done = utilsRL.performAction(state, action, threshold, pid, framesDropInfo, framesCount)
        memory.push(pid, framesCount, state, action, nextState, reward, framesDropInfo)
        state = nextState.clone()
        if done:
            episodeDurations.append(t + 1)
            break
