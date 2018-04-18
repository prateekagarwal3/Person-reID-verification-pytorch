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

model.load_state_dict(torch.load(dirPath + 'gpu-rl/runs/model_run_dqn.pt'))
testTriplets = torch.load('testTriplets.pt')
testPairs = utilsVer.generatePairs(testTriplets)
# print testPairs

correctPairs = 0
for pair in testPairs:

    pid, framesDropInfo, framesCount, threshold, initialState = utilsVer.getPairInfo(pair)
    state = initialState.clone()
    # print framesDropInfo

    for t in count():
        print("T Loop Running current t=", t)
        action = utilsVer.getAction(pair, Variable(pid.clone()), framesCount.clone(), state.clone(), framesDropInfo.clone(), model)

        nextState, framesDropInfo, reward = utilsVer.performAction(state, action, threshold, pid, framesDropInfo, framesCount)

        state = nextState.clone()

        done = utilsVer.checkTerminal(pid, framesCount, state, framesDropInfo, model)

        if done:
            break

    terminalState = state.clone()
    sim = utilsVer.findSimilarity(pair, terminalState)
    if pair[0][0] == pair[1][0] and sim == 1:
        correctPairs += 1
    elif pair[0][0] != pair[1][0] and sim == 0:
        correctPairs += 1
    print("Accuracy in 225 triplets = {}".format(float(correctPairs) / float(len(testPairs)) * 100))
