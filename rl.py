import utilsRL
import prepareDataset

import sys
import time
import copy
import math
import random
from PIL import Image
from itertools import count
from torchviz import make_dot
from collections import namedtuple
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

num_epochs = 1
testTrainSplit = 0.75
BATCH_SIZE = 8
GAMMA = 0.98
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000

Transition = namedtuple('Transition', ('pid', 'framesCount', 'state', 'action', 'nextState', 'reward', 'framesDropInfo'))

if sys.platform.startswith('linux'):
    dirPath = '/data/home/prateeka/'
elif sys.platform.startswith('darwin'):
    dirPath = '/Users/prateek/8thSem/'

policy_net = utilsRL.DQN()
target_net = utilsRL.DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if torch.cuda.is_available():
    policy_net = policy_net.cuda()
    target_net = target_net.cuda()

torch.backends.cudnn.enabled = True

optimizer = optim.RMSprop(policy_net.parameters(), lr = 1e-6)
memory = utilsRL.ReplayMemory(10000)
episodeDurations = []
steps_done = 0

seqRootRGB = dirPath + 'dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
trainTriplets, testTriplets = prepareDataset.generateTriplets(len(personFramesDict), testTrainSplit)
personNoDict = dict([v,k] for k,v in personIdxDict.items())
utilsRL.saveTestTriplet(testTriplets)

def optimizeModel():
    # print("Optimizing Model Begin")
    if len(memory) < BATCH_SIZE:
        # print("Optimizing Model End because of not sufficient training examples in replay memory")
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # print batch

    pid_batch = Variable(torch.cat(batch.pid).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.pid))
    # pid_batch = pid_batch.view(BATCH_SIZE, -1)

    framesCount_batch = Variable(torch.cat(batch.framesCount).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.framesCount))
    # framesCount_batch = framesCount_batch.view(BATCH_SIZE, -1)

    state_batch = Variable(torch.cat(batch.state).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.state))
    # state_batch = state_batch.view(BATCH_SIZE, 3, -1)

    action_batch = Variable(torch.cat(batch.action).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.action))
    # action_batch = action_batch.view(BATCH_SIZE, -1)

    reward_batch = Variable(torch.cat(batch.reward).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.reward))
    reward_batch = reward_batch.view(BATCH_SIZE, -1)

    framesDropInfo_batch = Variable(torch.cat(batch.framesDropInfo).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.framesDropInfo))
    # print framesDropInfo_batch.size()

    # framesDropInfo_batch = framesDropInfo_batch.view(BATCH_SIZE, 3, -1)

    nextState_batch = Variable(torch.cat(batch.nextState).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.nextState))
    # nextState_batch = nextState_batch.view(BATCH_SIZE, 3, -1)

    stateValues = Variable(utilsRL.getStateValues(pid_batch, framesCount_batch, state_batch, framesDropInfo_batch, policy_net).data, requires_grad=True)

    nextStateValues = utilsRL.getStateValues(pid_batch, framesCount_batch, nextState_batch, framesDropInfo_batch, target_net)

    expectedStateValues = (nextStateValues * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(stateValues, expectedStateValues)

    lossTic = time.time()
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-0.5, 0.5)
    optimizer.step()
    lossToc = time.time()
    # print("Optimizing Model End")

temp = 1
for epoch in range(num_epochs):
    epochTic = time.time()
    tripletCounter = 0
    for triplet in trainTriplets:
        tripletCounter += 1
        tripletTic = time.time()

        # temp += 1
        # if temp > 2:
        #     break
        # triplet = [103, 103, 107]
        # triplet = [34, 34, 172]
        # triplet = [259, 259, 133]

        print("Triplet Loop Running, triplet=", triplet)
        pid, framesDropInfo, framesCount, threshold, initialState = utilsRL.getTripletInfo(triplet, personIdxDict, personFramesDict)
        # print framesDropInfo
        state = initialState.clone()
        for t in count():
            # print("initialState", state)
            # print("T Loop Running current t=", t)
            # print framesDropInfo.view(3, 81)
            done, action = utilsRL.getAction(Variable(pid.clone()), framesCount.clone(), state.clone(), framesDropInfo.clone(), policy_net)
            # print("initialState", state)
            # print framesDropInfo
            # print action
            nextState, framesDropInfo, reward, doneP = utilsRL.performAction(state, action, threshold, pid, framesDropInfo, framesCount)
            # print framesDropInfo
            memory.push(pid, framesCount, state, action, nextState, reward, framesDropInfo)
            # print("nextState", nextState)
            state = nextState.clone()
            # print(sum(state[0]), threshold[0])
            # print(sum(state[1]), threshold[1])
            # print(sum(state[2]), threshold[2])
            # print done
            optimizeModel()
            # cprint("done", doneP)
            if done or doneP:
                episodeDurations.append(t + 1)
                break

        if tripletCounter % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        tripletToc = time.time()
        print("Time taken by triplet : {}seconds".format( tripletToc-tripletTic))
    epochToc = time.time()
    print("Time by Epoch [{}/{}] : {}seconds".format(epoch+1, num_epochs, epochToc - epochTic))

# print episodeDurations
# print memory.__len__()
torch.save(target_net.state_dict(), dirPath + 'gpu-rl/runs/model_run_dqn.pt')
