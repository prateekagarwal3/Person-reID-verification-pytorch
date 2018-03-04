import utilsRL
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

torch.manual_seed(7)

testTrainSplit = 0.8
BATCH_SIZE = 64
GAMMA = 0.98
EPS_START = 1.0
EPS_END = 0.11
EPS_DECAY = 200

Transition = namedtuple('Transition', ('pid', 'state', 'action', 'next_state', 'reward'))

model = utilsRL.DQN()
optimizer = optim.RMSprop(model.parameters(), lr = 1e-6)
memory = utilsRL.ReplayMemory(10000)
episodeDurations = []

seqRootRGB = '/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
trainTriplets, testTriplets = prepareDataset.generateTriplets(len(personFramesDict), testTrainSplit)
# trainTriplets, testTriplets = utilsRL.modifyTriplets(trainTriplets, testTriplets, personIdxDict)
# print trainTriplets, testTriplets

def optimizeModel(model):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

    non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-0.5, 0.5)
    optimizer.step()

temp = 1
for triplet in trainTriplets:
    temp += 1
    if temp > 2:
        break
    triplet = [34, 34, 172]
    print("Current triplet", triplet)
    framesDropInfo, threshold, initialState = utilsRL.getTripletInfo(triplet, personIdxDict, personFramesDict)
    pid = {}
    pid['A'] = personIdxDict[triplet[0]]
    pid['B'] = personIdxDict[triplet[1]]
    pid['C'] = personIdxDict[triplet[2]]
    print("PersonID", pid)
    print("FrameInfo for anchor : {}, for positive : {}, for negative : {}".format(personFramesDict[pid['A']][0], personFramesDict[pid['B']][1], personFramesDict[pid['C']][1]))
    state = copy.deepcopy(initialState)
    for t in count():
        print("Initial State", state)
        action, framesDropInfo = utilsRL.getAction(state, framesDropInfo)
        print("Action to be taken on {}, dropped frame = {}".format(action[0], action[1]),framesDropInfo[action[0]])
        # utilsRL.checkTerminalState(state, threshold, pid, framesDropInfo)
        nextState, reward, done = utilsRL.performAction(state, action, threshold, pid, framesDropInfo)
        print("New State",nextState)
        print("________________________________________________________________________________________________________________________")
        print("________________________________________________________________________________________________________________________")
        if done:
            nextState = None
        memory.push(pid, state, action, nextState, reward)
        state = copy.deepcopy(nextState)
        # optimizeModel(model)
        if done:
            episodeDurations.append(t + 1)
            break
print episodeDurations
# print memory.printMemory()
