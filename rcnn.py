import buildModel
import prepareDataset

import os
import time
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

torch.manual_seed(7)

sequence_length = 16
input_size = 64
hidden_size = 64
num_layers = 2
learning_rate = 0.001
momentum = 0
alpha = torch.FloatTensor([0.4])
if torch.cuda.is_available():
    alpha = alpha.cuda()
alpha = Variable(alpha)
num_epochs = 400
batch_size = 1
testTrainSplit = 0.75

seqRootRGB = '/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
seqRootOP = '/Users/prateek/8thSem/dataset/iLIDS-VID-OF-HVP/sequences'

personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
personNoDict = dict([v,k] for k,v in personIdxDict.items())
nTotalPersons = len(personFramesDict)
trainTriplets, testTriplets = prepareDataset.generateTriplets(nTotalPersons, testTrainSplit)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        h0 = Variable(h0)
        c0 = Variable(c0)
        out, hidden_data = self.lstm(x, (h0, c0))
        return out

rnn = RNN(input_size, hidden_size, num_layers)
tripletRNNRGB = buildModel.TripletNet(rnn)
if torch.cuda.is_available():
    tripletRNNRGB.cuda()

tripletRNNOP = buildModel.TripletNet(rnn)
if torch.cuda.is_available():
    tripletRNNOP.cuda()

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
optimizerRGB = torch.optim.SGD(tripletRNNRGB.parameters(), lr=learning_rate, momentum=momentum)
optimizerOP = torch.optim.SGD(tripletRNNOP.parameters(), lr=learning_rate, momentum=momentum)

# if torch.cuda.is_available():
torch.backends.cudnn.benchmark = True
count = 1
tripletRNNRGB.train()
for epoch in range(num_epochs):
    eTic = time.time()
    print("Epoch {}/{} starts".format(epoch+1, num_epochs))
    for triplet in trainTriplets:
        tic = time.time()
        # triplet = [15,15,213]
        # count += 1
        # if(count > 2):
            # break
        print("Triplet being used : ", triplet)

        anchorFrames = torch.load('/Users/prateek/8thSem/features/featuresRGB/cam1/' + str(triplet[0])+'.pt')
        positiveFrames = torch.load('/Users/prateek/8thSem/features/featuresRGB/cam2/' + str(triplet[1])+'.pt')
        negativeFrames = torch.load('/Users/prateek/8thSem/features/featuresRGB/cam2/' + str(triplet[2])+'.pt')
        anchorFC = anchorFrames.size(0)
        positiveFC = positiveFrames.size(0)
        negativeFC = negativeFrames.size(0)
        maxFC = max(anchorFC, positiveFC, negativeFC)
        anchorBatchSize = anchorFC / sequence_length + 1
        positiveBatchSize = positiveFC / sequence_length + 1
        negativeBatchSize = negativeFC / sequence_length + 1
        maxBatchSize = max(anchorBatchSize, positiveBatchSize, negativeBatchSize)
        anchorIP = torch.Tensor(maxBatchSize, sequence_length, 64)
        positiveIP = torch.Tensor(maxBatchSize, sequence_length, 64)
        negativeIP = torch.Tensor(maxBatchSize, sequence_length, 64)
        for i in range(maxBatchSize):
            if i < anchorBatchSize-1:
                anchorIP[i] = anchorFrames[sequence_length*i : sequence_length*(i+1)]
            else:
                anchorIP[i] = anchorFrames[0 : sequence_length]
        for i in range(maxBatchSize):
            if i < positiveBatchSize-1:
                positiveIP[i] = positiveFrames[sequence_length*i : sequence_length*(i+1)]
            else:
                positiveIP[i] = positiveFrames[0 : sequence_length]
        for i in range(maxBatchSize):
            if i < negativeBatchSize-1:
                negativeIP[i] = negativeFrames[sequence_length*i : sequence_length*(i+1)]
            else:
                negativeIP[i] = negativeFrames[0 : sequence_length]
        H, Hp, Hn = tripletRNNRGB(anchorIP, positiveIP, negativeIP)
        # print H, Hp, Hn

        lossRGB = torch.zeros(1)
        if torch.cuda.is_available():
            lossRGB = loss.cuda()
        lossRGB = Variable(lossRGB)
        zero = torch.zeros(1)
        if torch.cuda.is_available():
            zero = zero.cuda()
        zero = Variable(zero)
        lossRGB += torch.sum(torch.max(zero, alpha - cos(H[0], Hp[0]) + cos(H[0], Hn[0])))
        lossRGB /= sequence_length
        print lossRGB
        optimizerRGB.zero_grad()
        lossRGB.backward()
        optimizerRGB.step()

        anchorOPFrames = torch.load('/Users/prateek/8thSem/features/featuresOP/cam1/' + str(triplet[0])+'.pt')
        positiveOPFrames = torch.load('/Users/prateek/8thSem/features/featuresOP/cam2/' + str(triplet[1])+'.pt')
        negativeOPFrames = torch.load('/Users/prateek/8thSem/features/featuresOP/cam2/' + str(triplet[2])+'.pt')
        anchorOPFC = anchorOPFrames.size(0)
        positiveOPFC = positiveOPFrames.size(0)
        negativeOPFC = negativeOPFrames.size(0)
        maxOPFC = max(anchorOPFC, positiveOPFC, negativeOPFC)
        anchorOPBatchSize = anchorOPFC / sequence_length + 1
        positiveOPBatchSize = positiveOPFC / sequence_length + 1
        negativeOPBatchSize = negativeOPFC / sequence_length + 1
        maxOPBatchSize = max(anchorOPBatchSize, positiveOPBatchSize, negativeOPBatchSize)
        anchorOPIP = torch.Tensor(maxOPBatchSize, sequence_length, 64)
        positiveOPIP = torch.Tensor(maxOPBatchSize, sequence_length, 64)
        negativeOPIP = torch.Tensor(maxOPBatchSize, sequence_length, 64)
        for i in range(maxBatchSize):
            if i < anchorOPBatchSize-1:
                anchorOPIP[i] = anchorOPFrames[sequence_length*i : sequence_length*(i+1)]
            else:
                anchorOPIP[i] = anchorOPFrames[0 : sequence_length]
        for i in range(maxOPBatchSize):
            if i < positiveOPBatchSize-1:
                positiveOPIP[i] = positiveOPFrames[sequence_length*i : sequence_length*(i+1)]
            else:
                positiveOPIP[i] = positiveOPFrames[0 : sequence_length]
        for i in range(maxOPBatchSize):
            if i < negativeOPBatchSize-1:
                negativeOPIP[i] = negativeOPFrames[sequence_length*i : sequence_length*(i+1)]
            else:
                negativeOPIP[i] = negativeOPFrames[0 : sequence_length]

        HOP, HpOP, HnOP = tripletRNNOP(anchorOPIP, positiveOPIP, negativeOPIP)
        # print HOP, HpOP, HnOP

        lossOP = torch.zeros(1)
        if torch.cuda.is_available():
            lossOP = lossOP.cuda()
        lossOP = Variable(lossOP)
        zero = torch.zeros(1)
        if torch.cuda.is_available():
            zero = zero.cuda()
        zero = Variable(zero)
        lossOP += torch.sum(torch.max(zero, alpha - cos(HOP[0], HpOP[0]) + cos(HOP[0], HnOP[0])))
        lossOP /= sequence_length
        print lossOP
        optimizerOP.zero_grad()
        lossOP.backward()
        optimizerOP.step()
        toc = time.time()
        print("Time being taken by one Triplet is ", str(toc-tic) + 's')
    eToc = time.time()
    print("Epoch {}/{} ends, Time Taken is : {} seconds".format(epoch+1, num_epochs, eToc-eTic))

torch.save(tripletRNNRGB.state_dict(), '/Users/prateek/8thSem/rl-person-verification/runs/model_run_rgb.pt')
torch.save(tripletRNNOP.state_dict(), '/Users/prateek/8thSem/rl-person-verification/runs/model_run_op.pt')
