import prepareDataset
import tripletNetwork

import os
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
input_size = 128
hidden_size = 128
num_layers = 2
learning_rate = 0.001
momentum = 0
alpha = torch.FloatTensor([0.4])
num_epochs = 2
batch_size = 1
testTrainSplit = 0.8

seqRootRGB = '/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
nTotalPersons = len(personFramesDict)
trainTriplets, testTriplets = prepareDataset.generateTriplets(nTotalPersons, testTrainSplit)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        out, hidden_data = self.lstm(x, (h0, c0))
        return out

rnn = RNN(input_size, hidden_size, num_layers)
tripletRNN = tripletNetwork.TripletNet(rnn)

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
optimizer = torch.optim.SGD(tripletRNN.parameters(), lr=learning_rate, momentum=momentum)

torch.backends.cudnn.benchmark = True
count = 1
tripletRNN.train()
for epoch in range(num_epochs):
    print("Epoch Loop Running", epoch)
    for triplet in trainTriplets:
        triplet = [15,15,213]
        print("Triplet being used : ", triplet)
        count += 1
        if(count > 2):
            break
        for cam in range(1,2):
            #print("Cam Loop Running")
            anchor = personIdxDict[triplet[0]]
            positive = personIdxDict[triplet[1]]
            negative = personIdxDict[triplet[2]]

            framesCountList = personFramesDict[anchor]
            anchorFramesCount = framesCountList[cam-1]
            framesCountList = personFramesDict[positive]
            positiveFramesCount = framesCountList[2-cam]
            framesCountList = personFramesDict[negative]
            negativeFramesCount = framesCountList[2-cam]
            actualFramesCount = max(anchorFramesCount, positiveFramesCount, negativeFramesCount)

            anchorFrames = prepareDataset.getPersonFrames(seqRootRGB, anchor, cam, actualFramesCount)
            positiveFrames = prepareDataset.getPersonFrames(seqRootRGB, positive, 3-cam, positiveFramesCount)
            negativeFrames = prepareDataset.getPersonFrames(seqRootRGB, negative, 3-cam, negativeFramesCount)

            quotient = actualFramesCount / sequence_length
            remainder = actualFramesCount % sequence_length

            anchorFramesRemaining = anchorFramesCount
            positiveFramesRemaining = positiveFramesCount
            negativeFramesRemaining = negativeFramesCount

            tempAnchorFramesRNNInput = anchorFrames[0:16]
            tempPositiveFramesRNNInput = positiveFrames[0:16]
            tempNegativeFramesRNNInput = negativeFrames[0:16]

            # print("Q, R:", quotient, remainder)
            for i in range(quotient):
                print("Quotient Loop Running")
                if anchorFramesRemaining < 16:
                    anchorFramesRNNInput = tempAnchorFramesRNNInput
                else:
                    anchorFramesRNNInput = anchorFrames[sequence_length*(i):sequence_length*(i+1)]
                    tempAnchorFramesRNNInput = anchorFramesRNNInput
                    anchorFramesRemaining -= 16

                if positiveFramesRemaining < 16:
                    positiveFramesRNNInput = tempPositiveFramesRNNInput
                else:
                    positiveFramesRNNInput = positiveFrames[sequence_length*(i):sequence_length*(i+1)]
                    tempPositiveFramesRNNInput = positiveFramesRNNInput
                    positiveFramesRemaining -= 16

                if negativeFramesRemaining < 16:
                    negativeFramesRNNInput = tempNegativeFramesRNNInput
                else:
                    negativeFramesRNNInput = negativeFrames[sequence_length*(i):sequence_length*(i+1)]
                    tempNegativeFramesRNNInput = negativeFramesRNNInput
                    negativeFramesRemaining -= 16

                H, Hp, Hn = tripletRNN(anchorFramesRNNInput, positiveFramesRNNInput, negativeFramesRNNInput)
                loss = Variable(torch.zeros(1))
                # print H, Hp, Hn
                for j in range(sequence_length):
                    loss = loss + torch.max(Variable(torch.zeros(1)), Variable(alpha) - cos(H[j], Hp[j]) + cos(H[j], Hn[j]))
                loss = loss / sequence_length
                print loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            H, Hp, Hn = tripletRNN(anchorFrames[anchorFramesCount - sequence_length:anchorFramesCount], positiveFrames[negativeFramesCount - sequence_length:negativeFramesCount], negativeFrames[negativeFramesCount - sequence_length:negativeFramesCount])
            loss = Variable(torch.zeros(1))
            for j in range(sequence_length):
                loss = loss + torch.max(Variable(torch.zeros(1)), Variable(alpha) - cos(H[j], Hp[j]) + cos(H[j], Hn[j]))
            loss = loss / sequence_length
            print loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

torch.save(tripletRNN.state_dict(), '/Users/prateek/8thSem/rl-person-verification/runs/model_run.pt')
