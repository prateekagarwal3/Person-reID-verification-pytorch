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

# torch.manual_seed(7)

sequence_length = 16
input_size = 128
hidden_size = 32 
num_layers = 2
learning_rate = 0.001
momentum = 0.9
alpha = torch.FloatTensor([0.3])
if torch.cuda.is_available():
    alpha = alpha.cuda()
alpha = Variable(alpha)
num_epochs = 200
testTrainSplit = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print device

trainTriplets, testTriplets = prepareDataset.generateTriplets(300, testTrainSplit)
features_dir = '/data/home/prateeka/ilids_features/'

if os.path.isfile("lossRGB.txt"):
    os.remove("lossRGB.txt")
if os.path.isfile("lossRGBv.txt"):
    os.remove("lossRGBv.txt")

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = 0.25)

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

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

print get_n_params(tripletRNNRGB)

model_parameters = filter(lambda p: p.requires_grad, tripletRNNRGB.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print params

cos = nn.CosineSimilarity(dim=2, eps=1e-6)

# def triplet_loss(H, Hp, Hn):
#     zero = Variable(torch.zeros(1).cuda()) if torch.cuda.is_available() else Variable(torch.zeros(1))
#     return torch.mean(torch.mean(torch.max(zero, alpha - cos(H, Hp) + cos(Hp, Hn)), dim=1))

triplet_loss = nn.TripletMarginLoss(margin=0.4, p=2)

optimizerRGB = torch.optim.Adam(tripletRNNRGB.parameters(), lr = learning_rate)#, momentum = momentum)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def input_creator(triplet):
    anchorFrames = torch.load(features_dir + 'cam1/' + str(triplet[0])+'.pt')
    positiveFrames = torch.load(features_dir + 'cam2/' + str(triplet[1])+'.pt')
    negativeFrames = torch.load(features_dir + 'cam2/' + str(triplet[2])+'.pt')
    anchorFC = anchorFrames.size(0)
    positiveFC = positiveFrames.size(0)
    negativeFC = negativeFrames.size(0)
    maxFC = min(anchorFC, positiveFC, negativeFC)
    anchorBatchSize = anchorFC / sequence_length + 1
    positiveBatchSize = positiveFC / sequence_length + 1
    negativeBatchSize = negativeFC / sequence_length + 1
    maxBatchSize = min(anchorBatchSize, positiveBatchSize, negativeBatchSize)
    anchorIP = torch.Tensor(maxBatchSize, sequence_length, 128)
    positiveIP = torch.Tensor(maxBatchSize, sequence_length, 128)
    negativeIP = torch.Tensor(maxBatchSize, sequence_length, 128)
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
    return anchorIP, positiveIP, negativeIP

def test(modelRGB):
    print("******** Testing ********")
    modelRGB.eval()
    for bIter in range(0, len(testTriplets), 15):
        triplet = [testTriplets[bIter][0].item(), testTriplets[bIter][1].item(), testTriplets[bIter][2].item()]
        anchorIP, positiveIP, negativeIP = input_creator(triplet)
        H, Hp, Hn = tripletRNNRGB(anchorIP, positiveIP, negativeIP)
        for tIter in range(bIter+1, bIter+15):
            triplet = [testTriplets[tIter][0].item(), testTriplets[tIter][1].item(), testTriplets[tIter][2].item()]
            anchorIP, positiveIP, negativeIP = input_creator(triplet)
            Ht, Hpt, Hnt = tripletRNNRGB(anchorIP, positiveIP, negativeIP)
            H = torch.cat((H, Ht), dim=0)
            Hp = torch.cat((Hp, Hpt), dim=0)
            Hn = torch.cat((Hn, Hnt), dim=0)
        lossRGB = triplet_loss(H, Hp, Hn)
        print lossRGB.data.item()
        f = open("lossRGBv.txt", "a+")
        f.write(str(lossRGB.data.item()) + "\n")
        f.close()

for epoch in range(num_epochs):
    eTic = time.time()
    print("Epoch {}/{} starts".format(epoch+1, num_epochs))
    tripletRNNRGB.train() 
    for bIter in range(0, len(trainTriplets), 15):
        triplet = [trainTriplets[bIter][0].item(), trainTriplets[bIter][1].item(), trainTriplets[bIter][2].item()]
        anchorIP, positiveIP, negativeIP = input_creator(triplet)
        H, Hp, Hn = tripletRNNRGB(anchorIP, positiveIP, negativeIP)
        for tIter in range(bIter+1, bIter+15):
            triplet = [trainTriplets[tIter][0].item(), trainTriplets[tIter][1].item(), trainTriplets[tIter][2].item()]
            anchorIP, positiveIP, negativeIP = input_creator(triplet)
            Ht, Hpt, Hnt = tripletRNNRGB(anchorIP, positiveIP, negativeIP)
            H = torch.cat((H, Ht), dim=0)
            Hp = torch.cat((Hp, Hpt), dim=0)
            Hn = torch.cat((Hn, Hnt), dim=0)
        lossRGB = triplet_loss(H, Hp, Hn)
        print lossRGB.data.item()
        f = open("lossRGB.txt", "a+")
        f.write(str(lossRGB.data.item()) + "\n")
        f.close()
        optimizerRGB.zero_grad()
        lossRGB.backward()
        optimizerRGB.step()
    test(tripletRNNRGB)
    eToc = time.time()
    print("Epoch {}/{} ends, Time Taken is : {} seconds".format(epoch+1, num_epochs, eToc-eTic))

torch.save(tripletRNNRGB.state_dict(), '/data/home/prateeka/gpu-rl/runs/model_run_rgb.pt')