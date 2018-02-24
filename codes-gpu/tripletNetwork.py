import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from sklearn.decomposition import PCA
import torchvision.transforms as transforms

class TripletNet(nn.Module):
    def __init__(self, rnnOutput):
        super(TripletNet, self).__init__()
        self.featureVector = rnnOutput

    def forward(self, x, y, z):
        featureVectorH = self.featureVector(Variable(x.float()).cuda)
        featureVectorHp = self.featureVector(Variable(y.float()).cuda)
        featureVectorHn = self.featureVector(Variable(z.float()).cuda)
        return featureVectorH, featureVectorHp, featureVectorHn
