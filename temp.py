import sys
import torch
import random
import torch.nn.functional as F
from torch.autograd import Variable

x = torch.ones(100)
x = x - 2
x = F.relu(Variable(x))

for i in range(100000000):
    f = open("test.txt", "a+")
    f.write(str(i) + "\n")
    f.close()
