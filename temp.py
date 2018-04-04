import sys
import torch
import random

count = torch.ones(12, 1)
count[0:2] = 2
print count
