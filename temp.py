import torch
import random

a = []
for i in range(8):
    a.append(torch.ones(2,10))

print torch.cat(a, dim=0)
