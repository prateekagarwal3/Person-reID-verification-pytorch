import sys
import torch
import random

a = [i for i in range(0, 24)]
print a
for i in range(0, 20, 5):
    a[i+1] = -1
    a[i+2] = -1
    a[i+3] = -1
    a[i+4] = -1
print a
a[i+6:i+9] = [-1, -1, -1, -1]
print a
