import sys
import torch
import random

print sys.platform
if sys.platform.startswith('linux'):
    dirPath = '/data/home/prateeka/'
    print "working"
elif sys.platform.startswith('darwin'):
    dirPath = '/Users/prateek/8thSem/'
    print "working"

from torchviz import make_dot
pip install git+https://github.com/szagoruyko/pytorchviz
make_dot(loss, params=dict(model.named_parameters()))
