import sys
import numpy

import torch

opt=sys.argv[1]

if opt=='0': # load weight+save model
    # /n/coxfs01/donglai/lib/malis-pytorch/scripts/db2.py 
    from model import ffn
    model = ffn()
    # load weight
    
