import argparse
import io
import json
import os
import pickle
import time
import types

import torch

def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True

def freeze_params(params):
    for p in params:
        p.requires_grad = False

def unfreeze_params(params):
    for p in params:
        p.requires_grad = True


def print_cuda_info():
    print('torch version:', torch.__version__)
    print('torch cuda version:', torch.version.cuda)
    print('cuda is available:', torch.cuda.is_available())
    print('cuda device count:', torch.cuda.device_count())
    print("cudnn version:", torch.backends.cudnn.version())

def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)
