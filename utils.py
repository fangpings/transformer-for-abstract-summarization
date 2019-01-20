import torch
import torch.nn as nn
import copy
import logging
import time

logger = logging.getLogger()

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def get_device(GPU=0):
    if torch.cuda.is_available():
        assert GPU < 4
        logger.info(f'GPU available, using NO.{GPU}.')
        return torch.device('cuda', GPU)
    else:
        logger.info('GPU unavailable, using CPU.')
        return torch.device('cpu')

def itos(input_tensor, vocab):
    input_tensor = input_tensor.data.cpu().numpy()
    return ' '.join([vocab.itos[o] for o in input_tensor])

def execute_and_time(func, *args, **kwargs):
    before_msg = kwargs.pop('before_msg', None)
    after_msg = kwargs.pop('after_msg', None)
    customize = kwargs.pop('customize', None)
    if before_msg:
        logger.info(before_msg)
    time1 = time.time()
    output = func(*args, **kwargs)
    time2 = time.time()
    if not customize:
        logger.info(f'{after_msg} {time2 - time1}s elapesd.' if after_msg else f'{time2 - time1}s elapesd.')
        return output
    else:
        return output, time2 - time1