import torch
import torch.nn as nn
import copy
import logging
import time
import os

logger = logging.getLogger()

MAX_MODEL_NUM = 3
MODEL_PATH = './model'

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

def estimate_time(time2, time1):
    total = time2 - time1
    second = total % 60
    minutes = second // 60
    hours = minutes // 60
    minutes = minutes % 60
    if hours == 0:
        return f'{minutes}min {second}s'
    else:
        return f'{hours}h {minutes}min {second}s'

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

def save_model(model):
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    files = os.listdir(MODEL_PATH)
    files = filter(lambda x: x.endswith('.pt'), files)
    files = sorted(files, key=lambda x: os.path.getctime(os.path.join(MODEL_PATH, x)))
    if len(files) > MAX_MODEL_NUM:
        models_to_delete = MAX_MODEL_NUM - len(files)
        for i in range(models_to_delete):
            os.remove(os.path.join(MODEL_PATH, files[i]))
    time_str = '-'.join(time.strftime('%x_%X').split('/'))
    path = os.path.join(MODEL_PATH, 'model_' + time_str + '.pt')
    torch.save(model.state_dict(), path)
    logger.info('Model saved.')