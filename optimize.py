import torch
import torch.nn as nn
import logging

from utils import execute_and_time

logger = logging.getLogger()

class RevisedAdamOptimizer(object):
    """
    In the original paper, they used a revised adam optimizer that increasing the learning rate linearly for the ﬁrst warmup_steps training steps, 
    and decreasing it thereafter proportionally to the inverse square root of the step number.
    Actuall, pytorch implements several classes to dynamically change the learning rate, see torch.optim.lr_scheduler.LambdaLR
    """
    def __init__(self, model_size, warmup_steps, factor, optimizer):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.factor = factor  # not in original paper, but appear in reference code
        self._step = 0  # _step means don't want outside to call this variable
        self._lr = 0

    def update_lr(self):
        self._lr = self.factor * (self.model_size ** (-0.5) * min(self._step ** (-0.5), self._step * self.warmup_steps ** (-1.5)))

    def step(self):
        self._step += 1
        self.update_lr()
        self.optimizer.zero_grad()
        for p in self.optimizer.param_groups:  # it seems parameters are divided into groups, and for different groups we have different lr
            p['lr'] = self._lr
        self.optimizer.step()

def get_default_optimizer(model):
    return RevisedAdamOptimizer(model.model_size, 4000, 2, torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9))

# In original paper and reference paper, they use label smoothing in loss computation, but here we do not implement it. (wondering whether it will still work)
def run_one_train(model, batch_tuple, optimizer, criterion):
    """
    source/target -> (batch_size, seq_len)
    output -> batch_size, seq_len, vocab_size
    """
    source, target, src_mask, tgt_mask = batch_tuple
    """
    the structure of transformer requires source and target to be of same length
    it is understandable in NMT because usually source and target are approximately of same length
    but in summarization, usually target is much shorter.
    TODO: We need to change the structure to solve this problem
    """
    assert source.size(-1) == target.size(-1)
    batch_size = source.size(0)
    output = model(source, target, src_mask, tgt_mask)
    loss = 0
    for i in range(batch_size):
        loss += criterion(output[i], target[i])
    loss.backward()
    optimizer.step()  # optimizer.zero_grad() already included

    return loss.item() / batch_size

def train(model, batch_iter, iters, optimizer, criterion, print_every=500):
    logger.info('Start traning')
    for i, batch in enumerate(batch_iter):
        if i == iters:
            logger.info('Training over.')
            return
        if i % print_every:
            loss, time = execute_and_time(run_one_train, model, batch, optimizer, criterion, customize=True)
            remaining_time = (iters - i) * time
            rem_str = '%d min %d s' % (time // 60, time % 60)
            logger.info(f'Iteration: {i}, loss: {loss}, estimated remaining time: {rem_str}')
        else:
            run_one_train(model, batch, optimizer, criterion)
        






