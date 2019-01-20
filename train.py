#%%
import torch
from torchtext import data, vocab

import time
import datetime
import logging
import sys

from utils import execute_and_time, get_device, itos
from preprocess import Batch, embedding_param
from model import Transformer
from optimize import get_default_optimizer, train

LOG_FILE = False
logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
if LOG_FILE:
    file_handler = logging.FileHandler('log.out')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler) 
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

DATA_PATH = 'data/'
SAMPLE_DATA_PATH = f'{DATA_PATH}sample_data/'
PROCESSED_DATA_PATH = f'{DATA_PATH}processed_data/'

VERBOSE = True
def verbose_log(msg):
    if VERBOSE:
        logger.info(msg)

pre_trained_vector_type = 'glove.6B.200d' 
batch_size = 64
device = get_device()
stack_number = 6
heads_number = 8

#%%
if __name__ == "__main__":
    """ Data loading and preprocessing """
    tokenizer = data.get_tokenizer('spacy')
    TEXT = data.Field(tokenize=tokenizer, lower=True, eos_token='_eos_')
    trn_data_fields = [("source", TEXT), ("target", TEXT)]
    trn, vld = execute_and_time(
        data.TabularDataset.splits, 
        before_msg='Loading dataset...', 
        after_msg='Dataset loaded.',
        path=f'{SAMPLE_DATA_PATH}',
        train='train_ds.csv', 
        validation='valid_ds.csv',
        format='csv', 
        skip_header=True, 
        fields=trn_data_fields
    )

    execute_and_time(
        TEXT.build_vocab,
        trn,
        vectors=pre_trained_vector_type,
        before_msg='Building vocabulary...',
        after_msg='Vocabulary built.'
    )
    vocabulary = TEXT.vocab
    vocab_size = len(vocabulary)

    train_iter, val_iter = data.BucketIterator.splits(
        (trn, vld), 
        batch_sizes=(batch_size, int(batch_size * 1.6)),
        device=device, 
        sort_key=lambda x: len(x.source),
        shuffle=True, 
        sort_within_batch=False, 
        repeat=True
    )
    train_iter = Batch(train_iter, "source", "target")
    val_iter = Batch(val_iter, "source", "target")
    logger.info('Batch iterator created.')

    pre_trained_vector, embz_size, padding_idx = embedding_param(SAMPLE_DATA_PATH, TEXT, pre_trained_vector_type)

    """ Creating model """
    model = Transformer(
        embz_size,
        vocab_size,
        padding_idx,
        pre_trained_vector,
        stack_number,
        heads_number
    )

    """ Training """
    optimizer = get_default_optimizer(model)
    criterion = torch.nn.NLLLoss()
    train(model, train_iter, 5000, optimizer, criterion)






    