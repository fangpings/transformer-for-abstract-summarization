import torch
import torch.nn as nn
import torchtext
import logging
import numpy as np

logger = logging.getLogger()

class Batch():
    def __init__(self, dataset, source, target, device, vocabulary=None):
        self.dataset, self.source, self.target = dataset, source, target
        self.vocabulary = vocabulary
        self.device = device
        
    def __next__(self):
        # self.init_epoch()
        for batch in self.dataset:
            source_batch = getattr(batch, self.source) 
            target_batch = getattr(batch, self.target)
            if self.vocabulary is not None:
                src_mask = self.create_src_mask(source_batch)
                tgt_mask = self.create_tgt_mask(target_batch) 
                return (source_batch.transpose(0, 1), target_batch.transpose(0, 1), src_mask, tgt_mask)
            else:
                return (source_batch, target_batch)
    
    def __iter__(self):
        """
        for a in b expression does 2 things: 
        1. get an iterator by calling __iter__()
        2. in the loop, calling __next__()
        """
        return self
            
    def __len__(self):
        return len(self.dataset)

    """
    Masking plays an important role in the transformer. It serves two purposes:
    - In the encoder and decoder: To zero attention outputs wherever there is just padding in the input sentences.
    - In the decoder: To prevent the decoder ‘peaking’ ahead at the rest of the translated sentence when predicting the next word.
    """
    def create_src_mask(self, batch):
        """
        batch(transposed) -> (batch_size, seq_len)
        mask -> (batch_size, 1, 1, seq_len)

        QEUSTION: the effect of broadcasting on mask is still not clear. Needs more test.

        In the place where mask is used:
            if mask is not None: 
                scores = scores.masked_fill(mask == 0, -1e9)
        where scores -> (batch_size, heads_num, seq_len, seq_len)
        """
        assert self.vocabulary is not None
        batch = batch.transpose(0, 1)
        pad_index = self.vocabulary.stoi['<pad>']
        return (batch != pad_index).unsqueeze(1).unsqueeze(1)
    
    def create_tgt_mask(self, batch):
        """
        target mask -> (batch_size, 1, seq_len, seq_len)
        """
        assert self.vocabulary is not None
        batch = batch.transpose(0, 1)
        pad_index = self.vocabulary.stoi['<pad>']
        mask1 = (batch != pad_index).unsqueeze(1)

        seq_len = batch.size(1)
        mask2 = np.triu(np.ones((1, seq_len, seq_len), dtype='uint8'), k=1)
        mask2 = (torch.from_numpy(mask2).to(self.device)) == 0

        return (mask1 & mask2).unsqueeze(1)

    def init_epoch(self):
        self.dataset.init_epoch()

def norm_pre_trained_embeddings(vecs, itos, em_sz, padding_idx, vec_mean, vec_std):
    """
    Function to load and normalize pretrained vectors
    itos -> List: ['<sos>, '<pad>', ...]
    vecs -> (vocab_size, emb_size)
    """
    emb = nn.Embedding(len(itos), em_sz, padding_idx=padding_idx)  # padding_index:pads the output with the embedding vector at padding_idx (initialized to zeros) whenever it encounters the index.
    emb.weight.data = torch.from_numpy((vecs - vec_mean) / vec_std)
    emb.weight.requires_grad = False    
    return emb


def embedding_param(path, data_field, pre_trained_vector_type):
    """Returns embedding parameters"""
    pre_trained = None
    padding_idx = data_field.vocab.stoi['<pad>']
    index_to_string, string_to_index = data_field.vocab.itos, data_field.vocab.stoi

    if pre_trained_vector_type:
        vec_mean, vec_std = data_field.vocab.vectors.numpy().mean(), data_field.vocab.vectors.numpy().std()
        logger.info('pre_trained_vector_mean = %s, pre_trained_vector_std = %s' % (vec_mean, vec_std))
        vector_weight_matrix = data_field.vocab.vectors
        embz_size = vector_weight_matrix.size(1)
        logger.info('Normalizing embeddings...')
        pre_trained = norm_pre_trained_embeddings(vector_weight_matrix.numpy(), index_to_string, embz_size, padding_idx, vec_mean, vec_std)
        logger.info('pre_trained_vector_mean = %s, pre_trained_vector_std = %s' % (pre_trained.weight.data.numpy().mean(), pre_trained.weight.data.numpy().std()))
    return pre_trained, embz_size, padding_idx