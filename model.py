import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import clones
import math
from preprocess import Batch

DROP_PROB = 0.1  # in original paper, they use this value

class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See citation for details).
    """
    def __init__(self, model_size, eps=1e-6):  # model_size: size of embedding/input/output
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(model_size))
        self.b_2 = nn.Parameter(torch.zeros(model_size))
        self.eps = eps

    def forward(self, input_tensor):
        """
        input_tensor -> (batch_size, seq_len, emb_size)
        """
        mean = input_tensor.mean(-1, keepdim=True)  # w.r.t last dim, make normalization within every position
        std = input_tensor.std(-1, keepdim=True)
        return self.a_2 * (input_tensor - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    A copy from already implemented module.
    TODO: understand how it works
    """
    def __init__(self, model_size, dropout=DROP_PROB, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, model_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_size, 2).float() *
                             -(math.log(10000.0) / model_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, input_tensor):
        input_tensor = input_tensor + Variable(self.pe[:, :input_tensor.size(1)], requires_grad=False)
        return self.dropout(input_tensor)

class PositionFeedForward(nn.Module):
    def __init__(self, model_size, dropout=DROP_PROB):
        super().__init__()
        self.model_size = model_size
        self.dropout = nn.Dropout(dropout)
        self.linear = clones(nn.Linear(self.model_size, self.model_size), 2)

    def forward(self, input_tensor):
        return self.dropout(self.linear[0](F.relu(self.linear[1](input_tensor))))
 
    
class MultiHeadAttention(nn.Module):
    def __init__(self, model_size, heads_number, vector_size, dropout=DROP_PROB):
        super().__init__()
        """ Parameters """
        self.heads_number = heads_number
        self.vector_size = vector_size
        self.model_size = model_size
        assert model_size % heads_number == 0 

        """ Layers """
        self.dropout = nn.Dropout(dropout)
        self.vector_matrix = clones(nn.Linear(self.model_size, self.model_size), 4)

    def forward(self, query, key, value, mask=None):
        """
        input query/key/value: -> (batch_size, seq_len, emb_size)
        output query/key/value/mask -> (batch_size, heads_num, seq_len, vector_size)
        """
        batch_number = query.size(0)
        # get the 3 vectors, transpose so heads are not invloved in further attention calculation, heads are like channels and should not be involved until we cat them
        query, key, value = \
            [matrix(x).view(batch_number, -1, self.heads_number, self.vector_size).transpose(1, 2) 
                for matrix, x in zip(self.vector_matrix, [query, key, value])]
        vector_size = query.size(-1)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(vector_size)  # QK^T/sqrt(d_k), torch.matmul() only multiply w.r.t. last two dims when handling batched vectors
        if mask is not None:  # QUESTION: When shouldn't we use mask?
            scores = scores.masked_fill(mask == 0, -1e9)  # Mask is an upper triangle matrix to prevent input i from seeing the input j>i, set to -1e9 because it will pass through a softmax
        attention_weights = self.dropout(F.softmax(scores, dim=-1))
        output_tensor = torch.matmul(attention_weights, value)  # in the same shape of input
        # tensor.contiguous: see https://stackoverflow.com/questions/48915810/pytorch-contiguous#, in short it creates a new tensor with same content but new memory
        output_tensor = output_tensor.transpose(1, 2).contiguous().view(batch_number, -1, self.heads_number * self.vector_size)  # heads_number * vector_size = model_size
        return self.vector_matrix[-1](output_tensor)
        

class ResidualNormWrap(nn.Module):
    def __init__(self, model_size, dropout=DROP_PROB):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalization = LayerNorm(model_size)
    
    def forward(self, input_tensor, sublayer):
        # execute a residual and layer normalization
        return self.normalization(input_tensor + self.dropout(sublayer(input_tensor)))


class EncoderCell(nn.Module):
    def __init__(self, heads_number, model_size, dropout=DROP_PROB):
        super().__init__()

        """ Parameters """
        self.heads_number = heads_number  # number of attention heads
        self.model_size = model_size  # last dim of the input, the embedding/model size
        assert model_size % heads_number == 0  # make this assertion so we can handle the shape of 3 vectors more easily
        self.vector_size = model_size // heads_number  # size of the query, key and value vectors

        """ Layers """
        self.self_attn = MultiHeadAttention(self.model_size, self.heads_number, self.vector_size)
        self.feed_forward = PositionFeedForward(self.model_size)
        self.wrap = clones(ResidualNormWrap(self.model_size), 2)

    def forward(self, input_tensor, mask):
        """
        input_tensor -> (batch_size, seq_len, emb_size): embedding of the input or last cell's output
        """
        """
        Flow: self attention -> layer normalization -> feed forward -> layer normalization
        """
        return self.wrap[0](self.wrap[1](input_tensor, lambda x: self.self_attn(x, x, x, mask)), self.feed_forward)
    
class DecoderCell(nn.Module):
    def __init__(self, heads_number, model_size, dropout=DROP_PROB):
        super().__init__()

        """ Parameters """
        self.heads_number = heads_number  
        self.model_size = model_size 
        assert model_size % heads_number == 0  
        self.vector_size = model_size // heads_number  

        """ Layers """
        self.self_attn = MultiHeadAttention(self.model_size, self.heads_number, self.vector_size)
        self.intra_attn = MultiHeadAttention(self.model_size, self.heads_number, self.vector_size)
        self.feed_forward = PositionFeedForward(self.model_size)
        self.wrap = clones(ResidualNormWrap(self.model_size), 3)
    
    def forward(self, input_tensor, encoder_output, mask):
        """
        Flow: self attention -> layer normalization -> intra-attention -> layer normalization -> feed forward -> layer normalization
        """
        output_tensor = self.wrap[0](input_tensor, lambda x: self.self_attn(x, x, x, mask))
        output_tensor = self.wrap[1](output_tensor, lambda x: self.intra_attn(x, encoder_output, encoder_output, mask))
        return self.wrap[2](output_tensor, self.feed_forward)

class Transformer(nn.Module):
    def __init__(self, model_size, vocabulary_size, padding_index, pretrained_embedding, stack_number, heads_number, dropout=(DROP_PROB, DROP_PROB)):
        super().__init__()

        """ Parameters """
        self.model_size = model_size
        self.vocabulary_size = vocabulary_size  # size of the total vocabulary
        self.padding_index = padding_index  # decide the index that <pad> token use in embedding
        self.pretrained_embedding = pretrained_embedding  # type: nn.Embedding()
        self.encoder_dropout, self.decoder_dropout = dropout  # dropout[0]: encoder dropout probability, dropout[1]: decoder
        self.stack_number = stack_number  # decide how many encoder/decoder cells are stakced
        self.heads_number = heads_number

        """ Layers """
        self.encoder_embedding_layer = nn.Embedding(self.vocabulary_size, self.model_size, self.padding_index)
        if pretrained_embedding:
            self.encoder_embedding_layer.weight.data.copy_(self.pretrained_embedding.weight.data)  # If we have pretrained embedding weight, copy it to our own embedding layer
        self.positional_encoding = PositionalEncoding(self.model_size)
        self.encoder_dropout = nn.Dropout(self.encoder_dropout)
        self.encoder_cells = clones(EncoderCell(self.heads_number, self.model_size), self.stack_number)

        self.decoder_embedding_layer = nn.Embedding(self.vocabulary_size, self.model_size, self.padding_index)
        if pretrained_embedding:
            self.decoder_embedding_layer.weight.data.copy_(self.pretrained_embedding.weight.data)  # If we have pretrained embedding weight, copy it to our own embedding layer
        self.positional_encoding = PositionalEncoding(self.model_size)
        self.decoder_dropout = nn.Dropout(self.decoder_dropout)
        self.decoder_cells = clones(DecoderCell(self.heads_number, self.model_size), self.stack_number)

        self.last_liner = nn.Linear(self.model_size, self.vocabulary_size)

    def encoder(self, input_tensor, mask):
        """
        embedding -> (batch_size, seq_len, emb_size): embedding of input tensor
        """
        """
        Flow: embedding -> positional encoding -> (encoder_cell)*n
        """
        output_tensor = self.encoder_dropout(self.encoder_embedding_layer(input_tensor))
        output_tensor = self.positional_encoding(output_tensor)
        for i in range(self.stack_number):
            output_tensor = self.encoder_cells[i](output_tensor, mask)
        return output_tensor
    
    def decoder(self, input_tensor, encoder_output, mask):
        output_tensor = self.decoder_dropout(self.decoder_embedding_layer(input_tensor))
        output_tensor = self.positional_encoding(output_tensor)
        for i in range(self.stack_number):
            output_tensor = self.decoder_cells[i](output_tensor, encoder_output, mask)
        return output_tensor
    
    def forward(self, input_tensor, target_tensor, encoder_mask, decoder_mask):
        """
        input_tensor -> (batch_size, seq_len)
        output_tensor -> (batch_size, seq_len, vocab_size)
        """     
        encoder_output = self.encoder(input_tensor, encoder_mask)
        decoder_output = self.decoder(target_tensor, encoder_output, decoder_mask)
        return F.log_softmax(self.last_liner(decoder_output), dim=-1)
    
    def greedy_decode(self, input_tensor, max_length, vocabulary):
        """
        input_tensor -> (batch_size, seq_len)
        """
        encoder_mask = Batch.create_src_mask(input_tensor, vocabulary)
        encoder_output = self.encoder(input_tensor, encoder_mask)
        batch_size = input_tensor.size(0)
        sos = vocabulary.stoi['<sos>']
        output = torch.ones(batch_size, 1).fill_(sos).type_as(input_tensor.data)
        for i in range(max_length - 1):
            decoder_mask = Batch.create_tgt_mask(output, vocabulary)
            decoder_output = self.decoder(output, encoder_output, decoder_mask)
            next_word = F.log_softmax(self.last_liner(decoder_output), dim=-1)
            next_word = torch.argmax(next_word, dim=-1)
            output = torch.cat((output, next_word), dim=-1)
        
        return output
