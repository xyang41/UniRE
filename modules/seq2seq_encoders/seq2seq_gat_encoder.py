import torch
import torch.nn as nn
import math
from utils.nn_utils import gelu
from modules.token_embedders.bert_encoder import BertLinear


class GATEncoder(nn.Module):
    """This class is bidirectional gcn encoder
    """
    def __init__(self, input_size, num_layers=2, is_bidirectional=True, gat_dropout=0.1, adj_dropout=0.0, prune=False, min_threshold=0.1, max_threshold=0.4):
        """This function sets bidirectional graph neural network parameters

        Arguments:
            input_size {int} -- input dimensions

        Keyword Arguments:
            num_layers {int} -- the number of layers (default: {1})
            is_bidirectional {bool} -- whether bidirectional or not (default: {True})
            gat_dropout {float} -- dropout rate between gat layers (default: {0.5})
            adj_dropout {float} -- dropout rate for adjacent matrices (default: {0.0})
            prune {bool} -- whether to hard-prune the adjacent matrices (default: {False})
            min_threshold {int} -- minimum threshold for pruning, if less than this, weight = 0 (default: {0.1})
            max_threshold {int} -- minimum threshold for pruning, if bigger than this, weight = 1 (default: {0.4})
        """

        super().__init__()
        self.is_bidirectional = is_bidirectional
        #assert is_bidirectional == True, "Using undefined unidirectional GCN now!"

        self.input_size = input_size
        self.num_layers = num_layers
        # pruning related
        self.prune = prune
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.gat = GAT(input_size, input_size)
        self.layernorm = nn.LayerNorm(input_size)
        
        if gat_dropout > 0:
            self.gat_dropout = nn.Dropout(p=gat_dropout)
        else:
            self.gat_dropout = lambda x: x
        
        self.adj_dropout = nn.Dropout(p=adj_dropout, inplace=True) if adj_dropout > 0 else lambda x : x
        
    def get_input_dims(self):
        return self.input_size

    def get_output_dims(self):
        return self.input_size

    def get_num_layers(self):
        return self.num_layers

    def forward(self, inputs, adj_fw):
        """This function propagates forwardly

        Arguments:
            inputs {tensor} -- input data, shape: (batch_size, sequence_len, input_size)
            adj_fw {tensor} -- forward adjacent matrix of each sequence for the bidirectional gcn

        Returns:
            tensor -- output after gcn
        """

        batch_size, sequence_len, input_size = inputs.size()

        assert batch_size == len(adj_fw), \
            "batch size is not equal to the size of sequence adj fw list"
        assert input_size == self.input_size, \
            "input size of input data is not equal to `BiLSTMEncode` input size"
        
        adj_fw = adj_fw + adj_fw.permute(0, 2, 1)
        denom_row = torch.sum(adj_fw, dim=2).unsqueeze(-1)
        denom_row[denom_row==0] = 1
        adj_fw = adj_fw / denom_row

        # GAT
        outputs = self.gat(inputs, adj_fw)
        outputs = self.layernorm(self.gat_dropout(inputs + outputs))

        return outputs


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GAT, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.W = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.empty(self.out_dim, ))
        self.g = nn.LeakyReLU()

        self.init()
    
    def init(self):
        stdv = 1/math.sqrt(self.out_dim)
        # randomly fill the param from a uniform distribution
        #self.W.data.uniform_(-stdv, stdv)
        #self.b.data.uniform_(-stdv, stdv)
        
        # orthogonal init
        nn.init.orthogonal_(self.W)
        nn.init.zeros_(self.b)
    
    # inp: a batch of input word_emb_seq, adj: a batch of adjacent matrix
    def forward(self, inp, adj):
        batch_size, max_len, feat_dim = inp.size()
        attn_matrix = torch.matmul(inp, inp.transpose(1, 2))
        attn_score = attn_matrix / feat_dim ** 0.5

        #softmax
        exp_attn_score = torch.exp(attn_score)
        #print(exp_attn_score.size())
        #input()
        exp_attn_score = torch.mul(exp_attn_score, adj)
        #print(exp_attn_score.size())
        #input()
        sum_attn_score = torch.sum(exp_attn_score, dim=-1).unsqueeze(dim=-1)
        #print(sum_attn_score.size())
        #input()
        attn_score = torch.div(exp_attn_score, sum_attn_score + 1e-10)
        #print(attn_score.size())
        #input()

        out = torch.matmul(inp, self.W) 
        out = torch.matmul(attn_score, out)
        out = out + self.b

        return self.g(out)
    
    def __repr__(self):
        return self.__class__.__name__+'(in_dim=%d, out_dim=%d)'%(self.in_dim, self.out_dim)

