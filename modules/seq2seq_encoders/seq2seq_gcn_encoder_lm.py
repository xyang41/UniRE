import torch
import torch.nn as nn
import math
from utils.nn_utils import gelu
from modules.token_embedders.bert_encoder import BertLinear


class GCNEncoder(nn.Module):
    """This class is bidirectional gcn encoder
    """
    def __init__(self, input_size, num_layers=2, is_bidirectional=True, gcn_dropout=0.1, adj_dropout=0.0, prune=False, min_threshold=0.1, max_threshold=0.4):
        """This function sets bidirectional graph neural network parameters

        Arguments:
            input_size {int} -- input dimensions

        Keyword Arguments:
            num_layers {int} -- the number of layers (default: {1})
            is_bidirectional {bool} -- whether bidirectional or not (default: {True})
            gcn_dropout {float} -- dropout rate between gcn layers (default: {0.5})
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

        self.gcns = nn.ModuleList()
        for i in range(num_layers):            
            self.gcns.append(nn.ModuleDict())
            if is_bidirectional:
                self.gcns[-1]['fw'] = GCN(input_size, input_size // 2)
                self.gcns[-1]['bw'] = GCN(input_size, input_size // 2)
            else:
                self.gcns[-1]['fw'] = GCN(input_size, input_size)

        self.layernorm = nn.LayerNorm(input_size)
        
        if gcn_dropout > 0:
            self.gcn_dropout = nn.Dropout(p=gcn_dropout)
        else:
            self.gcn_dropout = lambda x: x
        
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
        
        # hard-prune
        if self.prune:
            if self.max_threshold <= 0:
                assert adj_fw.dim() == 3, "The size of adj_fw should be [batch_size, sequence_len, sequence_len]"
                adj_fw = nn.functional.softmax(adj_fw, 2)
            else:
                adj_fw[adj_fw < self.min_threshold] = 0
                adj_fw[adj_fw >= self.max_threshold] = 1

        # process adj matrices
        denom_row = torch.sum(adj_fw, dim=2).unsqueeze(-1)
        denom_col = torch.sum(adj_fw, dim=1).unsqueeze(-1)
        denom_row[denom_row==0], denom_col[denom_col==0] = 1, 1
        
        adj_fw = adj_fw / denom_row
        adj_fw = self.adj_dropout(adj_fw)
        adj_bw = adj_fw.permute(0, 2, 1) / denom_col

        # stack GCNs
        for i in range(self.num_layers):
            repr_fw = self.gcns[i]['fw'](inputs, adj_fw)
            if self.is_bidirectional:
                repr_bw = self.gcns[i]['bw'](inputs, adj_bw)
                outputs = torch.cat([repr_fw, repr_bw], dim=-1)
            else:
                outputs = repr_fw
            
            #outputs = self.gcn_dropout(outputs)
            #outputs = self.layernorm(inputs + outputs)
            outputs = self.gcn_dropout(self.layernorm(outputs))
            inputs = outputs

        return outputs


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.W = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.empty(self.out_dim, ))
        
        self.init()
    
    def init(self):
        stdv = 1/math.sqrt(self.out_dim)
        # randomly fill the param from a uniform distribution
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)
        
        # orthogonal init
        #nn.init.orthogonal(self.W.data)
    
    # inp: a batch of input word_emb_seq, adj: a batch of adjacent matrix
    def forward(self, inp, adj, is_relu=True):
        out = torch.matmul(inp, self.W) 
        out = torch.matmul(adj, out)
        out = out + self.b
        
        if is_relu==True:
            out = nn.functional.relu(out)
        
        return out
    
    def __repr__(self):
        return self.__class__.__name__+'(in_dim=%d, out_dim=%d)'%(self.in_dim, self.out_dim)

