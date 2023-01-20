import logging

import torch
import torch.nn as nn

from modules.seq2seq_encoders.seq2seq_gcn_encoder import GCNEncoder

logger = logging.getLogger(__name__)

class GNNBiaffine(nn.Module):
    def __init__(self, cfg, input_size, vocab):
        super().__init__()

        self.gcn_layers = cfg.gcn_layers
        self.input_size = input_size
        self.vocab = vocab

        self.gcn_encoder = GCNEncoder(input_size=self.input_size,
                                          num_layers=cfg.gcn_layers,
                                          gcn_dropout=cfg.gcn_dropout,
                                          adj_dropout=cfg.adj_dropout,
                                          prune=cfg.prune,
                                          min_threshold=cfg.min_threshold,
                                          max_threshold=cfg.max_threshold)
        self.encoder_output_size = self.gcn_encoder.get_output_dims()
        '''
        self.U = nn.Parameter(
            torch.FloatTensor(self.vocab.get_vocab_size('ent_rel_id'), self.input_size + 1,
                              self.input_size + 1))
        '''
        self.U = nn.Parameter(torch.FloatTensor(1, self.input_size + 1, self.input_size + 1))
        
        self.U.data.zero_()

    def forward(self, batch_inputs):
        batch_seq_tokens_encoder_repr = batch_inputs['seq_encoder_reprs']
        
        batch_seq_adj= batch_inputs['adj_fw']
        batch_seq_encoder_repr = self.gcn_encoder(batch_seq_tokens_encoder_repr, batch_seq_adj)          

        batch_seq_encoder_repr = torch.cat(
            [batch_seq_encoder_repr,
             torch.ones_like(batch_seq_encoder_repr[..., :1])], dim=-1)

        batch_joint_score = torch.einsum('bxi, oij, byj -> boxy', batch_seq_encoder_repr, self.U,
                                         batch_seq_encoder_repr).permute(0, 2, 3, 1)
        
        #print("size of batch_joint_score: ", batch_joint_score.size())
        #print(batch_seq_encoder_repr.size(), self.U.size())
        #input()
        return batch_joint_score
