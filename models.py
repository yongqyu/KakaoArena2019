import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class RNN(nn.Module):
    def __init__(self, num_readers, num_writers, num_keywd, num_items, num_magazine, latent_dim):
        super(RNN, self).__init__()

        self.num_readers = num_readers
        self.num_writers = num_writers
        self.num_keywd = num_keywd
        self.num_items = num_items
        self.num_magazine = num_magazine
        self.latent_dim = latent_dim
        self.num_layer = 2
        self.seq_length = 5

        #skipgram = Skipgram(num_readers, num_writers, latent_dim)
        #skipgram.load_state_dict(torch.load('./models/9_rnn_pretrain.pkl'))
        self.embedding_reader = nn.Embedding(num_embeddings=self.num_readers, embedding_dim=self.latent_dim)
        #self.embedding_reader.weight.data.copy_(skipgram.embedding_reader.weight)
        #self.embedding_items = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_writer = nn.Embedding(num_embeddings=self.num_writers, embedding_dim=self.latent_dim)
        #self.embedding_writer.weight.data.copy_(skipgram.embedding_writer.weight)
        self.embedding_keywd = nn.Embedding(num_embeddings=self.num_keywd, embedding_dim=self.latent_dim, padding_idx=1)
        self.embedding_magazine = nn.Embedding(num_embeddings=self.num_magazine, embedding_dim=self.latent_dim)
        self.embedding_rt = torch.from_numpy(np.load('/data/private/Arena/prepro_results/ts_array.npy'))

        self.keywd_rnn = nn.GRU(self.latent_dim, self.latent_dim, self.num_layer, batch_first=True)
        self.items_rnn = nn.GRU(2*self.latent_dim, self.latent_dim+1, self.num_layer, batch_first=True)
        self.items_fc = nn.Linear(3*self.latent_dim+1, 2*self.latent_dim)
        self.item_idx_fc = nn.Linear(in_features=self.latent_dim+1, out_features=self.num_items)
        self.writer_idx_fc = nn.Linear(in_features=self.latent_dim+1, out_features=self.num_writers)
        self.keywd_idx_fc = nn.Linear(in_features=self.latent_dim+1, out_features=self.num_keywd)

        nn.init.xavier_uniform_(self.items_fc.weight)
        nn.init.xavier_uniform_(self.item_idx_fc.weight)
        nn.init.xavier_uniform_(self.writer_idx_fc.weight)
        nn.init.xavier_uniform_(self.keywd_idx_fc.weight)
        for name, param in list(self.keywd_rnn.named_parameters()) + \
                               list(self.items_rnn.named_parameters()):
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)

    def forward(self, data):
        # batch_size x length x reader readat item_id writer keywd*5 reg_ts magazine_id
        reader_embedding = self.embedding_reader(data[:,0,0].long())
        readat_embedding = data[:,0,1].float()
        #item_embedding = self.embedding_items(data[:,:,2].long())
        writer_embedding = self.embedding_writer(data[:,:,3].long())
        keywd_embedding = self.embedding_keywd(data[:,:,4:9].long())
        reg_embedding = (data[:,:,9] - 1430409600)/8640000
        magazine_embedding = self.embedding_magazine(data[:,:,10].long())

        hidden, _ = self.keywd_rnn(keywd_embedding.view(-1,self.seq_length,self.latent_dim)) # reader_embedding
        keywd_embedding = hidden[:,-1,:].view(data.size(0), data.size(1), self.latent_dim)
        merged_item_embedding = self.items_fc(torch.cat([reg_embedding.unsqueeze(-1).float(),
                                                        writer_embedding, #item_embedding,
                                                        keywd_embedding, magazine_embedding], 2))
        reader_embedding = torch.cat([reader_embedding, readat_embedding.unsqueeze(-1)], 1)
        reader_embedding = reader_embedding.unsqueeze(0).expand(self.num_layer, -1, -1).contiguous()
        hidden, _ = self.items_rnn(merged_item_embedding, reader_embedding)
        #output = self.fc(torch.cat([hidden[:,-1], reader_embedding[0]], 1))#, readat_embedding], 1))
        item_output = self.item_idx_fc(hidden[:,-1])#, readat_embedding], 1))
        writer_output = self.writer_idx_fc(hidden[:,-1])
        keywd_output = self.keywd_idx_fc(hidden[:,-1])

        return item_output, writer_output, keywd_output


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    """

    def __init__(self, encoder_layer, num_layers,
                 num_readers, num_writers, num_keywd, num_items, num_magazine, latent_dim, norm=None):
        super(TransformerEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.item_seq_length = 4
        self.keywd_seq_length = 4

        self.embedding_reader = nn.Embedding(num_embeddings=num_readers, embedding_dim=latent_dim)
        self.embedding_items = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)
        self.embedding_writer = nn.Embedding(num_embeddings=num_writers, embedding_dim=latent_dim)
        self.embedding_keywd = nn.Embedding(num_embeddings=num_keywd, embedding_dim=latent_dim//self.keywd_seq_length, padding_idx=1)
        self.embedding_magazine = nn.Embedding(num_embeddings=num_magazine, embedding_dim=latent_dim)

        self.items_fc = nn.Linear(self.item_seq_length*latent_dim+3, latent_dim)
        self.fc = nn.Linear(self.item_seq_length*latent_dim, num_items)

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, data, mask=None):
        r"""Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            src_mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        reader_embedding = self.embedding_reader(data[:,0,0].long())
        item_embedding = self.embedding_items(data[:,:,1].long())
        writer_embedding = self.embedding_writer(data[:,:,2].long())
        keywd_embedding = self.embedding_keywd(data[:,:,4:8].long())
        keywd_embedding = keywd_embedding.view(keywd_embedding.size(0), keywd_embedding.size(1), -1)
        magazine_embedding = self.embedding_magazine(data[:,:,11].long())
        reg_ts_gap = data[:,:,8:11].float()

        merged_item_embedding = self.items_fc(torch.cat([item_embedding, writer_embedding,
                                                        keywd_embedding, magazine_embedding, reg_ts_gap], 2))

        target = reader_embedding.unsqueeze(0)
        output = merged_item_embedding.permute(1,0,2)

        for i in range(self.num_layers):
            output = self.layers[i](target, output, mask)

        if self.norm:
            output = self.norm(output)

        return self.fc(output.permute(1,0,2).contiguous().view(data.size(0), -1))


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, src, src_mask=None):
        r"""Pass the input through the endocder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(query, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
