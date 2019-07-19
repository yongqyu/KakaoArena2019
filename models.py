import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class RNN(nn.Module):
    def __init__(self, num_readers, num_writers, num_keywd, num_items, num_magazine, latent_dim, dev_tensor):
        super(RNN, self).__init__()

        self.num_readers = num_readers
        self.num_writers = num_writers
        self.num_keywd = num_keywd
        self.num_items = num_items
        self.num_magazine = num_magazine
        self.latent_dim = latent_dim
        self.dev_tensor = dev_tensor
        self.num_layer = 2
        self.keywd_seq_length = 5

        self.embedding_reader = nn.Embedding(num_embeddings=self.num_readers, embedding_dim=self.latent_dim)
        self.embedding_writer = nn.Embedding(num_embeddings=self.num_writers, embedding_dim=self.latent_dim)
        self.embedding_keywd = nn.Embedding(num_embeddings=self.num_keywd, embedding_dim=self.latent_dim, padding_idx=1)
        self.embedding_magazine = nn.Embedding(num_embeddings=self.num_magazine, embedding_dim=self.latent_dim)

        self.ts_min_embedding = nn.Parameter(torch.Tensor(1,self.latent_dim), requires_grad=True)
        self.ts_max_embedding = nn.Parameter(torch.Tensor(1,self.latent_dim), requires_grad=True)
        nn.init.xavier_normal_(self.ts_min_embedding.data)
        nn.init.xavier_normal_(self.ts_max_embedding.data)

        self.encoder_rnn = nn.GRU(self.latent_dim, 2*self.latent_dim, self.num_layer, batch_first=True)
        self.decoder_attn = nn.MultiheadAttention(2*self.latent_dim, 8, 0.1)
        self.encoder_fc = nn.Linear(4*self.latent_dim, self.latent_dim)
        self.decoder_fc = nn.Linear(4*self.latent_dim, 2*self.latent_dim)
        #self.writer_idx_fc = nn.Linear(in_features=2*self.latent_dim, out_features=self.num_writers)
        self.keywd_idx_fc = nn.Linear(in_features=self.latent_dim, out_features=self.num_keywd)

        self.predict_bias = nn.Parameter(torch.Tensor(1,self.num_items), requires_grad=True)
        nn.init.xavier_normal_(self.predict_bias)

        nn.init.xavier_uniform_(self.encoder_fc.weight)
        nn.init.xavier_uniform_(self.decoder_fc.weight)
        #nn.init.xavier_uniform_(self.writer_idx_fc.weight)
        nn.init.xavier_uniform_(self.keywd_idx_fc.weight)
        for name, param in list(self.encoder_rnn.named_parameters()):# + \
                           #list(self.decoder_rnn.named_parameters()):
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)

    def interpolate(self, input):
        return self.ts_max_embedding*input + self.ts_min_embedding*(1-input)

    def forward(self, data, mode='Train'):
        # batch_size x length x reader readat*2 item_id writer keywd*5 reg_ts magazine_id
        ## reader + readat
        reader_embedding = self.embedding_reader(data[:,0,0].long())
        if mode == 'Train':
            readat_embedding_0 = self.interpolate(data[:,0,1].float().unsqueeze(-1))
            #readat_embedding_1 = self.interpolate(data[:,0,2].float().unsqueeze(-1))
            #readat_embedding = torch.stack([readat_embedding_0, readat_embedding_1], 1)
            readat_embedding = readat_embedding_0.unsqueeze(1)
        elif mode == 'Valid':
            readat_embedding = self.interpolate(data[:,0,1].float().unsqueeze(-1)).unsqueeze(1)
        elif mode == 'Test':
            readat_embedding_0 = self.interpolate(torch.tensor([[0.915]]*data.size(0)).cuda())
            readat_embedding_1 = self.interpolate(torch.tensor([[0.951]]*data.size(0)).cuda())
            readat_embedding_2 = self.interpolate(torch.tensor([[0.99]]*data.size(0)).cuda())
            #readat_embedding = torch.stack([readat_embedding_0, readat_embedding_1, readat_embedding_2], 1)
            readat_embedding = readat_embedding_0.unsqueeze(1)

        ## item elements
        writer_embedding = self.embedding_writer(data[:,:,3])
        keywd_embedding = torch.mean(self.embedding_keywd(data[:,:,4:9]), 2)
        reg_embedding = self.interpolate(data[:,:,9].float().view(-1,1)).view(-1,writer_embedding.size(1),self.latent_dim)
        magazine_embedding = self.embedding_magazine(data[:,:,10])

        ## encode + decode
        merged_item_embedding = self.encoder_fc(torch.cat([reg_embedding, writer_embedding, #item_embedding,
                                                           keywd_embedding, magazine_embedding], 2))
        hidden, _ = self.encoder_rnn(merged_item_embedding)#, reader_embedding)
        hidden = hidden.permute(1,0,2)
        query_embedding = torch.cat([reader_embedding.unsqueeze(0), readat_embedding.permute(1,0,2)], 2)
        output, _ = self.decoder_attn(query_embedding, hidden, hidden)

        ## dev embedding
        dev_writer_emb = self.embedding_writer(self.dev_tensor[:,1].long())
        dev_keywd_emb = torch.mean(self.embedding_keywd(self.dev_tensor[:,2:7].long()), 1)
        dev_reg_emb = self.interpolate(self.dev_tensor[:,7].float().unsqueeze(-1))
        dev_magazine_emb = self.embedding_magazine(self.dev_tensor[:,8].long())
        merged_dev_embedding = self.decoder_fc(torch.cat([dev_reg_emb, dev_writer_emb,
                                                          dev_keywd_emb, dev_magazine_emb], 1))
        merged_dev_embedding = F.layer_norm(merged_dev_embedding, [merged_dev_embedding.size(-1)])

        ## predict
        item_output = torch.einsum('bld,dv->blv', output.permute(1,0,2), merged_dev_embedding.t())
        item_output += self.predict_bias.expand_as(item_output)
        #writer_output = self.writer_idx_fc(output)
        #keywd_output = self.keywd_idx_fc(output)

        return item_output, None#keywd_output#, None, None#



class CNN(nn.Module):
    def __init__(self, num_readers, num_writers, num_keywd, num_items, num_magazine, latent_dim, dev_tensor):
        super(CNN, self).__init__()

        self.num_readers = num_readers
        self.num_writers = num_writers
        self.num_keywd = num_keywd
        self.num_items = num_items
        self.num_magazine = num_magazine
        self.latent_dim = latent_dim
        self.dev_tensor = dev_tensor.long()
        self.num_layer = 2
        self.keywd_seq_length = 5

        self.embedding_reader = nn.Embedding(num_embeddings=self.num_readers, embedding_dim=self.latent_dim)
        self.embedding_writer = nn.Embedding(num_embeddings=self.num_writers, embedding_dim=self.latent_dim)
        self.embedding_keywd = nn.Embedding(num_embeddings=self.num_keywd, embedding_dim=self.latent_dim, padding_idx=1)
        self.embedding_magazine = nn.Embedding(num_embeddings=self.num_magazine, embedding_dim=self.latent_dim)

        self.ts_min_embedding = nn.Parameter(torch.Tensor(1,self.latent_dim), requires_grad=True)
        self.ts_max_embedding = nn.Parameter(torch.Tensor(1,self.latent_dim), requires_grad=True)
        nn.init.xavier_normal_(self.ts_min_embedding.data)
        nn.init.xavier_normal_(self.ts_max_embedding.data)

        self.encoder_cnn = nn.Sequential(
                                nn.Conv1d(4*self.latent_dim, 2*self.latent_dim, 2),
                                nn.ReLU(),
                                nn.Conv1d(2*self.latent_dim, self.latent_dim, 3),
                                nn.ReLU(),)
        self.encoder_fc = nn.Linear(3*self.latent_dim, self.latent_dim)
        self.decoder_fc = nn.Linear(4*self.latent_dim, self.latent_dim)
        #self.writer_idx_fc = nn.Linear(in_features=2*self.latent_dim, out_features=self.num_writers)
        self.keywd_idx_fc = nn.Linear(in_features=self.latent_dim, out_features=self.num_keywd)

        self.predict_bias = nn.Parameter(torch.Tensor(1,self.num_items), requires_grad=True)
        nn.init.xavier_normal_(self.predict_bias)

        nn.init.xavier_uniform_(self.encoder_fc.weight)
        nn.init.xavier_uniform_(self.decoder_fc.weight)
        #nn.init.xavier_uniform_(self.writer_idx_fc.weight)
        nn.init.xavier_uniform_(self.keywd_idx_fc.weight)

    def interpolate(self, input):
        return self.ts_max_embedding*input + self.ts_min_embedding*(1-input)

    def forward(self, data, mode='Train'):
        # batch_size x length x reader readat item_id writer keywd*5 reg_ts magazine_id
        ## reader + readat
        reader_embedding = self.embedding_reader(data[:,0,0])
        if mode == 'Train':
            readat_embedding_0 = self.interpolate(data[:,0,1].float().unsqueeze(-1))
            #readat_embedding_1 = self.interpolate(data[:,0,2].float().unsqueeze(-1))
            #readat_embedding = torch.stack([readat_embedding_0, readat_embedding_1], 1)
            readat_embedding = readat_embedding_0.unsqueeze(1)
        elif mode == 'Valid':
            readat_embedding = self.interpolate(data[:,0,1].float().unsqueeze(-1)).unsqueeze(1)
        elif mode == 'Test':
            readat_embedding_0 = self.interpolate(torch.tensor([[0.915]]*data.size(0)).cuda())
            readat_embedding_1 = self.interpolate(torch.tensor([[0.951]]*data.size(0)).cuda())
            readat_embedding_2 = self.interpolate(torch.tensor([[0.99]]*data.size(0)).cuda())
            #readat_embedding = torch.stack([readat_embedding_0, readat_embedding_1, readat_embedding_2], 1)
            readat_embedding = readat_embedding_0.unsqueeze(1)

        ## item elements
        writer_embedding = self.embedding_writer(data[:,:,3])
        keywd_embedding = torch.mean(self.embedding_keywd(data[:,:,4:9]), 2)
        reg_embedding = self.interpolate(data[:,:,9].float().view(-1,1)).view(-1,writer_embedding.size(1),self.latent_dim)
        magazine_embedding = self.embedding_magazine(data[:,:,10])

        ## encode + decode
        merged_item_embedding = self.encoder_cnn(torch.cat([reg_embedding, writer_embedding, #item_embedding,
                                                           keywd_embedding, magazine_embedding], 2).permute(0,2,1))
        print(merged_item_embedding.permute(0,2,1).size(), reader_embedding.size(), readat_embedding.size())
        output = self.encoder_fc(torch.cat([merged_item_embedding.permute(0,2,1),
                                            reader_embedding, readat_embedding], 2))

        ## dev embedding
        dev_writer_emb = self.embedding_writer(self.dev_tensor[:,1])
        dev_keywd_emb = torch.mean(self.embedding_keywd(self.dev_tensor[:,2:7]), 1)
        dev_reg_emb = self.interpolate(self.dev_tensor[:,7].float().unsqueeze(-1))
        dev_magazine_emb = self.embedding_magazine(self.dev_tensor[:,8])
        merged_dev_embedding = self.decoder_fc(torch.cat([dev_reg_emb, dev_writer_emb,
                                                          dev_keywd_emb, dev_magazine_emb], 1))

        ## predict
        item_output = torch.einsum('bld,dv->blv', output, merged_dev_embedding.t())
        item_output += self.predict_bias.expand_as(item_output)
        #writer_output = self.writer_idx_fc(output)
        #keywd_output = self.keywd_idx_fc(output)

        return item_output, None#keywd_output#, None, None#
