import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class GMF(nn.Module):
    def __init__(self, num_readers, num_writers, num_keywd, num_items, latent_dim, valid_tensor, readerid2items):
        super(GMF, self).__init__()
        self.num_readers = num_readers
        self.num_writers = num_writers
        self.num_keywd = num_keywd
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.valid_tensor = valid_tensor
        self.readerid2items = readerid2items

        self.embedding_reader = nn.Embedding(num_embeddings=self.num_readers, embedding_dim=self.latent_dim)
        self.embedding_items = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_writer = nn.Embedding(num_embeddings=self.num_writers, embedding_dim=self.latent_dim)
        self.embedding_keywd = nn.Embedding(num_embeddings=self.num_keywd, embedding_dim=self.latent_dim, padding_idx=1)

        self.keywd_rnn = nn.GRU(self.latent_dim, self.latent_dim, 1, batch_first=True)
        self.item_fc = torch.nn.Linear(in_features=3*self.latent_dim, out_features=self.latent_dim)

        nn.init.xavier_uniform_(self.item_fc.weight)
        self.logistic = nn.Sigmoid()

    def forward(self, data):#reader_indices, item_indices, item_keywd, negs_indices, negs_keywd):
        # reader, subs, negs, subs_elem, negs_elem
        reader_embedding = self.embedding_reader(data[:,0])
        item_embedding = self.embedding_items(data[:,1])
        #writer_embedding = self.embedding_writer(data[:,3])
        #keyword_embedding = self.embedding_keywd(data[:,4:9])
        negs_embedding = self.embedding_items(data[:,2])
        #negs_writer_embedding = self.embedding_writer(data[:,11])
        #negs_kw_embedding = self.embedding_keywd(data[:,12:17])

        #keyword_output, _ = self.keywd_rnn(keyword_embedding)
        #item_joint_embedding = self.item_fc(torch.cat([item_embedding, writer_embedding, keyword_output[:,-1,:]], 1))
        pos_logits = torch.mm(reader_embedding, item_embedding.t())

        #negs_kw_output, _ = self.keywd_rnn(negs_kw_embedding)
        #negs_joint_embedding = self.item_fc(torch.cat([negs_embedding, negs_writer_embedding, negs_kw_output[:,-1,:]], 1))
        neg_logits = torch.mm(reader_embedding, negs_embedding.t())
        loss = - self.logistic(pos_logits) + self.logistic(neg_logits)

        return torch.sum(loss)

    def predict(self, reader, top=100):
        # [item_id, writer] + keywd + [reg_ts, meg_id]
        reader_embedding = self.embedding_reader(reader)
        item_embedding = self.embedding_items(self.valid_tensor[:,0])
        #writer_embedding = self.embedding_writer(self.valid_tensor[:,1])
        #keywd_embedding = self.embedding_keywd(self.valid_tensor[:,2:7])

        #keywd_output, _ = self.keywd_rnn(keywd_embedding)
        #item_joint_embedding = self.item_fc(torch.cat([item_embedding, writer_embedding, keywd_output[:,-1,:]], 1))
        dot_product = torch.mm(reader_embedding, item_embedding.t())
        for i, r in enumerate(reader):
            if self.readerid2items.get(r.item()) != None:
                #read_items = list(set([a for b in self.readerid2items[r.item()] for a in b]))
                #dot_product[i][read_items] = 0
                dot_product[i][self.readerid2items[r.item()]] = 0
        split_size = 1024
        hidden_dot_product = [torch.sort(dot_product[:,split_size*k:split_size*(k+1)], descending=True)[1][:,:top] + k*split_size
                              for k in range((dot_product.size(1)//split_size)+1)]
        hidden_dot_product = torch.cat(hidden_dot_product, 1)
        sorted_dot_product = torch.sort(dot_product.gather(1, hidden_dot_product), descending=True)[1][:,:top]
        sorted_dot_product = hidden_dot_product.gather(1, sorted_dot_product)
        return sorted_dot_product.cpu().numpy()

        '''
        writer_logits = torch.mm(reader_embedding[:,:self.latent_dim], total_writer_embedding.t())
        _, writer_indices = torch.sort(writer_logits, descending=True)
        writer_indices = list(dict.fromkeys(writer_indices[:,:10].flatten().tolist()))
        cutted_valid_tensor = [self.writerid2items[writer_index] for writer_index in writer_indices]
        cutted_valid_tensor = torch.from_numpy(np.concatenate(cutted_valid_tensor, axis=0)).cuda()

        writer_embedding = self.embedding_writer(cutted_valid_tensor[:,1])
        keywd_embedding = self.embedding_keywd(cutted_valid_tensor[:,2])
        logits = torch.mm(reader_embedding, torch.cat([writer_embedding, keywd_embedding], 1).t())
        _, sorted_indices = torch.sort(logits, 1, descending=True)
        sorted_indices = sorted_indices.squeeze()[:,:100]#.cpu().numpy().tolist()
        return cutted_valid_tensor[sorted_indices,0].cpu().numpy()
        '''

    def init_weight(self):
        pass


class RNN(nn.Module):
    def __init__(self, num_readers, num_writers, num_keywd, num_items, latent_dim, valid_tensor, readerid2items):
        super(RNN, self).__init__()

        self.num_readers = num_readers
        self.num_writers = num_writers
        self.num_keywd = num_keywd
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.num_layer = 2

        self.embedding_reader = nn.Embedding(num_embeddings=self.num_readers, embedding_dim=self.latent_dim)
        self.embedding_items = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        #self.embedding_writer = nn.Embedding(num_embeddings=self.num_writers, embedding_dim=self.latent_dim)
        #self.embedding_keywd = nn.Embedding(num_embeddings=self.num_keywd, embedding_dim=self.latent_dim, padding_idx=1)

        self.rnn = nn.GRU(self.latent_dim, self.latent_dim, self.num_layer, batch_first=True)
        self.fc = nn.Linear(in_features=self.latent_dim, out_features=self.num_items)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, data):
        reader_embedding = self.embedding_reader(data[:,0])
        item_embedding = self.embedding_items(data[:,1:])

        reader_embedding = reader_embedding.unsqueeze(0).expand(self.num_layer, -1, -1).contiguous()
        hidden, _ = self.rnn(item_embedding, reader_embedding)
        output = self.fc(hidden[:,-1,:])

        return output
