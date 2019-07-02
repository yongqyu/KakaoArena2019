import numpy as np
import torch
from tqdm import tqdm

class GMF(torch.nn.Module):
    def __init__(self, num_users, num_writers, num_keywd, latent_dim, writerid2items, userid2followid):
        super(GMF, self).__init__()
        self.num_users = num_users
        self.num_writers = num_writers
        self.num_keywd = num_keywd
        self.latent_dim = latent_dim
        self.writerid2items = writerid2items
        self.userid2followid = userid2followid
        #self.userid2keyword = userid2keyword

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=2 * self.latent_dim)
        self.embedding_writer = torch.nn.Embedding(num_embeddings=self.num_writers, embedding_dim=self.latent_dim)
        self.embedding_keywd = torch.nn.Embedding(num_embeddings=self.num_keywd, embedding_dim=self.latent_dim)

        self.item4valid = None

        self.logistic = torch.nn.Sigmoid()

    def forward(self, data):#user_indices, item_indices, item_keywd, negs_indices, negs_keywd):
        user_embedding = self.embedding_user(data[0])
        #self.userid2followid[u] for u in data[:,0]
        print(data[2])
        writer_embedding = self.embedding_writer(data[2])
        negs_embedding = self.embedding_writer(data[4])
        item_kw_embedding = self.embedding_keywd(data[1])
        negs_kw_embedding = self.embedding_keywd(data[3])

        #element_product = torch.mul(user_embedding, torch.cat([writer_embedding, item_kw_embedding], 1))
        #pos_logits = self.affine_output(element_product)
        pos_logits = torch.mm(torch.cat([writer_embedding, item_kw_embedding], 1), user_embedding[0].unsqueeze(-1))

        #element_product = torch.mul(user_embedding, torch.cat([negs_embedding, negs_kw_embedding], 1))
        #neg_logits = self.affine_output(element_product)
        neg_logits = torch.mm(torch.cat([negs_embedding, negs_kw_embedding], 1), user_embedding[0].unsqueeze(-1))
        loss = self.logistic(pos_logits) - self.logistic(neg_logits)

        return torch.sum(loss)

    def predict(self, users, top=100):
        user_embedding = self.embedding_user(users)
        total_writer_embedding = self.embedding_writer.weight#(valid_tensor[:,0])

        #element_product = torch.mm(user_embedding, torch.cat([writer_embedding, keywd_embedding], 1))
        #logits = self.affine_output(element_product)
        writer_logits = torch.mm(user_embedding[:,:self.latent_dim], total_writer_embedding.t())
        _, writer_indices = torch.sort(writer_logits, descending=True)
        writer_indices = list(dict.fromkeys(writer_indices[:,:10].flatten().tolist()))
        cutted_valid_tensor = [self.writerid2items[writer_index] for writer_index in writer_indices]
        cutted_valid_tensor = torch.from_numpy(np.concatenate(cutted_valid_tensor, axis=0)).cuda()
        writer_embedding = self.embedding_writer(cutted_valid_tensor[:,1])
        keywd_embedding = self.embedding_keywd(cutted_valid_tensor[:,2])
        logits = torch.mm(user_embedding, torch.cat([writer_embedding, keywd_embedding], 1).t())
        _, sorted_indices = torch.sort(logits, 1, descending=True)
        sorted_indices = sorted_indices.squeeze()[:,:100]#.cpu().numpy().tolist()
        return cutted_valid_tensor[sorted_indices,0].cpu().numpy()

    def set_item4valid(self, item4valid):
        self.item4valid = item4valid

    def init_weight(self):
        pass
