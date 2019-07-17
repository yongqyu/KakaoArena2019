import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--val_step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)

    parser.add_argument('--num_keywords', type=int, default=96894)
    parser.add_argument('--num_readers', type=int, default=310759)
    parser.add_argument('--num_writers', type=int, default=19066)
    parser.add_argument('--num_items', type=int, default=643105)
    parser.add_argument('--num_magazines', type=int, default=28130)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--train_dataset_path', type=str,
                        default='/data/private/Arena/prepro_results/train_dataset.pkl')
    parser.add_argument('--valid_dataset_path', type=str,
                        default='/data/private/Arena/prepro_results/valid_dataset.pkl')
    parser.add_argument('--test_dataset_path', type=str,
                        default='/data/private/Arena/prepro_results/test_dataset.pkl')
    parser.add_argument('--save_path', type=str,
                        default='./models/')

    args = parser.parse_args()
    print(args)

    return args


    '''
    self.keywd_rnn = nn.GRU(self.latent_dim, self.latent_dim, self.num_layer, batch_first=True)
    self.items_rnn = nn.GRU(2*self.latent_dim, self.latent_dim, self.num_layer, batch_first=True)
    self.items_fc = nn.Linear(4*self.latent_dim+3, 2*self.latent_dim)
    self.fc = nn.Linear(in_features=self.latent_dim, out_features=self.num_items)

    nn.init.xavier_uniform_(self.fc.weight)
    for name, param in list(self.keywd_rnn.named_parameters()) + \
                       list(self.items_rnn.named_parameters()):
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)


    def forward(self, data):
        # batch_size x length x reader item_id writer keywd*5 reg_ts*3 magazine_id
        reader_embedding = self.embedding_reader(data[:,0,0].long())
        item_embedding = self.embedding_items(data[:,:,1].long())
        writer_embedding = self.embedding_writer(data[:,:,2].long())
        keywd_embedding = self.embedding_keywd(data[:,:,3:8].long())
        magazine_embedding = self.embedding_magazine(data[:,:,11].long())
        reg_ts_gap = data[:,:,8:11].float()

        hidden, _ = self.keywd_rnn(keywd_embedding.view(-1,self.seq_length,self.latent_dim)) # reader_embedding
        keywd_embedding = hidden[:,-1,:].view(data.size(0), data.size(1), self.latent_dim)
        merged_item_embedding = self.items_fc(torch.cat([item_embedding, writer_embedding,
                                                        keywd_embedding, magazine_embedding, reg_ts_gap], 2))
        reader_embedding = reader_embedding.unsqueeze(0).expand(self.num_layer, -1, -1).contiguous()
        hidden, _ = self.items_rnn(merged_item_embedding, reader_embedding)
        output = self.fc(hidden[:,-1])

        return output
    '''
