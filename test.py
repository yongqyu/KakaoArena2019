import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from models import RNN, CNN
from utils import hsort
from config import get_args
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dev_users_path = '/data/private/Arena/datasets/predict/dev.users'
test_users_path = '/data/private/Arena/datasets/predict/test.users'
dev_mask = torch.from_numpy(np.load('/data/private/Arena/prepro_results/dev_mask.npy')).float().cuda()
item_list = np.load('/data/private/Arena/prepro_results/item_list.npy')
valid_tensor = torch.load('/data/private/Arena/prepro_results/valid_writer_keywd.pkl').to(device)

model = CNN(args.num_readers, args.num_writers, args.num_keywords,
            args.num_items, args.num_magazines, args.hid_dim, valid_tensor).to(device)
#encoder_layer = TransformerEncoderLayer(args.hid_dim, args.num_heads)
#model = TransformerEncoder(encoder_layer, 4,#num_layers,
#                           args.num_readers, args.num_writers, args.num_keywords,
#                           args.num_items, args.num_magazines, args.hid_dim).to(device)
print(model)
model.eval()
model.load_state_dict(torch.load('./models/11_rnn_residual.pkl'))

file_w = open('./recommend.txt', 'w')
file = open(dev_users_path, 'r')
readers = file.read().splitlines()
dev_dataset = torch.load(args.test_dataset_path)
dev_loader = data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
# [item_id, writer] + keywd + [reg_ts, meg_id]
with torch.no_grad():
    for i, input in enumerate(tqdm(dev_loader)):
        input = input[0].long().to(device)
        preds,_ = model(input, mode='Test')
        preds = torch.mul(preds[:,0], dev_mask[i*args.batch_size:(i+1)*args.batch_size])

        sorted_dot_product = hsort(preds)
        for reader, pred in zip(readers[i*args.batch_size:],sorted_dot_product):
            file_w.write(reader+' '+' '.join(list(map(lambda x: item_list[x], pred)))+'\n')

file_w.close()
file.close()
