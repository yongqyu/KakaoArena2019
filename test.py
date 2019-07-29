import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from models import RNN
from utils import hsort
from config import get_args
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.mode == 'Dev':
    dev_users_path = args.dataset_root+'predict/dev.users'
    dev_mask = torch.from_numpy(np.load(args.prepro_root+'dev_mask.npy')).float().cuda()
    dev_dataset = torch.load(args.prepro_root+args.dev_dataset_path)
elif args.mode == 'Test':
    dev_users_path = args.dataset_root+'predict/test.users'
    dev_mask = torch.from_numpy(np.load(args.prepro_root+'test_mask.npy')).float().cuda()
    dev_dataset = torch.load(args.prepro_root+args.test_dataset_path)
item_list = np.load(args.prepro_root+'item_list.npy')
valid_tensor = torch.load(args.prepro_root+'valid_writer_keywd.pkl').to(device)

model = RNN(args.num_readers, args.num_writers, args.num_keywords,
            args.num_items, args.num_magazines, args.hid_dim, valid_tensor).to(device)
print(model)
model.eval()
model.load_state_dict(torch.load('./models/9_rnn_attention.pkl'))

file_w = open('./recommend.txt', 'w')
file = open(dev_users_path, 'r')
readers = file.read().splitlines()
dev_loader = data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
# [item_id, writer] + keywd + [reg_ts, meg_id]
with torch.no_grad():
    for i, input in enumerate(tqdm(dev_loader)):
        input = input[0].to(device)
        items = input[:,18:].contiguous().view(-1,5,9)
        preds = model(input[:,:18], items, mode=args.mode)
        preds = torch.mul(preds[:,0], dev_mask[i*args.batch_size:(i+1)*args.batch_size])

        sorted_dot_product = hsort(preds)
        for reader, pred in zip(readers[i*args.batch_size:],sorted_dot_product):
            file_w.write(reader+' '+' '.join(list(map(lambda x: item_list[x], pred)))+'\n')

file_w.close()
file.close()
