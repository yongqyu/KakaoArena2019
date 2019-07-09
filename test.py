import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from models import RNN
from utils import hsort
from config import get_args
args = get_args()

dev_users_path = '/data/private/Arena/datasets/predict/dev.users'
test_users_path = '/data/private/Arena/datasets/predict/test.users'
dev_mask = torch.from_numpy(np.load('/data/private/Arena/prepro_results/dev_mask.npy')).float().cuda()
item_list = np.load('/data/private/Arena/prepro_results/item_list.npy')

model = RNN(args.num_readers, args.num_writers, args.num_keywords,
            args.num_items, args.num_magazines, args.hid_dim).cuda()
model.eval()
model.load_state_dict(torch.load('./models/7_rnn_keywd.pkl'))
print(model)

file_w = open('./recommend.txt', 'w')
file = open(dev_users_path, 'r')
readers = file.read().splitlines()
dev_dataset = torch.load(args.test_dataset_path)
dev_loader = data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
# [item_id, writer] + keywd + [reg_ts, meg_id]
with torch.no_grad():
    for i, input in enumerate(tqdm(dev_loader)):
        preds = model(input[0].cuda())
        preds = torch.mul(preds, dev_mask[i*args.batch_size:(i+1)*args.batch_size])

        sorted_dot_product = hsort(preds)
        for reader, pred in zip(readers[i*args.batch_size:],sorted_dot_product):
            file_w.write(reader+' '+' '.join(list(map(lambda x: item_list[x], pred)))+'\n')

file_w.close()
file.close()
