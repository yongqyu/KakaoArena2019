import os
import random
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data

from models import RNN
from utils import ap
from config import get_args
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.prepro_root):
    os.makedirs(args.prepro_root)
train_dataset = torch.load(args.prepro_root + args.train_dataset_path)
valid_dataset = torch.load(args.prepro_root + args.valid_dataset_path)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
valid_tensor = torch.load(args.prepro_root+'valid_writer_keywd.pkl').to(device)

model = RNN(args.num_readers, args.num_writers, args.num_keywords,
            args.num_items, args.num_magazines, args.hid_dim, valid_tensor).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(model)
print('# of params : ', params)

if args.start_epoch:
    model.load_state_dict(torch.load(args.save_path+'%d_rnn_attention.pkl' % args.start_epoch))

best_loss = 9999999
for epoch in range(args.num_epochs):
    model.train()
    for i, data in enumerate(tqdm.tqdm(train_loader, desc='Train')):
        # reader readerat reader_f*8 reader_k*8 (item writer keywd*5 reg_ts maga)*N
        data = data[0].to(device)
        items = data[:,18:].contiguous().view(-1,5,9)
        item_logits = model(data[:,:18], items[:,:-1], mode=args.mode)
        loss = criterion(item_logits[:,0], items[:,-1,0].long())

        model.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%args.val_step == 0:
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            for i, data in enumerate(tqdm.tqdm(valid_loader, desc='Valid')):
                data = data[0].to(device)
                items = data[:,18:].contiguous().view(-1,5,9)
                item_preds = model(data[:,:18], items[:,:-1], mode=args.mode)
                loss = criterion(item_preds[:,0], items[:,-1,0].long()).cpu().item()
                valid_loss += loss

        print('epoch: '+str(epoch+1)+' Loss: '+str(valid_loss/(i+1)))
        if best_loss > valid_loss/(i+1):
            best_loss = valid_loss/(i+1)
            best_epoch = epoch+1
            torch.save(model.state_dict(), args.save_path+'%d_rnn_attention.pkl' % (epoch+1))
    scheduler.step()
