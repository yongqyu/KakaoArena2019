import random
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import ml_metrics as metrics

from models import RNN
from metrics import evaluate
from utils import ap
from config import get_args
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

train_dataset = torch.load(args.train_dataset_path)
valid_dataset = torch.load(args.valid_dataset_path)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

model = RNN(args.num_readers, args.num_writers, args.num_keywords,
            args.num_items, args.num_magazines, args.hid_dim).to(device)
#encoder_layer = TransformerEncoderLayer(args.hid_dim, args.num_heads)
#model = TransformerEncoder(encoder_layer, 4,#num_layers,
#                           args.num_readers, args.num_writers, args.num_keywords,
#                           args.num_items, args.num_magazines, args.hid_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
b_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')#none')
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(model)
print('# of params : ', params)

if args.start_epoch:
    model.load_state_dict(torch.load(args.save_path+'%d_transformer.pkl' % args.start_epoch))

best_loss = 9999999
for epoch in range(args.num_epochs):
    model.train()
    for data in tqdm.tqdm(train_loader, desc='Train'):
        item_logits, writer_logits, keywd_logits = model(data[0][:,:-1].to(device))
        keywd_target = torch.zeros([data[0].size(0), args.num_keywords])
        keywd_target = keywd_target.scatter_(1,data[0][:,-1,4:9].long(),1).to(device)
        #loss = criterion(logits, target)
        loss = criterion(item_logits, data[0][:,-1,2].to(device).long()) + \
               criterion(writer_logits, data[0][:,-1,3].to(device).long()) + \
               b_criterion(keywd_logits, keywd_target)
        #loss = ap(logits, data[0][:,-2:,1].cuda())

        model.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%args.val_step == 0:
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            for i, data in enumerate(tqdm.tqdm(valid_loader, desc='Valid')):
                item_preds, writer_logits, keywd_logits = model(data[0][:,:-1].cuda())
                loss = criterion(item_preds, data[0][:,-1,2].cuda().long())
                valid_loss += torch.mean(loss).cpu().item()

        print('epoch: '+str(epoch+1)+' Loss: '+str(valid_loss/(i+1)))
        if best_loss > valid_loss/(i+1):
            best_loss = valid_loss
            best_epoch = epoch+1
            torch.save(model.state_dict(), args.save_path+'%d_rnn_residual.pkl' % (epoch+1))
    scheduler.step()
