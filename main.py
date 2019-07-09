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

train_dataset = torch.load(args.train_dataset_path)
valid_dataset = torch.load(args.valid_dataset_path)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

#model = GMF(num_readers, num_writers, num_keywords, num_items,
#            hidden_dim, valid_tensor, readerid2items).cuda()
model = RNN(args.num_readers, args.num_writers, args.num_keywords,
            args.num_items, args.num_magazines, args.hid_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')#none')
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(model)
print('# of params : ', params)

best_loss = 9999
for epoch in range(args.num_epochs):
    model.train()
    for data in tqdm.tqdm(train_loader, desc='Train'):
        logits = model(data[0][:,:-1].to(device))
        #target = torch.zeros([data[0].size(0), args.num_items]).to(device)
        #for x,y in enumerate(data[0][:,-2:,1]): target[x,y] = 1
        #loss = criterion(logits, target)
        loss = criterion(logits, data[0][:,-1,1].to(device))
        #loss = ap(logits, data[0][:,-2:,1].cuda())

        model.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%args.val_step == 0:
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            for i, data in enumerate(tqdm.tqdm(valid_loader, desc='Valid')):
                preds = model(data[0][:,:-1].cuda())
                loss = criterion(preds, data[0][:,-1,1].cuda())
                valid_loss += torch.mean(loss).cpu().item()

        print('epoch: '+str(epoch+1)+' Loss: '+str(valid_loss/(i+1)))
        if best_loss > valid_loss:
            best_loss = valid_loss
            best_epoch = epoch+1
            torch.save(model.state_dict(), args.save_path % (epoch+1))
