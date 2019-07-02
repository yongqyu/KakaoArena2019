import random
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import ml_metrics as metrics

from data_loader import get_loader
from models import GMF
from metrics import evaluate

num_epochs = 10
learning_rate = 0.0001
hidden_dim = 128
val_step = 1
batch_size = 2048
num_keywords = 96892; num_readers = 310759; num_writers = 19066

#valid_tensor = torch.load('/data/private/Arena/prepro_results/valid_writer_keywd.pkl').cuda()
writerid2items = np.load('/data/private/Arena/prepro_results/writerid2items.npy', allow_pickle=True).item()
userid2followid = np.load('/data/private/Arena/prepro_results/userid2followid.npy', allow_pickle=True).item()
train_path = '/data/private/Arena/prepro_results/train_data.npy'
valid_path = '/data/private/Arena/prepro_results/valid_data.npy'
train_loader = get_loader(train_path, batch_size=batch_size, mode='train')
valid_loader = get_loader(valid_path, batch_size=batch_size, mode='valid')

model = GMF(num_readers, num_writers, num_keywords, hidden_dim, writerid2items, userid2followid)#.cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(model)
print('# of params : ', params)

best_map = 0
for epoch in range(num_epochs):
    model.train()
    for data in tqdm.tqdm(train_loader, desc='Train'):
        loss = model(data[0])#.cuda())

        model.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%val_step == 0:
        with torch.no_grad():
            model.eval()
            valid_map = 0
            for i, data in enumerate(tqdm.tqdm(valid_loader, desc='Valid')):
                actual = data[0][:,3].cpu().numpy().tolist()
                preds = model.predict(data[0][:,0]).tolist()
                print('actual', actual)
                print('preds', preds)
                #ap = [metrics.apk(a, p, 100) for (a,p) in zip(actual, preds)]
        '''
                ap = [1/(p.index(a)+1) if a in p else 0 for (a,p) in zip(actual, preds)]
                valid_map += np.mean(ap)

        print('epoch: '+str(epoch+1)+' MAP: '+str(valid_map/(i+1)))
        if best_map < valid_map:
            best_map = valid_map
            best_epoch = epoch+1
            torch.save(model.state_dict(), './models/%d_gmf.pkl' % (epoch+1))
        '''
