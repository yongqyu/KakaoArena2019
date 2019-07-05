import random
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import ml_metrics as metrics

from models import GMF
from metrics import evaluate

num_epochs = 12
learning_rate = 0.001
hidden_dim = 128
val_step = 2
batch_size = 2048
num_keywords = 96894; num_readers = 310759; num_writers = 19066; num_items = 643105

train_dataset = torch.load('/data/private/Arena/prepro_results/train_dataset.pkl')
valid_dataset = torch.load('/data/private/Arena/prepro_results/valid_dataset.pkl')
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
valid_tensor = torch.load('/data/private/Arena/prepro_results/valid_writer_keywd.pkl').cuda()
readerid2items = np.load('/data/private/Arena/prepro_results/reader2item.npy', allow_pickle=True).item()

model = GMF(num_readers, num_writers, num_keywords, num_items,
            hidden_dim, valid_tensor, readerid2items).cuda()
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
            loss = model(data[0].cuda())

            model.zero_grad()
            loss.backward()
            optimizer.step()

    if (epoch+1)%val_step == 0:
        with torch.no_grad():
            model.eval()
            valid_map = 0
            for i, data in enumerate(tqdm.tqdm(valid_loader, desc='Valid')):
                actual = data[0][:,3].numpy().tolist()
                preds = model.predict(data[0][:,0].cuda()).tolist()
                #ap = [metrics.apk(a, p, 100) for (a,p) in zip(actual, preds)]
                ap = [1/(p.index(a)+1) if a in p else 0 for (a,p) in zip(actual, preds)]
                valid_map += np.mean(ap)

        print('epoch: '+str(epoch+1)+' MAP: '+str(valid_map/(i+1)))
        if best_map < valid_map:
            best_map = valid_map
            best_epoch = epoch+1
            torch.save(model.state_dict(), './models/%d_joint_gmf.pkl' % (epoch+1))
