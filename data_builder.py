import os
import random
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

item_dict = np.load('/data/private/Arena/prepro_results/item_dict.npy', allow_pickle=True).item()
item_list = np.load('/data/private/Arena/prepro_results/item_list.npy')
item2elem = np.load('/data/private/Arena/prepro_results/item2elem.npy', allow_pickle=True).item()
keyword_dict = np.load('/data/private/Arena/prepro_results/keyword_dict.npy', allow_pickle=True).item()
keyword_list = np.load('/data/private/Arena/prepro_results/keyword_list.npy').tolist()
id2reader = np.load('/data/private/Arena/prepro_results/id2reader.npy')
reader2id = np.load('/data/private/Arena/prepro_results/reader2id.npy', allow_pickle=True).item()
reader2elem = np.load('/data/private/Arena/prepro_results/reader2elem.npy', allow_pickle=True).item()
id2writer = np.load('/data/private/Arena/prepro_results/id2writer.npy')
writer2id = np.load('/data/private/Arena/prepro_results/writer2id.npy', allow_pickle=True).item()

max_keylen = 5
num_keywords = len(keyword_dict)
num_readers = len(id2reader)
num_writers = len(id2writer)
num_items = len(item_list)
print(num_keywords, num_readers, num_writers, num_items)

read_path = '/data/private/Arena/datasets/read/'
read_files = os.listdir(read_path)
train_read_files = read_files[:int(len(read_files)*0.8)]
valid_read_files = read_files[int(len(read_files)*0.8):]

train_data = []
for read_file in tqdm.tqdm(train_read_files, desc='Train'):
    file = open(read_path+read_file, 'r')
    data_ = file.readlines()
    for line in data_:
        line = line.split(' ')
        subs = [item_dict[item] for item in line[1:-1] if item_dict.get(item) != None]
        if not subs:
            continue
        subs_elem = np.array([item2elem.get(sub) for sub in subs])
        #negs = random.sample(list(set(range(num_items))-set(subs)), len(subs))
        negs = [sum(x)%(num_items-1)+1 for x in zip(subs, np.random.choice(range(1024,2048), len(subs)))]
        negs_elem = np.array([item2elem.get(neg) for neg in negs])
        reader = [reader2id[line[0]] if reader2id.get(line[0]) else reader2id['unk']]*len(subs)

        row = np.concatenate((np.stack((reader, subs, negs), -1), subs_elem, negs_elem), 1)
        train_data.append(torch.from_numpy(row))

train_data = torch.cat(train_data, 0)
train_dataset = data.TensorDataset(train_data)
torch.save(train_dataset, '/data/private/Arena/prepro_results/train_dataset.pkl')
train_data_loader = data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

valid_data = []
for read_file in tqdm.tqdm(valid_read_files, desc='Valid'):
    file = open(read_path+read_file, 'r')
    try:
        data_ = file.readlines()
    except:
        print(read_file)
    for line in data_:
        line = line.split(' ')
        subs = [item_dict[item] for item in line[1:-1] if item_dict.get(item) != None]
        if not subs:
            continue
        subs_elem = np.array([item2elem.get(sub) for sub in subs])
        reader = [reader2id[line[0]] if reader2id.get(line[0]) else reader2id['unk']]*len(subs)

        row = np.concatenate((np.stack((reader, subs), -1), subs_elem), 1)
        valid_data.append(row)

valid_data = np.concatenate(valid_data, 0)
valid_dataset = data.TensorDataset(torch.from_numpy(valid_data))
torch.save(valid_dataset, '/data/private/Arena/prepro_results/valid_dataset.pkl')
valid_data_loader = data.DataLoader(valid_dataset, batch_size=1024, shuffle=False)
