import os
import random
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

reader2id = np.load('/data/private/Arena/prepro_results/reader2id.npy', allow_pickle=True).item()
item_dict = np.load('/data/private/Arena/prepro_results/item_dict.npy', allow_pickle=True).item()

read_path = '/data/private/Arena/datasets/read/'
read_files = os.listdir(read_path)
train_read_files = read_files[:int(len(read_files)*0.8)]
valid_read_files = read_files[int(len(read_files)*0.8):]

train_data = []
for read_file in tqdm.tqdm(train_read_files, desc='Train'):
    file = open(read_path+read_file, 'r')
    data_ = file.readlines()
    for line in data_:
        tokens = line.split(' ')
        for x in tokens[1:-1]: train_data.append([reader2id[tokens[0]] if reader2id.get(tokens[0]) != None else reader2id['unk'],
                                                  x])
                                                  #item_dict[x] if item_dict.get(x) != None else item_dict['unk']])
np.save('/data/private/Arena/prepro_results/train_data.npy', train_data)

valid_data = []
for read_file in tqdm.tqdm(valid_read_files, desc='Valid'):
    file = open(read_path+read_file, 'r')
    try:
        data_ = file.readlines()
    except:
        print(read_file)
        continue
    for line in data_:
        tokens = line.split(' ')
        for x in tokens[1:-1]: valid_data.append([reader2id[tokens[0]] if reader2id.get(tokens[0]) != None else reader2id['unk'],
                                                  x])
                                                  #item_dict[x] if item_dict.get(x) != None else item_dict['unk']])

np.save('/data/private/Arena/prepro_results/valid_data.npy', valid_data)
