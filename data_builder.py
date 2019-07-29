import os
import random
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

item_dict = np.load(args.prepro_root+'item_dict.npy', allow_pickle=True).item()
item_list = np.load(args.prepro_root+'item_list.npy')
item2elem = np.load(args.prepro_root+'item2elem.npy', allow_pickle=True).item()
keyword_dict = np.load(args.prepro_root+'keyword_dict.npy', allow_pickle=True).item()
keyword_list = np.load(args.prepro_root+'keyword_list.npy').tolist()
id2reader = np.load(args.prepro_root+'id2reader.npy')
reader2id = np.load(args.prepro_root+'reader2id.npy', allow_pickle=True).item()
reader2elem = np.load(args.prepro_root+'reader2elem.npy', allow_pickle=True).item()
id2writer = np.load(args.prepro_root+'id2writer.npy')
writer2id = np.load(args.prepro_root+'writer2id.npy', allow_pickle=True).item()
id2magazine = np.load(args.prepro_root+'id2magazine.npy')
magazine2id = np.load(args.prepro_root+'magazine2id.npy', allow_pickle=True).item()

max_keylen = 5
num_keywords = len(keyword_dict)
num_readers = len(id2reader)
num_writers = len(id2writer)
num_items = len(item_list)
num_magazine = len(id2magazine)
print(num_keywords, num_readers, num_writers, num_items, num_magazine)

rnn_train_data = np.load(args.prepro_root+'rnn_train_data.npy')
rnn_valid_data = np.load(args.prepro_root+'rnn_valid_data.npy')
rnn_dev_data = np.load(args.prepro_root+'rnn_dev_data.npy')
rnn_test_data = np.load(args.prepro_root+'rnn_test_data.npy')
rnn_train_dataset = []
for data_ in rnn_train_data:
    reader = data_[0]
    readat1 = data_[1]
    readers = reader2elem[data_[0]]
    readed = [[item]+item2elem[item] for item in data_[2:]]
    readed = [a for b in readed for a in b]
    rnn_train_dataset.append([reader, readat1] + readers + readed)
rnn_valid_dataset = []
for data_ in rnn_valid_data:
    reader = data_[0]
    readat1 = data_[1]
    readers = reader2elem[data_[0]]
    readed = [[item]+item2elem[item] for item in data_[2:]]
    readed = [a for b in readed for a in b]
    rnn_valid_dataset.append([reader, readat1] + readers + readed)
rnn_dev_dataset = []
for data_ in rnn_dev_data:
    reader = data_[0]
    readat1 = data_[1]
    readers = reader2elem[data_[0]]
    readed = [[item]+item2elem[item] for item in data_[2:]]
    readed = [a for b in readed for a in b]
    rnn_dev_dataset.append([reader, readat1] + readers + readed)
rnn_test_dataset = []
for data_ in rnn_test_data:
    reader = data_[0]
    readat1 = data_[1]
    readers = reader2elem[data_[0]]
    readed = [[item]+item2elem[item] for item in data_[2:]]
    readed = [a for b in readed for a in b]
    rnn_test_dataset.append([reader, readat1] + readers + readed)
train_dataset = data.TensorDataset(torch.from_numpy(np.array(rnn_train_dataset)))
valid_dataset = data.TensorDataset(torch.from_numpy(np.array(rnn_valid_dataset)))
dev_dataset = data.TensorDataset(torch.from_numpy(np.array(rnn_dev_dataset)))
test_dataset = data.TensorDataset(torch.from_numpy(np.array(rnn_test_dataset)))
torch.save(train_dataset, args.prepro_root+args.train_dataset_path)
torch.save(valid_dataset, args.prepro_root+args.valid_dataset_path)
torch.save(dev_dataset, args.prepro_root+args.dev_dataset_path)
torch.save(test_dataset, args.prepro_root+args.test_dataset_path)
