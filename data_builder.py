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
id2magazine = np.load('/data/private/Arena/prepro_results/id2magazine.npy')
magazine2id = np.load('/data/private/Arena/prepro_results/magazine2id.npy', allow_pickle=True).item()

max_keylen = 5
num_keywords = len(keyword_dict)
num_readers = len(id2reader)
num_writers = len(id2writer)
num_items = len(item_list)
num_magazine = len(id2magazine)
print(num_keywords, num_readers, num_writers, num_items, num_magazine)

rnn_train_data = np.load('/data/private/Arena/prepro_results/rnn_train_data.npy')
rnn_valid_data = np.load('/data/private/Arena/prepro_results/rnn_valid_data.npy')
rnn_test_data = np.load('/data/private/Arena/prepro_results/rnn_test_data.npy')
rnn_train_dataset = []
for data_ in rnn_train_data:
    reader = np.array([[data_[0]]] * (len(data_)-2))
    readat1 = np.array([[data_[1]]] *  (len(data_)-2))
    #readat2 = np.array([[data_[2]]] *  (len(data_)-3))
    readed = np.array([[item]+item2elem[item] for item in data_[2:]])
    rnn_train_dataset.append(np.concatenate((reader, readat1, readed), 1))
rnn_valid_dataset = []
for data_ in rnn_valid_data:
    reader = np.array([[data_[0]]] * (len(data_)-2))
    readat1 = np.array([[data_[1]]] *  (len(data_)-2))
    #readat2 = np.array([[data_[2]]] *  (len(data_)-3))
    readed = np.array([[item]+item2elem[item] for item in data_[2:]])
    rnn_valid_dataset.append(np.concatenate((reader, readat1, readed), 1))
rnn_test_dataset = []
for data_ in rnn_test_data:
    reader = np.array([[data_[0]]] * (len(data_)-2))
    readat1 = np.array([[data_[1]]] *  (len(data_)-2))
    #readat2 = np.array([[data_[2]]] *  (len(data_)-3))
    readed = np.array([[item]+item2elem[item] for item in data_[2:]])
    rnn_test_dataset.append(np.concatenate((reader, readat1, readed), 1))
train_dataset = data.TensorDataset(torch.from_numpy(np.array(rnn_train_dataset)))
valid_dataset = data.TensorDataset(torch.from_numpy(np.array(rnn_valid_dataset)))
test_dataset = data.TensorDataset(torch.from_numpy(np.array(rnn_test_dataset)))
torch.save(train_dataset, '/data/private/Arena/prepro_results/train_dataset.pkl')
torch.save(valid_dataset, '/data/private/Arena/prepro_results/valid_dataset.pkl')
torch.save(test_dataset, '/data/private/Arena/prepro_results/test_dataset.pkl')

'''
for data_ in rnn_train_data:
    reader = np.array([[data_[0]]] * (len(data_)-2))
    readat = data_[1]
    readed = np.array([[item]+item2elem[item] for item in data_[2:]])
    if sum(readat < readed[:,7]) > 0:
        print(readat, readed[:,7], readed[:,0])
        continue
    readed[:,7] = (readat - readed[:,7]) / ts_gap
    readed_sq = np.square(readed[:,7])
    readed_sqrt = np.sqrt(readed[:,7])
    readed = np.concatenate((readed[:,:8], np.array([readed_sq, readed_sqrt]).transpose(), readed[:,8:]), 1)
    rnn_train_dataset.append(np.concatenate((reader, readed), 1))
rnn_valid_dataset = []
for data_ in rnn_valid_data:
    reader = np.array([[data_[0]]] * (len(data_)-2))
    readat = data_[1]
    readed = np.array([[item]+item2elem[item] for item in data_[2:]])
    if sum(readat < readed[:,7]) > 0:
        print(readat, readed[:,7], readed[:,0])
        continue
    readed[:,7] = (readat - readed[:,7]) / ts_gap
    readed_sq = np.square(readed[:,7])
    readed_sqrt = np.sqrt(readed[:,7])
    readed = np.concatenate((readed[:,:8], np.array([readed_sq, readed_sqrt]).transpose(), readed[:,8:]), 1)
    rnn_valid_dataset.append(np.concatenate((reader, readed), 1))
rnn_test_dataset = []
for data_ in rnn_test_data:
    reader = np.array([[data_[0]]] * (len(data_)-2))
    readat = 1551369600 # 20190301000000 GMT+8
    readed = np.array([[item]+item2elem[item] for item in data_[2:]])
    readed[:,7] = (readat - readed[:,7]) / ts_gap
    readed_sq = np.square(readed[:,7])
    readed_sqrt = np.sqrt(readed[:,7])
    readed = np.concatenate((readed[:,:8], np.array([readed_sq, readed_sqrt]).transpose(), readed[:,8:]), 1)
    rnn_test_dataset.append(np.concatenate((reader, readed), 1))
train_dataset = data.TensorDataset(torch.from_numpy(np.array(rnn_train_dataset)))
valid_dataset = data.TensorDataset(torch.from_numpy(np.array(rnn_valid_dataset)))
test_dataset = data.TensorDataset(torch.from_numpy(np.array(rnn_test_dataset)))
torch.save(train_dataset, '/data/private/Arena/prepro_results/train_dataset.pkl')
torch.save(valid_dataset, '/data/private/Arena/prepro_results/valid_dataset.pkl')
torch.save(test_dataset, '/data/private/Arena/prepro_results/test_dataset.pkl')
'''
