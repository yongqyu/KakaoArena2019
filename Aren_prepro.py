#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import json
import numpy as np
from tqdm import tqdm # tqdm_notebook in .ipynb

dataset_path = '/data/private/Arena/datasets/'
prepro_path = '/data/private/Arena/prepro_results/'

magazine_path = dataset_path+'magazine.json'
metadata_path = dataset_path+'metadata.json'
users_path = dataset_path+'users.json'
predict_path = dataset_path+'predict/'
read_path = dataset_path+'read/'


# In[2]:


# magazine -> keyword list
magazine_list = []
for line in open(magazine_path, 'r', encoding='utf-8'):
    magazine_list.append(json.loads(line))
metadata_list = []
for line in open(metadata_path, 'r', encoding='utf-8'):
    metadata_list.append(json.loads(line))
users_list = []
for line in open(users_path, 'r', encoding='utf-8'):
    users_list.append(json.loads(line))


# In[3]:


magazine_list[0], metadata_list[0], users_list[0]


# ## keyword / reader / writer dict and list

# In[ ]:


keyword_list = ['unk', 'pad']
for data in tqdm_notebook(magazine_list):
    for keyword in data['magazine_tag_list']:
        if keyword not in keyword_list:
            keyword_list.append(keyword)
for data in tqdm_notebook(metadata_list):
    for keyword in data['keyword_list']:
        if keyword not in keyword_list:
            keyword_list.append(keyword)
for data in tqdm_notebook(users_list):
    for keyword in data['keyword_list']:
        if keyword not in keyword_list:
            keyword_list.append(keyword)
            
keyword_dict = {}
for i, keyword in enumerate(keyword_list):
    keyword_dict[keyword] = i
    
np.save(prepro_path+'keyword_dict.npy', keyword_dict)
np.save(prepro_path+'keyword_list.npy', keyword_list)


# In[60]:


id2writer = ['unk']
id2reader = ['unk']
for data in tqdm_notebook(users_list):
    id2reader.append(data['id'])
    for writer in data['following_list']:
        if writer not in id2writer:
            id2writer.append(writer)

reader2id = {}
for i, reader in enumerate(id2reader):
    reader2id[reader] = i
    
writer2id = {}
for i, writer in enumerate(id2writer):
    writer2id[writer] = i
    
np.save(prepro_path+'id2reader.npy', id2reader)
np.save(prepro_path+'reader2id.npy', reader2id)
np.save(prepro_path+'id2writer.npy', id2writer)
np.save(prepro_path+'writer2id.npy', writer2id)


# In[26]:


magazine2id = {'unk':0}
id2magazine = ['unk']
for data in tqdm_notebook(magazine_list):
    id_ = int(data['id'])
    
    if id_ not in id2magazine:
        magazine2id[id_] = len(id2magazine)
        id2magazine.append(id_)
        
np.save(prepro_path+'id2magazine.npy', id2magazine)
np.save(prepro_path+'magazine2id.npy', magazine2id)


# ## reader / item 2 elements & item dict and list

# In[31]:


reader2elem = {'unk':0}
follow_list = []
follow_maxlen = 8
keyword_maxlen = 8
for data in tqdm_notebook(users_list):
    #follow_list.append(len(data['following_list']))
    id_ = reader2id[data['id']]
    follow = [writer2id['unk']] * (follow_maxlen-len(data['following_list'])) +              list(map(writer2id.get, data['following_list'][-follow_maxlen:]))

    keywords = []
    if data['keyword_list']:
        keywords = [kw['keyword'].split(' ') for kw in data['keyword_list']]
        keywords = list(set([a for b in keywords for a in b]))
        keywords = [keyword_dict[kw] for kw in keywords if keyword_dict.get(kw) != None]
        
    keywords = [keyword_dict['pad']] * (keyword_maxlen-len(keywords)) + keywords[:keyword_maxlen]
    reader2elem[id_] = follow + keywords

np.save(prepro_path+'reader2elem.npy', reader2elem)


# In[361]:


ts_min = 1538319600 # 20181001000000 GMT+8
ts_max = 1552575600 # 20190315000000 GMT+8
ts_gap = ts_max - ts_min

keywd_maxlen = 5
item2elem = {0:[0,0,0,0,0,0,0,0]}
item_list = ['unk']
item_dict = {'unk':0}

for data in tqdm_notebook(metadata_list):

    if item_dict.get(data['id']) == None:
        item_dict[data['id']] = len(item_list)
        item_list.append(data['id'])
        
    if data['keyword_list']:
        keywd = [keyword_dict['pad']] * (keywd_maxlen-len(data['keyword_list'])) +                 list(map(keyword_dict.get, data['keyword_list'][::-1]))
    else:
        keywd = [keyword_dict['unk']] * keywd_maxlen
    writer = writer2id[data['user_id']]
    reg_ts = int(data['reg_ts'])/1000
    reg_ts = (reg_ts-ts_min)/ts_gap if reg_ts > ts_min else 0

    if magazine2id.get(int(data['magazine_id'])) == None:
        magazine2id[int(data['magazine_id'])] = len(id2magazine)
        id2magazine.append(int(data['magazine_id']))
    mag_id = magazine2id[int(data['magazine_id'])]
    
    item2elem[item_dict[data['id']]] = [writer] + keywd + [reg_ts, mag_id]

np.save(prepro_path+'item_dict.npy', item_dict)
np.save(prepro_path+'item_list.npy', item_list)
np.save(prepro_path+'item2elem.npy', item2elem)
np.save(prepro_path+'id2magazine.npy', id2magazine)
np.save(prepro_path+'magazine2id.npy', magazine2id)


# ## Valid Tensor & Writer 2 items

# In[22]:


import numpy as np
from tqdm import tqdm_notebook
import torch

valid_writer_keywd = [[0,0,0,0,0,0,0,0,0]]

for data in tqdm_notebook(metadata_list):
    if data['keyword_list']:
        keywd = [keyword_dict['pad']] * (keywd_maxlen-len(data['keyword_list'])) +                 list(map(keyword_dict.get, data['keyword_list'][::-1]))
    else:
        keywd = [keyword_dict['unk']] * keywd_maxlen
    writer = writer2id[data['user_id']]
    reg_ts = int(data['reg_ts'])/1000
    reg_ts = (reg_ts-ts_min)/ts_gap if reg_ts > ts_min else 0
    if magazine2id.get(int(data['magazine_id'])) == None:
        magazine2id[int(data['magazine_id'])] = len(id2magazine)
        id2magazine.append(int(data['magazine_id']))
    mag_id = magazine2id[int(data['magazine_id'])]
    item_id = item_dict[data['id']]
    
    valid_writer_keywd.append([item_id, writer] + keywd + [reg_ts, mag_id])

valid_writer_keywd = torch.from_numpy(np.array(valid_writer_keywd))
torch.save(valid_writer_keywd, prepro_path+'valid_writer_keywd.pkl')


# In[27]:


writerid2items = {}
for data in tqdm_notebook(metadata_list):
    user_id = writer2id[data['user_id']]
    id_ = item_dict[data['id']]
    keyword = keyword_dict[data['keyword_list'][0] if data['keyword_list'] is True else '없음']
    
    if writerid2items.get(user_id) == None:
        writerid2items[user_id] = [[id_, user_id, keyword]]
    else:
        writerid2items[user_id].append([id_, user_id, keyword])
        
np.save(prepro_path+'writerid2items.npy', writerid2items)


# ## From 2019022 : Reader 2 Read item

# In[53]:


import time
import datetime
reader2item = {}
file_list = os.listdir(read_path)
file_list.sort()

for read_file in tqdm_notebook(file_list[3456:]): #2월~(2952) #2월22~(3456) #2월20~(3408)
    try:
        file = open(read_path+read_file, 'r')
        data_ = file.readlines()
    except:
        print(read_file)
        continue
        
    file_ts = time.mktime(datetime.datetime.strptime(read_file[-10:], '%Y%m%d%H').timetuple()) + 32400
    file_ts = (file_ts-ts_min)/ts_gap if file_ts > ts_min else 0
    if file_ts < 0:
        print('xx')

    for line in data_:
        tokens = line.split(' ')
        try:
            reader = reader2id[tokens[0]]
        except:
            continue
        items = [[item_dict[x], file_ts] if item_dict.get(x)!=None else [item_dict['unk'], file_ts] for x in tokens[1:-1]]
        
        if reader2item.get(reader) != None:
            reader2item[reader] = reader2item[reader] + items
        else:
            reader2item[reader] = items
            
np.save(prepro_path+'reader2item.npy', reader2item)


# In[287]:


userid2followid = {}
for data in users_list:
    id_ = reader2id[data['id']]
    following_list = [writer2id[x] for x in data['following_list']]
    userid2followid[id_] = following_list
    
np.save(prepro_path+'userid2followid.npy', userid2followid)


# # RNN Based
# Train Valid Test data (from 4 to 1 : window size = 5) and Mask 

# In[343]:


rnn_train_data = []
rnn_valid_data = []
rnn_test_data = {}
window_size = 5
for reader, items_list in reader2item.items():
    if not items_list:
        continue
    items_array = np.array(items_list)
    items, read_ts = items_array[:,0].tolist(), items_array[:,1].tolist()

    if len(items) < window_size:
        items = [item_dict['unk']] * (window_size-len(items)) + items
        read_ts = [0] * (window_size-len(read_ts)) + read_ts
        
    rnn_data = []
    for i in range(len(items)-window_size+1):
        rnn_data.append([reader, read_ts[i+window_size-1]] + items[i:i+window_size])
        
    if len(items) > window_size+4:
        rnn_train_data += rnn_data[int(len(rnn_data)*0.1):]
        rnn_valid_data += rnn_data[:int(len(rnn_data)*0.1)]
    else:
        rnn_train_data += rnn_data
    rnn_test_data[reader] = rnn_data[-1]
        
np.save(prepro_path+'rnn_train_data.npy', rnn_train_data)
np.save(prepro_path+'rnn_valid_data.npy', rnn_valid_data)


# Test data as not a dictionary (REAL)

# In[344]:


pred_dev_file = predict_path+'dev.users'
pred_dev_data = open(pred_dev_file, 'r').readlines()
test_data = []
a = 0
for line in pred_dev_data:
    if reader2id.get(line.strip()) != None:
        reader = reader2id[line.strip()]
        readed = rnn_test_data[reader] if rnn_test_data.get(reader)!=None else [0] * (window_size+2)
    else:
        reader = reader2id['unk']
        readed = [0] * (window_size+2)
        a += 1
    test_data.append(readed)

np.save(prepro_path+'rnn_test_data.npy', np.array(test_data))


# In[70]:


pred_dev_file = predict_path+'dev.users'
pred_dev_data = open(pred_dev_file, 'r').readlines()

dev_mask = np.ones((len(pred_dev_data), 643105))
for i, line in enumerate(pred_dev_data):
    try:
        readed = reader2items[reader2id[line.strip()]]
        readed = list(set(np.array(readed)[:,0].astype(np.int32).tolist()))
    except:
        continue
    dev_mask[i,readed] = 0
dev_mask[:,0] = 0
    
np.save(prepro_path+'dev_mask.npy', dev_mask)

