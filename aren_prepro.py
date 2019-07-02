import os
import json
import numpy as np
from tqdm import tqdm

magazine_path = './datasets/magazine.json'
metadata_path = './datasets/metadata.json'
users_path = './datasets/users.json'
predict_path = './datasets/predict/'
read_path = './datasets/predicts'

#magazine -> keyword list
magazine_list = []
for line in open(magazine_path, 'r', encoding='utf-8'):
    magazine_list.append(json.loads(line))
metadata_list = []
for line in open(metadata_path, 'r', encoding='utf-8'):
    metadata_list.append(json.loads(line))
'''
users_list = []
for line in open(users_path, 'r', encoding='utf-8'):
    users_list.append(json.loads(line))
'''

keyword_list = []
for data in tqdm(magazine_list):
    for keyword in data['magazine_tag_list']:
        if keyword not in keyword_list:
            keyword_list.append(keyword)
for data in tqdm(metadata_list):
    for keyword in data['keyword_list']:
        if keyword not in keyword_list:
            keyword_list.append(keyword)
'''
for data in tqdm(users_list):
    for keyword in data['keyword_list']:
        if keyword not in keyword_list:
            keyword_list.append(keyword)
'''

keyword_dict = {}
for i, keyword in enumerate(keyword_list):
    keyword_dict[keyword] = i
    
np.save('./prepro_results/keyword_dict.npy', keyword_dict)
np.save('./prepro_results/keyword_list.npy', keyword_list)

