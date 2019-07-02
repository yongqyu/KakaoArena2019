import json
import numpy as np
from tqdm import tqdm

old_keyword_list = np.load('./prepro_results/keyword_list.npy')
old_keyword_dict = np.load('./prepro_results/keyword_dict.npy', allow_pickle=True).item()

users_path = './datasets/users.json'
users_list = []
for line in open(users_path, 'r', encoding='utf-8'):
    users_list.append(json.loads(line))
keyword_list = []
for data in tqdm(users_list):
    for keyword in data['keyword_list']:
        if keyword not in keyword_list:
            keyword_list.append(keyword)

for i, keyword in enumerate(keyword_list):
    keyword_dict[keyword] = i+len(old_keyword_list)

keyword_list = old_keyword_list + keyword_list

np.save('./prepro_results/new_keyword_dict.npy', keyword_dict)
np.save('./prepro_results/new_keyword_list.npy', keyword_list)
