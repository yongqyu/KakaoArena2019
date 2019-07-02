import random
import numpy as np
import torch
import torch.utils.data as data

item_dict = np.load('/data/private/Arena/prepro_results/item_dict.npy', allow_pickle=True).item()
item_list = np.load('/data/private/Arena/prepro_results/item_list.npy')
keyword_dict = np.load('/data/private/Arena/prepro_results/keyword_dict.npy', allow_pickle=True).item()
keyword_list = np.load('/data/private/Arena/prepro_results/keyword_list.npy')
id2reader = np.load('/data/private/Arena/prepro_results/id2reader.npy')
reader2id = np.load('/data/private/Arena/prepro_results/reader2id.npy', allow_pickle=True).item()
id2writer = np.load('/data/private/Arena/prepro_results/id2writer.npy')
writer2id = np.load('/data/private/Arena/prepro_results/writer2id.npy', allow_pickle=True).item()
item2keywd = np.load('/data/private/Arena/prepro_results/item2keywd.npy', allow_pickle=True).item()
keyword_dict['없음'] = len(keyword_list)
keyword_list = list(keyword_list)
keyword_list.append('없음')

num_keywords = len(keyword_dict)
num_readers = len(id2reader)
num_writers = len(id2writer)
print(num_keywords, num_readers, num_writers)

class DataFolder(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, dataset_path, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.dataset = np.load(dataset_path)
        self.mode = mode
        self.num_samples = self.dataset.shape[0]

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        line = self.dataset[index]
        reader = int(line[0])
        sub_keywd = keyword_dict[item2keywd[line[1]][0]
                    if item2keywd.get(line[1]) != None and item2keywd[line[1]] is True
                    else '없음']
        sub = writer2id[line[1].split('_')[0]] if writer2id.get(line[1].split('_')[0]) != None else writer2id['unk']
        neg_keywd = list(range(num_keywords)); neg_keywd.remove(sub_keywd)
        neg_keywd = np.array(random.sample(neg_keywd, 1))#len(sub_keywd))
        neg = list(range(num_writers)); neg.remove(sub)
        neg = np.array(random.sample(neg, 1))#len(sub))

        return reader, sub_keywd, sub, neg_keywd, neg

    def __len__(self):
        """Return the number of images."""
        return self.num_samples

def get_loader(dataset_path, batch_size=16, mode='train', num_workers=4):
    """Build and return a data loader."""
    dataset = DataFolder(dataset_path, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
