import math
import torch

def hsort(preds, split_size=1024):
    hidden_dot_product = [torch.sort(preds[:,split_size*k:split_size*(k+1)], descending=True)[1][:,:100] + k*split_size
                          for k in range((preds.size(1)//split_size)+1)]
    hidden_dot_product = torch.cat(hidden_dot_product, 1)
    sorted_dot_product = torch.sort(preds.gather(1, hidden_dot_product), descending=True)[1][:,:100]
    sorted_dot_product = hidden_dot_product.gather(1, sorted_dot_product)
    return sorted_dot_product

def ap(preds, targets):
    preds = hsort(preds)
    ap = []
    for pred, target in zip(preds, targets):
        #ap.append(sum([(pred == x).nonzero()/(i+1) if x in pred else torch.zeros(1)
        ap.append(sum([(x in pred)/(i+1) for i, x in enumerate(target)]) / len(pred))
    return sum(ap)/preds.size(0)

def cal_ndcg(pred, target):
    top_k = full[full['rank']<=top_k]
    test_in_top_k =top_k[top_k['test_item'] == top_k['item']]
    test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
    return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()
