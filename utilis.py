import json 
import numpy as np 
import os 
from collections import Counter 
import nltk 



# Create vocabulary from dataset conditioned on the min word freq 
def create_vocabulary(path, min_word_freq=5):
    train_path = os.path.join(path, 'captions_train2014.json')

    with open(train_path, 'r') as j:
        data = json.load(j)
    # ['info', 'images', 'licenses', 'annotations']
    # print(len(data['annotations']))

    word_freq = Counter()
    for ann in data['annotations']:
        #caption = nltk.word_tokenize(ann['caption'].lower())
        #pos_tags = nltk.pos_tag(caption)
        caption = ann['caption'].lower().split()
        word_freq.update(caption)

    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]

    #refine_words = []
    #for word, pos in pos_tags:
    #    if (pos == 'NN' or pos == 'VB' or pos == 'JJ' ) and word in words:
    #        refine_words.append(word)

    with open(os.path.join(path, 'vocab.txt'), 'w', encoding='utf-8') as f:
        for word in words:
            if '\'' in word:
                f.write(word.split('\'')[0] + '\n')
            elif word[-1] == '.' or word[-1] == ',':
                f.write(word[:-1] + '\n')
            else:
                f.write(word + '\n')


# real-time metric class
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 


# Generate training data pair 
def create_data_pair(data_path):
    uncertainty_json_path = os.path.join(data_path, 'uncertainty_captions.json')
    with open(uncertainty_json_path, 'r', encoding='utf-8') as j:
        uncertainty_json = json.load(j)
    train_data = uncertainty_json['annotations']
    data_pair_list = []
    for sample in train_data:
        print(sample)
        sample_uncertainty = sample['uncertainty'] 
        caption_uncertainty = sample['caption']
        order_list = [0] * len(sample_uncertainty)
        tree_construct(sample_uncertainty, 0, len(order_list)-1, order_list, 0)
        print(order_list)
        max_iter = 0 
        for x in order_list:
            max_iter = max(x, max_iter)
        
        for i in range(max_iter):
            break
        

    # print(b.argmax())


# create the binary tree structure recursively 
def tree_construct(uncertainty_list, left, right, res, level): 
    if left > right:
        return 
    if left == right:
        res[left] = level 
        return 
    idx = left
    max_value = uncertainty_list[idx]
    current_idx = left + 1  
    while current_idx <= right:
        if uncertainty_list[current_idx] > max_value:
            idx =  current_idx
            max_value = uncertainty_list[current_idx]
        current_idx += 1
    res[idx] = level
    tree_construct(uncertainty_list, left, idx-1, res, level+1)
    tree_construct(uncertainty_list, idx+1, right, res, level+1)
    return 


if __name__ == "__main__":
    # train_path = 'data/annotations'
    # create_vocabulary(train_path)

    data_path = 'data'
    create_data_pair(data_path)