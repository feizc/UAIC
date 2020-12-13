import json 
import numpy as np 
import os 
from collections import Counter 
import nltk

# Create vocabulary from dataset
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


if __name__ == "__main__":
    train_path = 'data/annotations'

    create_vocabulary(train_path)
