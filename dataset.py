import torch 
from torch.utils.data import Dataset
import json
import os 
import h5py
from transformers import BertTokenizer

class BagWordsDataset(Dataset): 

    "output: image region, one-hot vocabulary vector"
    def __init__(self, data_path, tokenizer):
        self.img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'))
        # 'xxx_features', 'xxx_boxes', 'xxx_cls_prob'
        # print(self.h['999_features'].shape) (x, 2048)
        caption_path = os.path.join(data_path, 'annotations')
        train_caption_path = os.path.join(caption_path, 'captions_train2014.json')
        with open(train_caption_path, 'r', encoding='utf-8') as j:
            self.captions = json.load(j) 
        # print(len(self.captions['annotations']))
        self.tokenizer = tokenizer

    def __getitem__(self, i): 
        cap_dict = self.captions['annotations'][i]
        img_id = str(cap_dict['image_id']) + '_features'
        img = torch.FloatTensor(self.img_features[img_id])
        caption = cap_dict['caption']
        caption = self.tokenizer.tokenize(caption)
        caption = self.tokenizer.convert_tokens_to_ids(caption) 
        label = torch.zeros(self.tokenizer.vocab.get('[UNK]') + 1)
        for idx in caption:
            label[idx] = 1 
        return img, label
    
    def __len__(self):
        return len(self.captions['annotations'])


if __name__ == "__main__":
    path = 'data'
    tokenizer = BertTokenizer('data/vocab.txt')
    data_set = BagWordsDataset(path, tokenizer)
    img, label = data_set[0]

