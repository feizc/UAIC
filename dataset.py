import torch 
from torch.utils.data import Dataset
import json
import os 
import h5py
from transformers import BertTokenizer

# SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[NONE]" "[IMG]", "[TXT]", "[PAD]"]


# Dataset for uncertainty measurer: bag-of-words 
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


def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)    


# Dataset for uncertainty-aware image captioning 
class UAICDataset(Dataset):

    "output: image region, input txt, output txt"
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer 
        self.img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'))
        data_pair_path = os.path.join(data_path, 'data_pair.json')
        with open(data_pair_path, 'r', encoding='utf-8') as j:
            self.txt = json.load(j)
        # self.bos = tokenizer._convert_token_to_id('[BOS]')
        self.img_label = tokenizer._convert_token_to_id('[IMG]')
        self.txt_label = tokenizer._convert_token_to_id('[TXT]')
    
    def __getitem__(self, i):
        cap_dict = self.txt[i]
        img_id = str(cap_dict['image_id']) + '_features' 
        img = torch.FloatTensor(self.img_features[img_id])
        place_holder = torch.Tensor([-100] * img.size(0)).long()
        input_ids = torch.Tensor(self.str2id('[BOS]'+ cap_dict['input'])).long()
        output_ids = torch.Tensor(self.str2id(cap_dict['output'])).long()
        output_ids = torch.cat([place_holder, output_ids], dim=0)
        token_type_ids = [self.img_label] * img.size(0) + [self.txt_label] * input_ids.size(0)
        token_type_ids = torch.Tensor(token_type_ids).long()

        return img, input_ids, output_ids, token_type_ids

    def str2id(self, sentence):
        sentence = self.tokenizer.tokenize(sentence)
        return self.tokenizer.convert_tokens_to_ids(sentence)

    def __len__(self):
        return len(self.txt)



if __name__ == "__main__":
    path = 'data'
    tokenizer = BertTokenizer('data/vocab.txt')
    #data_set = BagWordsDataset(path, tokenizer)
    #img, label = data_set[0] 
    data_set = UAICDataset(path, tokenizer)
    img, input, output = data_set[0]
    print(input)
    print(output)

