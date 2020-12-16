import json 
import torch 
import os 

from transformers import BertTokenizer, BertForMaskedLM 
from model import UAIC 



def train():
    model_path = 'model'

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = UAIC.from_pretrained(model_path)

    

    '''
    torch.save(model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))
    model.config.to_json_file(os.path.join(model_path, 'config.json'))
    tokenizer.save_vocabulary(model_path)
    '''


if __name__ == "__main__":
    train()

