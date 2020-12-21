from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  
import torch 
from model import UAIC 
from transformers import BertTokenizer
import os 
import json 
import h5py 



# generate candidates using inout image 
def generate_cap(img_feature, model, tokenizer):
    return 'a bicycle a clock wheel'


# concatenate the substage to total sentence 


def eval():
    ckpt_path = 'ckpt'
    data_path = 'data'

    tokenizer = BertTokenizer.from_pretrained(ckpt_path)
    model = UAIC.from_pretrained(ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    annotation_path = os.path.join(data_path, 'annotations')
    val_path = os.path.join(annotation_path, 'captions_val2014.json')
    val_data = json.load(open(val_path, 'r'))
    val_data = val_data['annotations']
    img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'))
    
    with torch.no_grad():
        for instance in val_data:
            print(instance)
            img_id = str(instance['image_id']) + '_features'
            img_feature = torch.FloatTensor(img_features[img_id])
            candidates = generate_cap(img_feature, model, tokenizer)
            reference = instance['caption']
            smooth = SmoothingFunction()
            print(corpus_bleu([[reference]], [candidates], smoothing_function=smooth.method1))
            break


if __name__ == "__main__": 
    eval()