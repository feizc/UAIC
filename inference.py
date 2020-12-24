from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  
import torch 
from model import UAIC 
from transformers import BertTokenizer
import os 
import json 
import h5py 
from train import SPECIAL_TOKENS 

#  ["[BOS]", "[EOS]", "[NONE]" "[IMG]", "[TXT]", "[PAD]"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# generate candidates using inout image 
def generate_cap(img_feature, model, tokenizer): 
    bos, eos, none, img, txt = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS) 
    input_txt = ['[bos]'] 
    output_ids = []
    while True: 
        if len(output_ids) > 0 and output_ids == [none] * len(output_ids): 
            break 
        input_txt = tokenizer.convert_tokens_to_ids(input_txt)
        input_txt = torch.Tensor(input_txt).long().to(device)
        input_embs = model.transformer.embeddings.word_embeddings(input_txt)
        
        img_embs = model.image_ff(img_feature)
        input_embs = torch.cat([img_embs, input_embs], dim=0)

        # print(input_txt.size())
        token_type_ids = [img] * img_embs.size(0) + [txt] * input_txt.size(0)
        token_type_ids = torch.Tensor(token_type_ids).long()
        
        token_type_embs = model.transformer.embeddings.word_embeddings(token_type_ids)
        
        input_embs = input_embs + token_type_embs 
        
        out = model(input_embs.view(1, -1, 768)) 
        out = out[0].squeeze(0)[-input_txt.size(0):,:]
        
        output = torch.argmax(out, dim=1)
        output_ids = output.cpu().numpy().tolist()
        output_txt = tokenizer.convert_ids_to_tokens(output_ids)
        input_txt = sequence_stage_combine(input_txt, output_txt)

    return input_txt 


# concatenate the substage to total sentence 
def sequence_stage_combine(input_txt, output_txt):
    new_sequence = []
    idx = 0
    while idx < len(input_txt):
        new_sequence.append(input_txt[idx])
        if output_txt[idx] !=  '[NONE]': 
            new_sequence.append(output_txt[idx])
        idx += 1 
    return new_sequence 


# evaluate the ckpt 
def eval():
    ckpt_path = 'ckpt'
    data_path = 'data'

    tokenizer = BertTokenizer.from_pretrained(ckpt_path)
    model = UAIC.from_pretrained(ckpt_path)
    model = model.to(device)
    model.eval()
    smooth = SmoothingFunction() 

    annotation_path = os.path.join(data_path, 'annotations')
    val_path = os.path.join(annotation_path, 'captions_val2014.json')
    val_data = json.load(open(val_path, 'r'))
    val_data = val_data['annotations']
    img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'))
    
    with torch.no_grad(): 
        results = []
        for instance in val_data:
            print(instance)
            img_id = str(instance['image_id']) + '_features'
            img_feature = torch.FloatTensor(img_features[img_id]).to(device)
            candidates = generate_cap(img_feature, model, tokenizer)
            reference = instance['caption']
            results.append(corpus_bleu([[reference]], [candidates], smoothing_function=smooth.method1))


if __name__ == "__main__": 
    eval()