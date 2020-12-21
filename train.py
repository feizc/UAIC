import json 
import torch 
import os 

from transformers import BertTokenizer, BertForMaskedLM, AdamW 
from model import UAIC 
from dataset import UAICDataset 

SPECIAL_TOKENS_DICT = {'bos_token': "[BOS]", 'eos_token': "[EOS]", 'additional_special_tokens': ["[NONE]", "[IMG]", "[TXT]"], 'pad_token': "[PAD]"}
SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[NONE]" "[IMG]", "[TXT]", "[PAD]"]

def train():
    model_path = 'ckpt'
    data_path = 'data'
    # ckpt_path = 'ckpt'
    lr = 1e-4
    epochs = 15 
    gradient_accumulation_steps = 5 

    tokenizer = BertTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model = UAIC.from_pretrained(model_path)
    # print(len(tokenizer))
    # model.transformer.resize_token_embeddings(10877)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    
    train_data = UAICDataset(data_path, tokenizer)
    # bos, eos, none, img_label, txt_label = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    for epoch in range(epochs):
        
        iteration = 1
        for img, input_txt, output, token_type_ids in train_data:
            # print(input.size(), output.size(), img.size()) 
            img = img.to(device)
            input_txt = input_txt.to(device)
            output = output.to(device)

            input_embs = model.transformer.embeddings.word_embeddings(input_txt)
            img_embs = model.image_ff(img)
            input_embs = torch.cat([img_embs, input_embs], dim=0)
            token_type_embs = model.transformer.embeddings.word_embeddings(token_type_ids)
            input_embs = input_embs + token_type_embs 
            out = model(input_embs.view(1, -1, 768), output)
            loss = out[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if iteration % gradient_accumulation_steps == 0: 
                optimizer.step()
                optimizer.zero_grad()
            
            print(loss.item())
            iteration += 1
            

    
    torch.save(model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))
    model.config.to_json_file(os.path.join(model_path, 'config.json'))
    # tokenizer.save_vocabulary(model_path)
    


if __name__ == "__main__":
    train()

