from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel, AdamW 
import torch.nn as nn
import torch 
from dataset import BagWordsDataset 
import torch.optim 
from utilis import AverageMeter
import os 
import json 
import h5py 
from torch.nn.functional import softmax 


# Uncertainty-aware image-conditioned bag-of-words
class BagofWords(BertPreTrainedModel):
    def __init__(self, config):
        super(BagofWords, self).__init__(config)
        self.feature_embd = nn.Linear(2048, config.hidden_size)
        self.transformer = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.init_weights()

    def acc_compute(self, output, labels):
        output = output * labels
        thres = torch.tensor([[0.3]]).expand_as(output)
        summ = torch.ge(output, thres).sum().item()
        total = labels.sum().item()
        return summ/total

    
    def forward(self, img_embs, labels=None):
        img_embs = self.feature_embd(img_embs)
        # print(img_embs.size())
        transformer_outputs = self.transformer(inputs_embeds=img_embs)
        hidden_states = transformer_outputs[1]
        pool_outputs = self.dropout(hidden_states)
        pool_outputs = self.classifier(pool_outputs)

        if labels is None:
            return pool_outputs

        criterion = nn.MultiLabelSoftMarginLoss()
        loss = criterion(pool_outputs, labels)
        acc = self.acc_compute(pool_outputs, labels)

        return loss, acc


# train the image conditioned bag-of-words 
def train():

    epochs = 25 
    model_path = 'model'
    gradient_accumlation_steps = 5 

    tokenizer = BertTokenizer('data/vocab.txt') 
    # print(tokenizer.vocab.get('[UNK]'))
    configuration = BertConfig(vocab_size=tokenizer.vocab.get('[UNK]') + 1, \
                                num_hidden_layers=3, \
                                intermediate_size=2048)
    model = BagofWords(configuration)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    dataset =  BagWordsDataset('data', tokenizer)
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model.train()
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        iteration = 1 
        for img, label in dataset:
            #print(img.size())
            #print(label.size())
            img = img.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            loss, acc = model(img, label)
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if iteration % gradient_accumlation_steps == 0: 
                optimizer.zero_grad()
                optimizer.step()
                avg_loss.update(loss.item() / gradient_accumlation_steps)
                break

            #print(loss, acc)
            avg_acc.update(acc)
            print('acc: ', acc)
            #break
            iteration += 1 
        torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict()},\
                    '%s/epoch%d_acc_%.3f'%(model_path, epoch, avg_acc.avg))
        model.config.to_json_file(os.path.join(model_path, 'config.json'))
        # tokenizer.save_vocabulary(model_path)
        break
        loss_list.append(avg_loss.avg)
        acc_list.append(avg_loss.avg)

    print(loss_list)
    print(acc_list)


# use the bag-of-words model to comput the uncertainty 
# output: json image_id, caption, uncertainty 
def uncertainty_estimation():

    path = 'model'
    ckpt_path = 'model/pytorch_model.bin'
    data_path = 'data'

    
    tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=True)
    model_config = BertConfig.from_pretrained(path)
    model = BagofWords(model_config)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval() 

    unk = tokenizer._convert_token_to_id('[UNK]')
    
    img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'))
    caption_path = os.path.join(data_path, 'annotations')
    train_caption_path = os.path.join(caption_path, 'captions_train2014.json')
    with open(train_caption_path, 'r', encoding='utf-8') as j:
        captions = json.load(j)
    annotation_list = []
    i = 1
    for sample in captions['annotations']:
        img_id = str(sample['image_id']) + '_features'
        caption = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['caption']))

        input_f = torch.FloatTensor(img_features[img_id]).view(1,-1, 2048)
        pro_vocab = model(img_embs = input_f).view(-1)
        # print(softmax(pro_vocab, 1))
        uncertainty = []
        for word_idx in caption:
            if word_idx == unk:
                uncertainty.append(0)
            else:
                uncertainty.append(pro_vocab[word_idx].item())
        sample['uncertainty'] = uncertainty
        annotation_list.append(sample)
        print(i)
        i += 1
        # print(annotation_list)
        # break
    new_captions = {}
    new_captions['annotations'] = annotation_list 
    with open(os.path.join(data_path, 'uncertainty_captions.json'), 'w', encoding='utf-8') as f:
        json.dump(new_captions, f, indent=4)


if __name__ == "__main__":
    # train the image conditioned bag-of-words 
    train()

    # uncertainty estimation with trained bag-of-word model 
    uncertainty_estimation()


    #s = 'a cat shsiss dog'
    #s = tokenizer.tokenize(s)
    # print(tokenizer.convert_tokens_to_ids(s))


