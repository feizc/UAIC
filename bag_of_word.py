from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel, AdamW 
import torch.nn as nn
import torch 
from dataset import BagWordsDataset 
import torch.optim 
from utilis import AverageMeter

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

    
    def forward(self, img_embs, labels):
        img_embs = self.feature_embd(img_embs)
        # print(img_embs.size())
        transformer_outputs = self.transformer(inputs_embeds=img_embs)
        hidden_states = transformer_outputs[1]
        pool_outputs = self.dropout(hidden_states)
        pool_outputs = self.classifier(pool_outputs)

        criterion = nn.MultiLabelSoftMarginLoss()
        loss = criterion(pool_outputs, labels)
        acc = self.acc_compute(pool_outputs, labels)

        return loss, acc


'''
def collate_fn(batch):
'''


def train():

    epochs = 25 
    model_path = 'model'

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
        for img, label in dataset:
            #print(img.size())
            #print(label.size())
            img = img.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            loss, acc = model(img, label)
        
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            #print(loss, acc)

            avg_loss.update(loss.item())
            avg_acc.update(acc)
            print('acc: ', acc, '  loss: ', loss)
            #break
        torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict()},\
                    '%s/epoch%d_acc_%.3f'%(model_path, epoch, avg_acc))

    print(loss_list)
    print(acc_list)




if __name__ == "__main__":
    train()

    #s = 'a cat shsiss dog'
    #s = tokenizer.tokenize(s)
    # print(tokenizer.convert_tokens_to_ids(s))


