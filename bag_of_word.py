from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel 
import torch.nn as nn
import torch


# Uncertainty-aware image-conditioned bag-of-words
class BagofWords(BertPreTrainedModel):
    def __init__(self, config):
        super(BagofWords, self).__init__(config)
        self.transformer = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.init_weights()
    
    def forward(self, img_embs, labels):
        transformer_outputs = self.transformer(inputs_embeds=img_embs)
        hidden_states = transformer_outputs[1]
        pool_outputs = self.dropout(hidden_states)
        pool_outputs = self.classifier(pool_outputs)

        criterion = nn.MultiLabelSoftMarginLoss()
        loss = criterion(pool_outputs, labels)

        return loss



if __name__ == "__main__":
    tokenizer = BertTokenizer('data/vocab.txt')
    s = 'a cat shsiss dog'
    s = tokenizer.tokenize(s)
    print(tokenizer.convert_tokens_to_ids(s))


