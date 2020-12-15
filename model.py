from transformers import * 
import torch.nn as nn
import torch 

class UAIC(BertPreTrainedModel):
    def __init__(self, config):
        super(UAIC, self).__init__(config)
        self.transformer = BertModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.image_ff = nn.Linear(2048, config.n_embd)
        self.image_inverse_ff = nn.Linear(config.n_embd, 2048)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.transformer.embeddings.word_embeddings)

    def forward(self, input_embs, labels=None):
        

