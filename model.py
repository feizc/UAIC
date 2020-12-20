from transformers import * 
import torch.nn as nn
import torch 
from torch.nn import CrossEntropyLoss 
import os 


class UAIC(BertPreTrainedModel):
    def __init__(self, config):
        super(UAIC, self).__init__(config)
        self.transformer = BertModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.image_ff = nn.Linear(2048, config.hidden_size)
        self.image_inverse_ff = nn.Linear(config.hidden_size, 2048)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.transformer.embeddings.word_embeddings)

    def forward(self, input_embs, labels=None):
        transformer_outputs = self.transformer(inputs_embeds=input_embs)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,) + transformer_outputs[1:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(lm_logits.squeeze(0), labels)
            outputs = (loss,) + outputs
        
        return outputs 


if __name__ == "__main__":
    model_path = 'ckpt'
    configration = BertConfig(vocab_size=10876)
    model = UAIC(configration)
    torch.save(model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))
    model.config.to_json_file(os.path.join(model_path, 'config.json'))
    # batch_size dim can not be forgot
    input_embs = torch.rand(5, 768).view(1, -1, 768)
    output = model(input_embs)
    print(output[0].size())
