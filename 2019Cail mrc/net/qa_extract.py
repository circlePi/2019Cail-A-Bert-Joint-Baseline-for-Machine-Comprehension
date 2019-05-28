import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class QaExtract(BertPreTrainedModel):
    def __init__(self, config):
        super(QaExtract, self).__init__(config)
        self.bert = BertModel(config)
        # for p in self.bert.parameters():
        #     p.requires_grad = False
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)
        self.answer_type_classifier = nn.Linear(config.hidden_size, 4)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        sequence_output, pooled_output = self.bert(input_ids,
                                                   token_type_ids,
                                                    attention_mask,
                                                    output_all_encoded_layers=output_all_encoded_layers)  # (B, T, 768)
        logits = self.classifier(sequence_output)                                          # (B, T, 2)
        start_logits, end_logits = logits.split(1, dim=-1)                                 # ((B, T, 1), (B, T, 1))
        start_logits = start_logits.squeeze(-1)                                            # (B, T)
        end_logits = end_logits.squeeze(-1)                                                # (B, T)
        answer_type_logits = self.answer_type_classifier(pooled_output)
        return start_logits, end_logits, answer_type_logits