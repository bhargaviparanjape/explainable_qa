import logging

from typing import Any, List

import torch
import torch.nn as nn

from transformers import BertForSequenceClassification, BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss, MSELoss
import pdb


class BertForTaskCumEvidenceClassfication(BertPreTrainedModel):
    def __init__(self, config):
        super(BertPreTrainedModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.max_query_length = config.max_query_length
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.evidence_classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, evidence_labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]
        sequence_output = outputs[0]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        sequence_output = sequence_output[:, self.max_query_length:]
        attention_mask = attention_mask[:, self.max_query_length:].contiguous()
        evidence_logits = self.evidence_classifier(sequence_output)
        outputs = (logits,evidence_logits) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        if evidence_labels is not None:
            evidence_loss_fct = CrossEntropyLoss()
            evidence_labels = evidence_labels[:, self.max_query_length:].contiguous()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = evidence_logits.view(-1, 2)[active_loss]
                active_labels = evidence_labels.view(-1)[active_loss]
                evidence_loss = evidence_loss_fct(active_logits, active_labels)
            else:
                evidence_loss = evidence_loss_fct(evidence_logits.view(-1, 2), evidence_labels.view(-1))
            outputs = (evidence_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertWithEvidenceFocusedAttention(BertPreTrainedModel):
    def __init__(self):
        pass

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, evidence_labels=None):
        # create a 768 dimensional representation that only focuses (equally on gold evidence)
        # Either Use a pal layer or
        # replace pooled attention with an interpolation of 12 heads of BERT and a specialized head for evidence
        # if this pooling works it provides evidence that while attention is not learning what it needs to...
        pass


