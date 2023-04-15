import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel

class BertVAD(nn.Module):
    def __init__(self, dropout=0.5, pre_train_name='bert-base-uncased'):
        super().__init__()
        self.pre_train_name = pre_train_name
        self.tokenizer = BertTokenizer.from_pretrained(
                            pre_train_name,
                            cache_dir='./.cache'
                         )
        self.bert = BertModel.from_pretrained(
                       pre_train_name,
                       cache_dir='./.cache'
                    )
        self.fc = nn.Sequential(
                    nn.Linear(self.bert.config.hidden_size, 64),
                    nn.Dropout(dropout),
                    nn.Linear(64, 3)
                  )

    def forward(self, text, device='cpu'):
        tokenized_text = self.tokenizer(
                            text,
                            add_special_tokens=True,
                            padding='max_length',
                            max_length=256,
                         )
        input_ids = torch.tensor(
                        tokenized_text['input_ids']
                    ).to(device)
        attn_masks = torch.tensor(
                        tokenized_text['attention_mask']
                     ).to(device)
        enc_outs = self.bert(input_ids, attn_masks)
        logits = self.fc(enc_outs.pooler_output)
        return logits
