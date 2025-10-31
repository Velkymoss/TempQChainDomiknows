import torch
from torch import nn
from transformers import (
    BertModel,
    BertTokenizer,
)


class BERTTokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        special_tokens = ["<e1>", "</e1>", "<e2>", "</e2>"]
        self.tokenizer.add_tokens(special_tokens)

    def __call__(self, _, question, story):
        encoded_input = self.tokenizer(question, story, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        return input_ids, attention_mask


class Bert(nn.Module):
    def __init__(self, num_classes=6, drp=False, device="cpu", tokenizer=None):
        super().__init__()

        self.device = device
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        if tokenizer is not None:
            self.bert.resize_token_embeddings(len(tokenizer), mean_resizing=True)
            self.register_buffer("e1_start_id", torch.tensor(tokenizer.convert_tokens_to_ids("<e1>")))
            self.register_buffer("e1_end_id", torch.tensor(tokenizer.convert_tokens_to_ids("</e1>")))
            self.register_buffer("e2_start_id", torch.tensor(tokenizer.convert_tokens_to_ids("<e2>")))
            self.register_buffer("e2_end_id", torch.tensor(tokenizer.convert_tokens_to_ids("</e2>")))

        dropout_prob = 0.0 if drp else self.bert.config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = self.bert.config.hidden_size
        self.num_classes = num_classes

        self.mlp1 = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size), nn.GELU(), nn.Dropout(dropout_prob)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.hidden_size // 2, num_classes),
        )

        self.to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        batch_size = hidden_states.size(0)

        e1_start_mask = input_ids == self.e1_start_id
        e1_end_mask = input_ids == self.e1_end_id
        e2_start_mask = input_ids == self.e2_start_id
        e2_end_mask = input_ids == self.e2_end_id

        e1_start_pos = e1_start_mask.long().argmax(dim=1)
        e1_end_pos = e1_end_mask.long().argmax(dim=1)
        e2_start_pos = e2_start_mask.long().argmax(dim=1)
        e2_end_pos = e2_end_mask.long().argmax(dim=1)

        batch_indices = torch.arange(batch_size, device=self.device)

        e1_start_emb = hidden_states[batch_indices, e1_start_pos]
        e1_end_emb = hidden_states[batch_indices, e1_end_pos]
        e2_start_emb = hidden_states[batch_indices, e2_start_pos]
        e2_end_emb = hidden_states[batch_indices, e2_end_pos]

        entity_embed = torch.cat([e1_start_emb, e1_end_emb, e2_start_emb, e2_end_emb], dim=-1)
        h_e1e2 = self.mlp1(entity_embed)
        logits = self.mlp2(h_e1e2)

        return logits
