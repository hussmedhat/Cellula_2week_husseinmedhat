import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BertBiLSTM(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=6, freeze_bert_layers=8):
        super().__init__()
        # Initialize the base BERT encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze the initial embeddings and specified lower layers to retain pre-trained knowledge
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:freeze_bert_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # Custom classification head: Bidirectional LSTM -> Attention -> Fully Connected[cite: 3]
        self.lstm = nn.LSTM(768, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        # Create an attention mask so BERT ignores padding tokens (0)[cite: 3]
        attention_mask = (x != 0).long()
        bert_out = self.bert(input_ids=x, attention_mask=attention_mask).last_hidden_state

        # Pack the sequences for efficient LSTM processing[cite: 3]
        packed = pack_padded_sequence(bert_out, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Apply custom attention mechanism over the LSTM outputs[cite: 3]
        mask = (x[:, :out.size(1)] != 0)
        attn_scores = self.attn(out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        
        # Calculate the final weighted context vector and pass through dropout to the linear layer[cite: 3]
        final = (out * attn_weights).sum(dim=1)
        return self.fc(self.dropout(final))