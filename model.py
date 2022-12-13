import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import MovieDataSet


class SentimentLSTM(nn.Module):
    def __init__(self, no_layers, vocab_size, output_dim, hidden_dim, embedding_dim, device, drop_prob=0.5):
        super().__init__()

        self.device = device
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_dim=self.hidden_dim,
                            num_layers=self.num_layers, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sg = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.no_layers, batch_size,
                         self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.no_layers, batch_size,
                         self.hidden_dim)).to(self.device)
        hidden = (h0, c0)
        return hidden
