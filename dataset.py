import torch
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
from konlpy.tag import Okt
from collections import Counter
from tqdm.notebook import tqdm
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd


class MovieDataSet(Dataset):
    def __init__(self, labels, text, vocab=None):
        super().__init__()
        self.labels = labels
        self.text = text

        self.tokenizer = Okt().morphs

        # torch.load('vocab_obj.pth')
        if vocab == None:
            self.vocab = self.make_vocab()
        else:
            self.vocab = vocab

    def __getitem__(self, idx):
        encoded_text = self.collate_fn(self.text[idx])
        return self.labels[idx], encoded_text

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        return self.vocab.lookup_indices(self.tokenizer(batch))

    def yield_tokens(self, texts):
        tokens = []
        for text in texts:
            tokens.append(self.tokenizer(text))
        return tokens

    def make_vocab(self):
        torch_vocab = build_vocab_from_iterator(
            self.yield_tokens(self.text), min_freq=3, specials=['<unk>'])
        torch.save(torch_vocab, 'vocab_obj.pth')

        return torch_vocab


if __name__ == '__main__':
    train_data = pd.read_csv("data/ratings_train.csv").dropna()

    train_label = train_data["label"]
    train_text = train_data["text"]

    train_dataset = MovieDataSet(train_label, train_text)

    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, drop_last=True)

    a = iter(train_dataloader)

    print(next(a))
