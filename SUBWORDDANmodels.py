import torch
from torch import nn
from torch.utils.data import Dataset

from DANmodels import PAD_IDX, UNK_IDX


class SentimentDatasetSubwordDAN(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        # Encode once, not again during every training epoch.
        self.subword_ids = [
            torch.tensor(tokenizer.encode_sentence(example.words) or [UNK_IDX],
                         dtype=torch.long)
            for example in examples
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.subword_ids[idx], self.examples[idx].label

    def average_length(self):
        return sum(ids.numel() for ids in self.subword_ids) / len(self)


class SubwordDAN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=100, dropout=0.3):
        super().__init__()
        # No GloVe: these vectors correspond to BPE symbols, not full words.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, subword_ids):
        # Average over all real subwords in a sentence, excluding padding.
        vectors = self.embedding(subword_ids)
        mask = (subword_ids != PAD_IDX).unsqueeze(-1)
        summed = (vectors * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        averaged = summed / lengths
        return self.classifier(averaged)
