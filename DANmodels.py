import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from sentiment_data import read_sentiment_examples


PAD_IDX = 0
UNK_IDX = 1


class SentimentDatasetDAN(Dataset):
    """Convert sentences to word IDs using the same indexer as the embeddings."""

    def __init__(self, infile, word_indexer):
        self.examples = read_sentiment_examples(infile)
        self.word_ids = []
        for example in self.examples:
            ids = []
            for word in example.words:
                idx = word_indexer.index_of(word)
                ids.append(idx if idx != -1 else UNK_IDX)
            # An empty sentence is represented by one UNK token.
            self.word_ids.append(torch.tensor(ids or [UNK_IDX], dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.word_ids[idx], self.examples[idx].label


def collate_dan_batch(batch):
    """Pad only to the longest sentence in this batch; labels remain integers."""
    word_ids, labels = zip(*batch)
    padded = pad_sequence(word_ids, batch_first=True, padding_value=PAD_IDX)
    return padded, torch.tensor(labels, dtype=torch.long)


class DAN(nn.Module):
    """Word embeddings -> masked average -> feedforward sentiment classifier."""

    def __init__(self, word_embeddings, hidden_size=100, dropout=0.3,
                 random_embeddings=False, frozen=False):
        super().__init__()
        embedding_dim = word_embeddings.get_embedding_length()

        if random_embeddings:
            self.embedding = nn.Embedding(
                len(word_embeddings.word_indexer), embedding_dim,
                padding_idx=PAD_IDX,
            )
        else:
            # Start with GloVe vectors; fine-tune them unless frozen=True.
            self.embedding = word_embeddings.get_initialized_embedding_layer(
                frozen=frozen
            )
            self.embedding.padding_idx = PAD_IDX

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, word_ids):
        # [batch, length] -> [batch, length, embedding_dim]
        vectors = self.embedding(word_ids)

        # Ignore PAD in both the sum and the denominator of the average.
        mask = (word_ids != PAD_IDX).unsqueeze(-1)
        summed = (vectors * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        averaged = summed / lengths

        # Raw scores (logits), to be used with CrossEntropyLoss.
        return self.classifier(averaged)
