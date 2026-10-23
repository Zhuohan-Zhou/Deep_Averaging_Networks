"""Small correctness checks; no training data or GloVe file is required."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from DANmodels import DAN, SentimentDatasetDAN, collate_dan_batch
from sentiment_data import WordEmbeddings
from utils import Indexer


class DANTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.indexer = Indexer()
        for word in ['PAD', 'UNK', 'good', 'movie']:
            self.indexer.add_and_get_index(word)
        self.embeddings = WordEmbeddings(
            self.indexer,
            np.array([[0., 0., 0.], [0., 0., 0.],
                      [1., 2., 3.], [3., 2., 1.]], dtype=np.float32),
        )

    def test_unknown_words_empty_sentence_and_collation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'examples.txt'
            path.write_text('1\tGOOD movie\n0\tunseen\n0\t\n')
            data = SentimentDatasetDAN(path, self.indexer)
        ids, labels = collate_dan_batch([data[i] for i in range(len(data))])
        self.assertEqual(ids.tolist(), [[2, 3], [1, 0], [1, 0]])
        self.assertEqual(labels.tolist(), [1, 0, 0])
        self.assertEqual(ids.dtype, torch.long)
        self.assertEqual(labels.dtype, torch.long)

    def test_padding_does_not_change_predictions(self):
        model = DAN(self.embeddings, dropout=0).eval()
        short = model(torch.tensor([[2, 3]]))
        padded = model(torch.tensor([[2, 3, 0, 0]]))
        batched = model(torch.tensor([[2, 3, 0], [2, 2, 3]]))
        torch.testing.assert_close(short, padded)
        torch.testing.assert_close(short[0], batched[0])
        # Even a nonzero PAD vector must be excluded from the average.
        with torch.no_grad():
            model.embedding.weight[0].fill_(100)
        torch.testing.assert_close(short, model(torch.tensor([[2, 3, 0]])))
        all_pad = model(torch.zeros((1, 3), dtype=torch.long))
        self.assertEqual(all_pad.shape, (1, 2))
        self.assertTrue(torch.isfinite(all_pad).all())

    def test_loss_updates_embeddings_and_classifier_but_not_pad(self):
        model = DAN(self.embeddings, dropout=0)
        embedding_before = model.embedding.weight.detach().clone()
        classifier_before = model.classifier[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        logits = model(torch.tensor([[2, 3, 0], [2, 0, 0]]))
        loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([1, 0]))
        loss.backward()
        torch.testing.assert_close(model.embedding.weight.grad[0], torch.zeros(3))
        self.assertGreater(model.embedding.weight.grad[2:].abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(embedding_before[2:], model.embedding.weight[2:]))
        self.assertFalse(torch.equal(classifier_before, model.classifier[0].weight))
        torch.testing.assert_close(embedding_before[0], model.embedding.weight[0])

    def test_random_and_frozen_embeddings(self):
        frozen = DAN(self.embeddings, frozen=True)
        self.assertFalse(frozen.embedding.weight.requires_grad)
        random = DAN(self.embeddings, random_embeddings=True)
        self.assertTrue(random.embedding.weight.requires_grad)
        torch.testing.assert_close(random.embedding.weight[0], torch.zeros(3))
        self.assertFalse(torch.equal(random.embedding.weight[2:], frozen.embedding.weight[2:]))


if __name__ == '__main__':
    unittest.main()
