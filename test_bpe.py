import unittest

import torch

from bpe import BPETokenizer, END_OF_WORD, merge_pair
from DANmodels import collate_dan_batch
from sentiment_data import SentimentExample
from SUBWORDDANmodels import SentimentDatasetSubwordDAN, SubwordDAN


class BPETests(unittest.TestCase):
    def test_merge_uses_non_overlapping_symbols(self):
        self.assertEqual(merge_pair(('a', 'a', 'a'), ('a', 'a')), ('aa', 'a'))
        self.assertEqual(merge_pair(('a', 'a', 'a', 'a'), ('a', 'a')), ('aa', 'aa'))
        self.assertEqual(merge_pair(('ab', 'c'), ('b', 'c')), ('ab', 'c'))

    def test_pair_frequencies_are_weighted_by_word_frequency(self):
        tokenizer = BPETokenizer(8).fit([['xy']] * 8 + [['ab']])
        self.assertEqual(tokenizer.merges, [('x', 'y')])
        self.assertEqual(len(tokenizer), 8)

    def test_ties_are_deterministic_and_ignore_corpus_order(self):
        first = BPETokenizer(8).fit([['ab', 'cd']])
        second = BPETokenizer(8).fit([['cd', 'ab']])
        self.assertEqual(first.merges, [('a', 'b')])
        self.assertEqual(first.merges, second.merges)
        self.assertEqual(first.token_to_id, second.token_to_id)

    def test_known_reference_first_merge(self):
        corpus = [['low']] * 5 + [['lower']] * 2
        corpus += [['newest']] * 6 + [['widest']] * 3
        tokenizer = BPETokenizer(14).fit(corpus)
        self.assertEqual(tokenizer.merges, [('e', 's')])
        self.assertEqual(tokenizer.tokenize_word('newest'),
                         ['n', 'e', 'w', 'es', 't', END_OF_WORD])

    def test_encoding_matches_replaying_learned_rules_and_preserves_text(self):
        corpus = [['low', 'lower', 'lowest', 'newer', 'wider']] * 3
        tokenizer = BPETokenizer(30).fit(corpus)
        for word in ['low', 'lower', 'lowest', 'newer', 'wider', 'newest', 'slow']:
            reference = tuple(word) + (END_OF_WORD,)
            for pair in tokenizer.merges:
                reference = merge_pair(reference, pair)
            tokens = tokenizer.tokenize_word(word)
            self.assertEqual(tokens, list(reference))
            self.assertEqual(''.join(tokens).replace(END_OF_WORD, ''), word)
        self.assertEqual(tokenizer.tokenize_word('LOW'), tokenizer.tokenize_word('low'))

    def test_word_boundaries_and_unknown_characters_do_not_change_vocabulary(self):
        tokenizer = BPETokenizer(30).fit([['ab', 'cd']])
        vocab_before = dict(tokenizer.token_to_id)
        merges_before = list(tokenizer.merges)
        self.assertEqual(tokenizer.encode_sentence(['ab', 'cd']),
                         tokenizer.encode_word('ab') + tokenizer.encode_word('cd'))
        self.assertNotIn('abcd', tokenizer.token_to_id)
        self.assertIn(1, tokenizer.encode_word('unseen\N{UNICORN FACE}'))
        self.assertNotIn(0, tokenizer.encode_word('unseen'))
        self.assertEqual(tokenizer.token_to_id, vocab_before)
        self.assertEqual(tokenizer.merges, merges_before)

    def test_small_corpora_stop_before_unreachable_target_and_refit_clears_cache(self):
        tokenizer = BPETokenizer(100).fit([['abc']])
        self.assertLess(len(tokenizer), 100)
        before = tokenizer.tokenize_word('abc')
        tokenizer.fit([['xy']])
        self.assertNotEqual(tokenizer.tokenize_word('abc'), before)
        self.assertEqual(tokenizer.encode_sentence([]), [])
        self.assertEqual(tokenizer.encode_word(''), [])

    def test_invalid_training_inputs(self):
        with self.assertRaises(RuntimeError):
            BPETokenizer().encode_word('word')
        with self.assertRaises(ValueError):
            BPETokenizer(2)
        with self.assertRaises(ValueError):
            BPETokenizer().fit([])
        with self.assertRaises(ValueError):
            BPETokenizer(5).fit([['abc']])


class SubwordDANTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = BPETokenizer(30).fit([['good', 'movie'], ['bad']])

    def test_dataset_and_batch_use_subword_ids_with_integer_labels(self):
        examples = [SentimentExample(['good', 'movie'], 1), SentimentExample([], 0)]
        dataset = SentimentDatasetSubwordDAN(examples, self.tokenizer)
        ids, labels = collate_dan_batch([dataset[0], dataset[1]])
        expected = self.tokenizer.encode_sentence(['good', 'movie'])
        self.assertEqual(ids[0].tolist(), expected)
        self.assertEqual(ids[1, 0].item(), 1)
        self.assertTrue((ids[1, 1:] == 0).all())
        self.assertEqual(ids.dtype, torch.long)
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(dataset.average_length(), (len(expected) + 1) / 2)

    def test_padding_invariance_and_finite_all_pad_output(self):
        model = SubwordDAN(len(self.tokenizer), dropout=0).eval()
        ids = torch.tensor([self.tokenizer.encode_word('good')])
        padded = torch.cat([ids, torch.zeros((1, 4), dtype=torch.long)], dim=1)
        torch.testing.assert_close(model(ids), model(padded))
        self.assertEqual(model(ids).shape, (1, 2))
        all_pad_output = model(torch.zeros((2, 3), dtype=torch.long))
        self.assertTrue(torch.isfinite(all_pad_output).all())

    def test_random_embeddings_are_trainable_and_pad_stays_zero(self):
        model = SubwordDAN(len(self.tokenizer), dropout=0)
        before = model.embedding.weight.detach().clone()
        ids, labels = collate_dan_batch([
            (torch.tensor(self.tokenizer.encode_word('good')), 1),
            (torch.tensor(self.tokenizer.encode_sentence(['bad', 'movie'])), 0),
        ])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(model(ids), labels)
        loss.backward()
        self.assertGreater(model.embedding.weight.grad[1:].abs().sum().item(), 0)
        torch.testing.assert_close(model.embedding.weight.grad[0], torch.zeros(50))
        optimizer.step()
        self.assertFalse(torch.equal(before[1:], model.embedding.weight[1:]))
        torch.testing.assert_close(model.embedding.weight[0], torch.zeros(50))


if __name__ == '__main__':
    unittest.main()
