from collections import Counter


PAD_TOKEN = 'PAD'
UNK_TOKEN = 'UNK'
END_OF_WORD = '</w>'


def merge_pair(symbols, pair):
    """Replace non-overlapping occurrences of a symbol pair from left to right."""
    merged = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


class BPETokenizer:
    def __init__(self, vocab_size=1000):
        if vocab_size < 3:
            raise ValueError('BPE vocab_size must leave room for PAD, UNK and symbols')
        self.target_vocab_size = vocab_size
        self.merges = []
        self.token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self._merge_ranks = {}
        self._cache = {}
        self._fitted = False

    def __len__(self):
        return len(self.token_to_id)

    def fit(self, sentences):
        self._fitted = False
        self.merges = []
        self._merge_ranks = {}
        self._cache = {}
        self.token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}

        word_counts = Counter(word.lower() for words in sentences
                              for word in words if word)
        if not word_counts:
            raise ValueError('Cannot train BPE on an empty corpus')

        characters = {char for word in word_counts for char in word}
        for symbol in sorted(characters | {END_OF_WORD}):
            self.token_to_id[symbol] = len(self.token_to_id)
        if len(self) > self.target_vocab_size:
            raise ValueError(
                f'BPE vocab_size={self.target_vocab_size} is too small: '
                f'at least {len(self)} entries are needed for base symbols, PAD and UNK'
            )

        # Count unique word types once, weighted by their corpus frequency.
        word_vocab = Counter({tuple(word) + (END_OF_WORD,): count
                              for word, count in word_counts.items()})
        while len(self) < self.target_vocab_size:
            pair_counts = Counter()
            for symbols, frequency in word_vocab.items():
                for pair in zip(symbols, symbols[1:]):
                    pair_counts[pair] += frequency
            if not pair_counts:
                break

            # Lexical tie-breaking makes the tokenizer independent of input order.
            best_pair = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
            self.merges.append(best_pair)
            new_symbol = ''.join(best_pair)
            if new_symbol not in self.token_to_id:
                self.token_to_id[new_symbol] = len(self.token_to_id)

            updated_vocab = Counter()
            for symbols, frequency in word_vocab.items():
                updated_vocab[merge_pair(symbols, best_pair)] += frequency
            word_vocab = updated_vocab

        for rank, pair in enumerate(self.merges):
            self._merge_ranks.setdefault(pair, rank)
        self._fitted = True
        return self

    def tokenize_word(self, word):
        if not self._fitted:
            raise RuntimeError('Call fit on training sentences before encoding')
        word = word.lower()
        if not word:
            return []
        if word not in self._cache:
            symbols = tuple(word) + (END_OF_WORD,)
            while len(symbols) > 1:
                candidates = [pair for pair in zip(symbols, symbols[1:])
                              if pair in self._merge_ranks]
                if not candidates:
                    break
                # The earliest learned applicable rule has highest priority.
                best_pair = min(candidates, key=self._merge_ranks.__getitem__)
                symbols = merge_pair(symbols, best_pair)
            self._cache[word] = symbols
        return list(self._cache[word])

    def encode_word(self, word):
        return [self.token_to_id.get(token, 1) for token in self.tokenize_word(word)]

    def encode_sentence(self, words):
        return [idx for word in words for idx in self.encode_word(word)]
