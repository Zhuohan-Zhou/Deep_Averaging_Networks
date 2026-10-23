# CSE256 Project 1: Deep Averaging Networks | Byte Pair Encoding | Skip-Gram

Install the dependencies:
```bash
python -m pip install torch numpy scikit-learn matplotlib
```

Run these commands from the directory containing `main.py`:
```bash
# Part 1a: DAN with trainable, GloVe-initialized embeddings.
python main.py --model DAN

# Use the provided 300-dimensional GloVe vectors instead of 50-dimensional ones.
python main.py --model DAN --embedding-dim 300

# Part 1b: random trainable embeddings.
python main.py --model DAN --random-embeddings

# Part 2: learn BPE from train.txt, then train a DAN over its subwords.
python main.py --model SUBWORDDAN
```

## Implementation

`DANmodels.py` defines the dataset, batch padding function, and model. Sentences
are converted into word IDs using the embedding indexer. Missing words use UNK
(index 1). Batches use PAD (index 0) to align sentence lengths. The model masks
PAD positions before averaging, so padding does not change the sentence vector.

The averaged vector passes through a linear layer, ReLU, dropout, and a final
linear layer with two outputs (negative/positive). These raw logits are trained
with cross-entropy loss. Embeddings are fine-tuned by default; use
`--freeze-embeddings` to keep the GloVe vectors fixed.

`main.py` keeps token IDs as integer tensors, trains with Adam, and evaluates on
the development set without computing gradients. Each DAN epoch prints train/dev
loss and accuracy. The final summary reports the best observed dev accuracy and
its epoch; it does not save or restore a model checkpoint. The accuracy plot is
written to `dan_accuracy.png` in the current working directory. Each DAN run also
saves epoch-by-epoch losses, accuracies, configuration, and timing in
`dan_metrics.json`. Use `--output-dir` to keep separate runs in separate folders.

## Part 2: BPE and Subword DAN

`bpe.py` implements word-boundary-aware character BPE from scratch. It starts
with the characters observed in the training text and the end-of-word marker
`</w>`. Each iteration counts adjacent symbol pairs, weighted by word frequency,
and merges the most frequent pair. Ties are broken lexically for reproducibility.
Pair replacements are non-overlapping and never cross word boundaries.

The target `--vocab-size` counts PAD, UNK, base symbols, and unique symbols created
by merges. Base symbols and intermediate merged symbols are retained. If the
requested size cannot hold the base symbols, the program reports an error. If
all words have become single symbols, BPE stops even if the target is not reached;
the program reports both requested and actual sizes.

At encoding time, the earliest learned applicable merge has priority. Repeated
words are cached. A standalone `</w>` is retained as a real token when it has not
merged with a preceding symbol; a merged ending such as `ing</w>` is one token.
Characters not seen in training map to UNK. The training and dev sets share the
same tokenizer: **only training text is used to learn the vocabulary and merges**.

`SUBWORDDANmodels.py` encodes each sentence once, concatenates its subword IDs,
and uses the existing batch padding function. Subword embeddings are always
randomly initialized and trainable; GloVe is not loaded. The model averages all
non-PAD subword embeddings in the sentence, then uses the same classifier
architecture as the word-level DAN. The loss is cross-entropy on raw logits.

Run vocabulary-size comparisons with all other settings fixed:

```bash
python main.py --model SUBWORDDAN --vocab-size 500 --output-dir results/subword_v500
python main.py --model SUBWORDDAN --vocab-size 1000 --output-dir results/subword_v1000
python main.py --model SUBWORDDAN --vocab-size 2000 --output-dir results/subword_v2000
```

The default vocabulary size is 1000. Other defaults match the word-level DAN:
50-dimensional embeddings, hidden size 100, dropout 0.3, batch size 32, learning
rate 0.001, 20 epochs and seed 42. `--embedding-dim 300` also works for subwords;
`--freeze-embeddings` is rejected because Part 2 trains embeddings from scratch.

Each subword run writes a vocabulary/seed-specific accuracy PNG and metrics JSON,
for example `subworddan_v1000_seed42_metrics.json`. The JSON records BPE learning
time, encoding time, average train/dev subword lengths, actual vocabulary size,
model parameter count, and all epoch metrics.

## Configuration

DAN defaults: 50-dimensional embeddings, hidden size 100, dropout 0.3, batch size
32, learning rate 0.001, 20 epochs, seed 42, and one CPU thread. BOW retains its
original defaults: batch size 16, learning rate 0.0001, and 100 epochs.

For example:

```bash
python main.py --model DAN --epochs 20 --lr 0.001 --batch-size 32 --hidden-size 100 --dropout 0.3 --seed 42
python main.py --help
```