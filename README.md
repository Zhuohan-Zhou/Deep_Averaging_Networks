# Sentiment classification: BOW, DAN and Subword DAN

Activate your Python environment and install the dependencies:

```bash
conda activate cse256-pa1
python -m pip install torch numpy scikit-learn matplotlib
```

Run these commands from the directory containing `main.py`:

```bash
# Provided bag-of-words baselines (two and three linear layers).
python main.py --model BOW

# Part 1a: DAN with trainable, GloVe-initialized embeddings.
python main.py --model DAN

# Use the provided 300-dimensional GloVe vectors instead of 50-dimensional ones.
python main.py --model DAN --embedding-dim 300

# Part 1b: same vocabulary and architecture, but random trainable embeddings.
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
model parameter count, and all epoch metrics. Model training time excludes BPE
learning and data preparation, so compare those times separately or add data
loading time to model training time for a combined preprocessing/training cost.
Plots are not overwritten by runs with a different vocabulary size or seed.

## Configuration

DAN defaults: 50-dimensional embeddings, hidden size 100, dropout 0.3, batch size
32, learning rate 0.001, 20 epochs, seed 42, and one CPU thread. BOW retains its
original defaults: batch size 16, learning rate 0.0001, and 100 epochs.

For example:

```bash
python main.py --model DAN --epochs 20 --lr 0.001 --batch-size 32 --hidden-size 100 --dropout 0.3 --seed 42
python main.py --help
```

Data paths are resolved relative to `main.py`, so the script can also be run
from another directory. Plot paths are relative to the current working directory.

## Verified run

With the default command `python main.py --model DAN`, Python 3.11 and PyTorch
2.14.1 on this Mac, the best dev accuracy was **79.82% at epoch 5**. The final
epoch (20) reached 78.6% dev accuracy. The training loop took 16.7 seconds,
excluding imports and data loading. These are results from one seed (42), not
an average across runs; other environments may produce different results.

Training accuracy continued to rise while dev loss increased after the early
epochs, indicating overfitting. The printed best score is an observed dev-set
result; the code does not restore the parameters from that epoch.

### Subword vocabulary comparison

All three runs below use the same default model settings (50d random embeddings,
hidden size 100, dropout 0.3, Adam at 0.001, batch size 32, 20 epochs, seed 42).
Times are measured on this Mac and are not averaged across repeated runs.

| Actual vocabulary | Best dev accuracy | Best epoch | BPE learning (s) | Model training (s) | Mean train subwords/sentence |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 70.76% | 8 | 15.25 | 3.83 | 42.02 |
| 1000 | 75.46% | 14 | 31.85 | 4.52 | 35.18 |
| 2000 | 76.26% | 10 | 63.56 | 4.93 | 29.91 |

The raw summary is in `results/subword_comparison.csv`; each run's JSON and plot
are in its `results/subword_vSIZE` folder. These results show higher best dev
accuracy and shorter input sequences with larger vocabularies, alongside larger
embedding tables and more expensive BPE learning. The 2000-symbol model's best
score remains below the word-level GloVe model's 79.82%. Later epochs can overfit,
so distinguish the best observed dev score from final-epoch accuracy. This is a
single-seed comparison and does not establish a universal ranking.

## Checks

```bash
python -m unittest -v test_dan test_bpe
```

The 15 checks cover unknown words, empty sentences, padding invariance, finite
outputs for all-padding input, gradient updates, embedding initialization, BPE
frequency weighting, deterministic merges, merge application, word boundaries,
vocabulary-size handling and tokenizer reuse without dev vocabulary leakage.

This implementation covers word-level DAN (Part 1) and BPE/Subword DAN (Part 2).
The skip-gram exercises (Part 3) and the written report are separate deliverables.
