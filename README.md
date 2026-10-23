# Sentiment classification: BOW and DAN

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

`main.py` keeps word IDs as integer tensors, trains with Adam, and evaluates on
the development set without computing gradients. Each DAN epoch prints train/dev
loss and accuracy. The final summary reports the best observed dev accuracy and
its epoch; it does not save or restore a model checkpoint. The accuracy plot is
written to `dan_accuracy.png` in the current working directory.

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

## Checks

```bash
python -m unittest -v test_dan
```

The checks cover unknown words, empty sentences, padding invariance, finite
outputs for all-padding input, gradient updates, and embedding initialization.

This implementation covers word-level DAN (Part 1). BPE/subword DAN and the
skip-gram exercises are separate assignment parts.
