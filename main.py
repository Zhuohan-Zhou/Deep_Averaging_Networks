# models.py

import torch
from torch import nn
from sentiment_data import read_word_embeddings
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import DAN, SentimentDatasetDAN, collate_dan_batch


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item() * len(y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / size
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
@torch.no_grad()
def eval_epoch(data_loader, model, loss_fn, optimizer=None):
    size = len(data_loader.dataset)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item() * len(y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / size
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader, epochs=100, lr=0.0001,
               loss_fn=None, log_every=10):
    if loss_fn is None:
        loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_train_accuracy = []
    all_test_accuracy = []
    best_dev_accuracy = -1.0
    best_epoch = 0
    start_time = time.time()
    for epoch in range(epochs):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if test_accuracy > best_dev_accuracy:
            best_dev_accuracy = test_accuracy
            best_epoch = epoch + 1

        if (epoch + 1) % log_every == 0 or epoch + 1 == epochs:
            print(f'Epoch #{epoch + 1}: train loss {train_loss:.4f}, '
                  f'train accuracy {train_accuracy:.3f}, dev loss {test_loss:.4f}, '
                  f'dev accuracy {test_accuracy:.3f}', flush=True)

    print(f'Best dev accuracy: {best_dev_accuracy:.4f} at epoch {best_epoch}; '
          f'training time: {time.time() - start_time:.1f}s')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', choices=['BOW', 'DAN'], required=True)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Default: 100 for BOW, 20 for DAN')
    parser.add_argument('--lr', type=float, default=None,
                        help='Default: 0.0001 for BOW, 0.001 for DAN')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Default: 16 for BOW, 32 for DAN')
    parser.add_argument('--hidden-size', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='DAN hidden-layer dropout probability')
    parser.add_argument('--embedding-dim', type=int, choices=[50, 300], default=50)
    embedding_options = parser.add_mutually_exclusive_group()
    embedding_options.add_argument('--random-embeddings', action='store_true',
                                   help='DAN: train random embeddings instead of GloVe')
    embedding_options.add_argument('--freeze-embeddings', action='store_true',
                                   help='DAN: keep GloVe embeddings fixed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1,
                        help='CPU threads; one is efficient for this small model')

    # Parse the command-line arguments
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.lr is not None and args.lr <= 0:
        parser.error('--lr must be positive')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.hidden_size < 1 or args.threads < 1:
        parser.error('--hidden-size and --threads must be positive')
    if not 0 <= args.dropout < 1:
        parser.error('--dropout must be between 0 (inclusive) and 1 (exclusive)')

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    data_dir = Path(__file__).resolve().parent / 'data'
    epochs = args.epochs if args.epochs is not None else (100 if args.model == 'BOW' else 20)
    lr = args.lr if args.lr is not None else (0.0001 if args.model == 'BOW' else 0.001)
    batch_size = args.batch_size if args.batch_size is not None else (16 if args.model == 'BOW' else 32)

    # Load dataset
    start_time = time.time()

    collate_fn = None
    if args.model == 'BOW':
        train_data = SentimentDatasetBOW(data_dir / 'train.txt')
        dev_data = SentimentDatasetBOW(data_dir / 'dev.txt', vectorizer=train_data.vectorizer, train=False)
    else:
        word_embeddings = read_word_embeddings(
            str(data_dir / f'glove.6B.{args.embedding_dim}d-relativized.txt')
        )
        train_data = SentimentDatasetDAN(data_dir / 'train.txt', word_embeddings.word_indexer)
        dev_data = SentimentDatasetDAN(data_dir / 'dev.txt', word_embeddings.word_indexer)
        collate_fn = collate_dan_batch

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(
            NN2BOW(input_size=512, hidden_size=args.hidden_size), train_loader,
            test_loader, epochs=epochs, lr=lr)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(
            NN3BOW(input_size=512, hidden_size=args.hidden_size), train_loader,
            test_loader, epochs=epochs, lr=lr)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        plt.close()
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        plt.close()
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        model = DAN(word_embeddings, hidden_size=args.hidden_size,
                    dropout=args.dropout, random_embeddings=args.random_embeddings,
                    frozen=args.freeze_embeddings)
        initialization = 'random' if args.random_embeddings else 'GloVe'
        print(f'\nDAN: {initialization}, {args.embedding_dim}d embeddings, '
              f'hidden size {args.hidden_size}, seed {args.seed}', flush=True)
        train_accuracy, dev_accuracy = experiment(
            model, train_loader, test_loader, epochs=epochs, lr=lr,
            loss_fn=nn.CrossEntropyLoss(), log_every=1,
        )
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epochs + 1), train_accuracy, label='Train')
        plt.plot(range(1, epochs + 1), dev_accuracy, label='Dev')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('DAN Training and Dev Accuracy')
        plt.legend()
        plt.grid()
        plt.savefig('dan_accuracy.png')
        plt.close()
        print('Accuracy plot saved as dan_accuracy.png')

if __name__ == "__main__":
    main()
