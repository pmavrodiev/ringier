from typing import Sequence
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import numpy as np
import pickle
import copy
import click

from model_classifier import Multiclass
from sklearn.model_selection import train_test_split

from loguru import logger



def train_classifier(model, X_train, y_train, X_test, y_test, n_epochs: int, batch_size: int):
    
    train_loss_hist = []
    test_loss_hist = []
    # because we have multiclass
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    batches_per_epoch = len(X_train) // batch_size

    # training loop
    rng = np.random.default_rng(seed=42)    
    for epoch in range(n_epochs):
        epoch_loss = []
        # shuffle data at the start of each epoch
        shuffle_idx = np.arange(len(X_train))
        rng.shuffle(shuffle_idx)    

        # X_train = X_train[shuffle_idx]
        # y_train = y_train[shuffle_idx]
        
        # set model in training mode and run through each batch
        model.train()
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=1) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:                
                # take a batch
                start = i * batch_size
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # store training loss
                epoch_loss.append(float(loss))
                bar.set_postfix(
                    loss=float(loss)                
                )
        # set model in evaluation mode and run through the test set
        model.eval()
        y_pred = model(X_test)        
        ce = loss_fn(y_pred, y_test)
        ce = float(ce)
        train_loss_hist.append(np.mean(epoch_loss))
        test_loss_hist.append(ce)
        logger.info(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}")
       
    # get the final model
    model_weights = copy.deepcopy(model.state_dict())

    return {'model_weights': model_weights,
            'test_loss_hist': test_loss_hist,
            'train_loss_hist': train_loss_hist}


@click.command()
@click.option('--embeddings', # default='MiniLM-L6-v2',
              required=True,
              help='Path to an input file with embeddings for the datasets in the challenge')
@click.option('--store_assets',
              required=True,
              help='Path to a file where to store the training history and weights of the trained classifier')
@click.option('--store_classifier', 
              required=True,
              help='Path to a file to store the .pt of the model')
def entry(embeddings, store_assets, store_classifier):
    # read in the input embeddings
    # d = pickle.load(open("input_embeddings_01/embeddings_paraphrase-mpnet-base-v2.pkl", "rb"))
    d = pickle.load(open(embeddings, "rb"))

    X = d['X_train']
    y = d['y_train']

    X_val = d['X_test']

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, train_size=0.8,
                                                      shuffle=True,
                                                      random_state=42)
    
    clf = Multiclass(input_dim = X_train.shape[1], hidden_layer_1=100, output_dim=y_train.shape[1])

    d = train_classifier(clf, X_train, y_train, X_test, y_test, n_epochs=100, batch_size=64)

    # pickle.dump(d, open("classifier_assets_all-MiniLM-L6-v2.pkl", "wb"))
    pickle.dump(d, open(store_assets, "wb"))
    
    clf_scripted = torch.jit.script(clf)
    # clf_scripted.save("classifier_all-MiniLM-L6-v2.pt")
    clf_scripted.save(store_classifier)


if __name__ == "__main__":
   entry()