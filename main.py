from typing import Sequence
import torch
import torch.nn as nn
import torch.optim as optim

import json
import click

import numpy as np

import json


from input_embeddings_01 import  Document, TextCorpus, Model, encode_batches


from loguru import logger

def logits_to_labels(y, category_mappings, top_N: int=3):
    top_N_idx = np.apply_along_axis(lambda i: (-i).argsort()[:top_N], axis=1, arr=y)
    top_N_vals = np.apply_along_axis(lambda i: i[(-i).argsort()[:top_N]], axis=1, arr=y)
    
    labels = []

    for doc_top_indeces, doc_top_confidences in zip(top_N_idx, top_N_vals):
        doc_top_N_names = [category_mappings[x] for x in doc_top_indeces ] 
        doc_top_N_vals =  [float(i) for i in doc_top_confidences ] 

        labels.append([[i, j] for i, j in zip(doc_top_N_names, doc_top_N_vals)])

    return labels

def generate_prediction_output(predict_documents, category_mapping, embedding_model, clf):

    # first compute embeddings with the language model that was used to train our classifier
    m = Model(embedding_model)   

    pool = m.model.start_multi_process_pool()

    doc_embeddings = encode_batches(predict_documents, batch_size=10, model=m, pool=pool)    

    m.model.stop_multi_process_pool(pool)

    # now use these embeddings as input to the classifier
    X_t = torch.tensor(doc_embeddings, dtype=torch.float32)

    y_raw = clf(X_t)
    # rectify the negative entries, which point strongly against belonging to the corresponding class
    y = nn.ReLU()(y_raw).detach().numpy()

    # use the taxonomy mapping to generate readable category labels together with confidence
    y_labels = logits_to_labels(y, category_mapping)
    return y_labels


@click.command()
@click.option('--predict_data', # default='MiniLM-L6-v2',
              required=True,
              help='Path to an input file with documents to classify')
@click.option('--taxonomy',
              required=True,
              help='Path to a file with the classification taxonomy')
@click.option('--classifier', 
              required=True,
              help='Path to a trained classifier')
@click.option('--lang_model', 
              required=True,
              help='Name of the language model to encode new documents. Must be the same as that used to train the classifier')

def entry(predict_data, taxonomy, classifier, lang_model):
    # load data
    tc = TextCorpus(train_data=None,  # optional this time
                    predict_data=predict_data,
                    taxo_map=taxonomy)
    
    
    # load trained classifier on the paraphrase-mpnet-base-v2 embeddings    
    # clf = torch.jit.load("classifier_02/classifier_paraphrase-mpnet-base-v2.pt")
    clf = torch.jit.load(classifier)
    clf.eval()    

    predicted_labels = generate_prediction_output(tc.predict_documents,
                                                  tc.taxo.taxo_mappings,
                                                  lang_model,
                                                  # "paraphrase-mpnet-base-v2",
                                                  clf)


    # an array of size equal to the size of the input
    # each element of the array is in turn an array with the top 3
    # predictions from the classifier
    json.dump(predicted_labels, open("probas.json", "w"))
    



if __name__ == "__main__":

    entry()