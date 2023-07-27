import numpy as np
import tqdm
import pickle
import click

from typing import Sequence
from loguru import logger

try:
    from .corpus import Document, TextCorpus
    from .model_embeddings import Model
except ImportError:
    from corpus import Document, TextCorpus
    from model_embeddings import Model
    
def encode_batches(documents: Sequence[Document], batch_size: int, model, pool):
    encoded_batches = []
    n_batches = len(documents) // batch_size + 1

    for n in tqdm.trange(n_batches):        
        start = n*batch_size
        end = start + batch_size
        to_encode = [x.get_text() for x in documents[start:end]]
        if to_encode:
            encoded_batches.append(np.array(model.encode(to_encode, pool)))
            
    return np.concatenate(encoded_batches, axis=0)

def encode_train_test_documents(corpus: TextCorpus, batch_size: int,  output_f: str, model, pool):
    logger.info("Computing training embeddings")    
    X_train = encode_batches(corpus.train_documents, batch_size, model, pool)
    y_train = np.array([x.labels for x in corpus.train_documents])

    logger.info("Computing test embeddings")    
    X_test = encode_batches(corpus.predict_documents, batch_size, model, pool)

    logger.info("Persisting to disk")
    d = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test}
    pickle.dump(d, open(output_f, "wb"))


@click.command()
@click.option('--train', # default='train_data_2021.json',
              required=True,
              help='Path to a json corpus with training documents')
@click.option('--test', # default='predict_payload.json',
              required=True,
              help='Path to a json corpus with test (predict) documents')
@click.option('--taxo', # default='taxonomy_mappings_2021.json',
              required=True,
              help='Path to a json document with taxonomy mappings')
@click.option('--model', default='all-MiniLM-L6-v2',
              help='Name of any pre-trained model from the SentenceTransformers framework')
@click.option('--batch_size', default=200,
              help='A batch of batch_size documents will be encoded by the chosen model at a time')

def entry(train: str, test: str, taxo: str, model: str, batch_size: int):
    tc = TextCorpus(train_data=train,
                    predict_data=test,
                    taxo_map=taxo)
    
    # choose paraphrase-mpnet-base-v2 for a deeper and better one - takes about 1.5 hours    
    m = Model(model)

    pool = m.model.start_multi_process_pool()

    encode_train_test_documents(corpus=tc, batch_size=batch_size,
                                output_f=f"embeddings_{model}.pkl",
                                model=m, pool=pool)

    m.model.stop_multi_process_pool(pool)




if __name__ == "__main__":
    entry()
