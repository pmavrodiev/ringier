import torch

from sentence_transformers import SentenceTransformer
from loguru import logger

class Model:
    def __init__(self, model: str):
        logger.info(f"Initializing model '{model}'")
        self.model = SentenceTransformer(model)
    def encode(self, sentences: list, pool):
        with torch.no_grad():
            # pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(sentences, pool)
            # self.model.stop_multi_process_pool(pool)
        return embeddings
    
