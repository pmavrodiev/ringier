# Usage

`python main.py --help`

# Example Usage

- `python main.py  --train ../data/train_data_2021.json --test ../data/predict_payload.json --taxo ../data/taxonomy_mappings_2021.json`

  will use the default *all-MiniLM-L6-v2* language model from the SentenceTransformers framework to compute embeddings of all training and test documents


- `python main.py  --train ../data/train_data_2021.json --test ../data/predict_payload.json --taxo ../data/taxonomy_mappings_2021.json --model paraphrase-mpnet-base-v2`

  will use the bigger *paraphrase-mpnet-base-v2* model that can embedd texts of  300-400 words. It takes about 1.5 hours on an average machine to encode all documents, though.



