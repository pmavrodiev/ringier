# TLDR;

## [https://pavlin.streamlit.app](https://pavlin.streamlit.app/)


![image](https://github.com/pmavrodiev/ringier/assets/1107931/cf03066c-77e1-48bb-8b07-08277a88867d)



# Submission

**Full discplaimer** - On my previous (unsuccessful) application to Ringier several years ago, I already received a very similar version of this challenge. Save for some minor simplifications, most notably in the hierarchical labels of the training documents, the present challenge is identical.
My previous submission can be found [here](https://colab.research.google.com/drive/1cD93kx3nNn_bFBZlIUAoNDF_-0S6EGnt?usp=sharing#scrollTo=LaxaUvAN2wn7). Nevertheless I have used an entirely different approach to keep it fair.

The high-level idea to tackle tasks of this sort is common enough. It consist of two stages. First, the Encoder part of a Transformer-based LLM can be used to compute the embeddings of the training documents. 
In a process called *encoding* we run the input texts through the Encoder part of an LLM to obtain their embeddings or compressed numerical representation.

Second, these embeddings can then be used in a multi-class classification set-up with any classification model desired.

Once this two-stage training process is finished, we are left with a trained classifier than can be used for inference.

During inference input texts are encoded with the LLM used during training and the trained classifier makes an inference on these embeddings.

I have implemented these two steps, separately, so that they can be run independently of one another. Conretely:

1. `input_embeddings_01/` - uses the SentenceTransformers framework to obtain pre-trained LLMs. I have currently used two models (*all-MiniLM-L6.v2* and *paraphrase-mpnet-base-v2*), that ultimately give similar results. However, other models can be supplied from the command line. The output of this stage are embeddings vectors for the training and predict datasets from the chosen language model stored in pickle files. The pickle files for the two LLMs mentioned have already been computed. Please refer to  `input_embeddings_01/README.md` for usage examples.
   
2. `classifier_02/` - builds a shallow neural network multi-class classifier with inputs the embedding vectors from the previous stage. The output is a trained classifier in the form of a serialized PyTorch model. Two serialized classifiers for each of the LLMs from the previous stage have already been saved. Please refer to `classifier_02/README.md`  for example usage on how to run this stage separately


# Example Usage

Putting it all together we can generate the required `probas.json` by running:

```
python main.py --predict_data data/predict_payload.json --taxonomy data/taxonomy_mappings_2021.json --classifier classifier_02/classifier_all-MiniLM-L6-v2.pt --lang_model all-MiniLM-L6-v2
```

This command will only read the predict payload and use its embeddings from the Mini LLM to classify the texts accoring to the provided taxonomy file.
