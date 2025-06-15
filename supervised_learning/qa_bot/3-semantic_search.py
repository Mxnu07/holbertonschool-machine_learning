#!/usr/bin/env python3
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, TFBertModel


def load_corpus(corpus_path):
    """
    Loads the corpus of documents from the specified path.

    :param corpus_path: str, path to the corpus of reference documents
    :return: list of str, containing the reference documents
    """
    documents = []
    for filename in os.listdir(corpus_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(
                    corpus_path, filename),
                    'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents


def encode_texts(texts, tokenizer, model):
    """
    Encodes a list of texts into embeddings using the BERT model.

    :param texts: list of str, containing the texts to encode
    :param tokenizer: BertTokenizer, tokenizer for the BERT model
    :param model: TFBertModel, pre-trained BERT model
    :return: np.ndarray, containing the embeddings of the texts
    """
    if not texts:
        return np.array([])

    inputs = tokenizer(
        texts, return_tensors='tf',
        padding=True, truncation=True)
    outputs = model(inputs['input_ids'])
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    :param corpus_path: str, path to the corpus of reference
    documents
    :param sentence: str, sentence from which to perform
    semantic search
    :return: str, containing the reference text of the
    document most similar to the sentence
    """
    # Load the corpus of documents
    documents = load_corpus(corpus_path)

    # Check if documents are loaded
    if not documents:
        return "No documents found in the corpus."

    # Load the pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')

    # Encode the documents and the query sentence
    document_embeddings = encode_texts(documents, tokenizer, model)
    query_embedding = encode_texts([sentence], tokenizer, model)

    # Check if embeddings are generated
    if document_embeddings.size == 0 or query_embedding.size == 0:
        return "Error in generating embeddings."

    # Compute cosine similarity between the
    # query embedding and each document embedding
    similarities = cosine_similarity(
        query_embedding, document_embeddings)

    # Find the index of the most similar document
    most_similar_index = np.argmax(similarities)

    # Return the most similar document
    return documents[
        most_similar_index]
