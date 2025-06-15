#!/usr/bin/env python3
"""
Semantic search and question answering
"""

import os
from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering


def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity value
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def semantic_search(corpus_path, sentence):
    """
    Perform semantic search on a corpus of documents.

    Args:
        corpus_path: Path to the corpus of reference documents
        sentence: The sentence to search with

    Returns:
        The reference text of the document most similar to the input sentence
    """
    # Load a pre-trained Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 1: Read the corpus documents
    documents = []
    file_names = os.listdir(corpus_path)
    for file_name in file_names:
        if file_name.endswith('.md'):
            file_path = os.path.join(corpus_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    # Generate embeddings for the corpus documents
    doc_embeddings = model.encode(documents)

    # Generate an embedding for the input sentence
    query_embedding = model.encode([sentence])[0]

    # Compute cosine similarities between the query and each document
    similarities = [cosine_similarity(query_embedding, doc_embedding)
                    for doc_embedding in doc_embeddings]

    # Find index of document with highest similarity score
    best_doc_index = np.argmax(similarities)

    # Return the most similar document
    return documents[best_doc_index]


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    :param question: str, containing the question to answer
    :param reference: str, containing the reference document from which to
    find the answer
    :return: str, containing the answer or None if no answer is
    found
    """
    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    # Load the BERT model for question answering
    model = TFBertForQuestionAnswering.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    # Tokenize the input question and reference
    inputs = tokenizer.encode_plus(
        question, reference, add_special_tokens=True, return_tensors="tf")

    # Get the input IDs, attention mask, and token type IDs
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    # Make predictions using the BERT model
    outputs = model(
        input_ids,
        attention_mask=attention_mask, token_type_ids=token_type_ids)
    start_logits, end_logits = outputs.start_logits, outputs.end_logits

    # Find the start and end positions of the answer
    start_index = tf.argmax(start_logits, axis=1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=1).numpy()[0]

    # Extract the answer from the reference document
    if start_index <= end_index:
        answer_tokens = input_ids[0, start_index:end_index + 1]
        answer = tokenizer.decode(answer_tokens)
    else:
        answer = None

    return answer


def question_answer_loop(corpus_path):
    """
    Continuously prompts the user for questions and answers
    them using the reference texts.

    :param corpus_path: str, containing the path to the
    corpus of reference documents
    """
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        question = input('Q: ')
        if question.lower() in exit_commands:
            print('A: Goodbye')
            break
        reference = semantic_search(
            corpus_path, question)
        answer = question_answer(question, reference)
        if answer:
            print(f'A: {answer}')
        else:
            print('A: Sorry, I do not understand your question.')
