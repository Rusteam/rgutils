# -*- coding: utf-8 -*-
"""
A module to handle NLP tasks
"""

import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def embed_documents(model, document_list, progress_step=50, print_stats=True):
    '''
    Creates document embedding using given model with `embed` method
    from flair.embeddings package
    ------
    Returns numpy embeddings for each document
    '''
    from flair.embeddings import Sentence
    doc_len = len(document_list)
    document_embeddings = np.zeros((doc_len, model.embedding_length))
    for i in range(0, doc_len, progress_step):
        end = min(doc_len, i + progress_step)
        if i >= end:
            break
        docs = document_list[i: end]
        sent = list(map(lambda x: Sentence(x), docs))
        model.embed(sent)
        document_embeddings[i: end] = \
                        np.array([s.embedding.detach().numpy() for s in sent])
        if print_stats:
            print('{} out of {} documents have been embedded'.format(end, doc_len))
    return document_embeddings


def segment_text_data(word_list, start_index, stop_index, word_len):
    '''
    Segments lists of words into different training sets
    By joining them into a string of specified word length
    ------
    Returns a list of segmented data
    '''
    train_data = []
    for begin in range(start_index, stop_index, word_len):
        end = min(begin + word_len, stop_index)
        one_exmp = ' '.join(word_list[begin: end])
        train_data.append(one_exmp)
    return train_data


def replace_chars(string, replace_chars, replacement=' '):
    '''
    Replace chars in string from replace_chars
    Replaces_chars should be like '\n|\d|\W'
    -----
    Returns updated string
    '''
    string = re.sub(replace_chars, replacement, string)
    string = re.sub('\s+', ' ', string)
    return string.strip()


def vectorize_text(*data, vectorizer, **kwargs):
    '''
    Vectorize input data in the text form on word level split by spaces
    Uses first data argument as the one for fitting, others are for transformation only
    Return fitted vectorizer and a list of transformed data
    '''
    vectorizer = vectorizer.lower()
    assert vectorizer in ['tfidf','bow']
    if vectorizer == 'tfidf':
        print('Using Tf-Idf with', kwargs)
        vectorizer = TfidfVectorizer(**kwargs)
    elif vectorizer == 'bow':
        print('Using bag-of-words with', kwargs)
        vectorizer = CountVectorizer(**kwargs)
    vectorizer.fit(data[0])
    transformed = []
    for d in data:
        t = vectorizer.transform(d).toarray()
        transformed.append(t)
    print("Vocab size:", len(vectorizer.vocabulary_))
    return vectorizer, transformed
