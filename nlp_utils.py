# -*- coding: utf-8 -*-
"""
A module to handle NLP tasks
"""

import numpy as np
import re
import os
import pickle as pkl


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
    strin = re.sub('\s+', ' ', string)
    return string.strip()


def read_txt(filepath, replace=None, min_line_len=1, 
             encoding='utf8', join_on=' '):
    '''
    Read txt file and return as a single string
    Preprocessing:
        replace some chars
        add length threshold
    '''
    with open(filepath, 'r', encoding=encoding) as f:
        txt = f.readlines()
    lines = []
    for line in txt:
        if replace:
            line = replace_chars(line, replace)
        if len(line) >= min_line_len:
            lines.append(line)
    return join_on.join(lines)



def get_class_sizes(base_dir, class_list, print_stats=True):
    '''
    Prints out the length of each class from a folder
    And returns them as a dictionary
    '''
    class_lens = {c: len(os.listdir(os.path.join(base_dir,c))) for c in class_list}    
    if print_stats:
        print('Files per each class:')
        print(class_lens)
    return class_lens
    

def get_class_files(base_dir, class_list):
    '''
    Get list of all filenames for each class
    And returns them as dictionary
    '''
    class_files = {}
    for c in class_list:
        class_dir = os.path.join(base_dir, c)
        class_files[c] = [os.path.join(base_dir,c,f) for f in os.listdir(class_dir)]
    return class_files


def pickle_data(filename, value=None, mode='w'):
    '''
    Saves a value into filename as pickle if mode == 'w'
    Else if mode == 'r' then reads a value from filename
    '''
    if mode == 'w':
        with open(filename, 'wb') as f:
            pkl.dump(value, f)
        return None
    elif mode == 'r':
        with open(filename, 'rb') as f:
            unpickled = pkl.load(f)
        return unpickled
    else:
        raise Exception('mode should be in ("w","r")')