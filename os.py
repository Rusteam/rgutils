# -*- coding: utf-8 -*-
"""
Operations on OS:
    Read, write, remove files
    Get a list of files
    Working with directories
"""
import os, shutil
import pickle as pkl
import json


def create_dir(dir_path, empty_if_exists=True):
    '''
    Creates a folder if not exists,
    If does exist, then empties it by default
    '''
    if os.path.exists(dir_path):
        if empty_if_exists:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)


def get_abs_path(src, dest, depth=2):
    '''
    Get correct absolute path by going deeper from source to find project dir
    And join it with dest
    '''
    project_dir = os.path.abspath(src)
    for i in range(depth):
        project_dir = os.path.dirname(project_dir)
    return os.path.join(project_dir, dest)


def read_txt(filepath, encoding='utf-8', sep='\n'):
    '''
    Read txt file and return as a list of strings
    '''
    with open(filepath, 'r', encoding=encoding) as f:
        txt = f.read().strip(sep).split(sep)
    return txt


def write_txt(filepath, data, encoding='utf-8', sep='\n'):
    '''
    Write a list of objects into a txt file
    '''
    with open(filepath, 'w', encoding=encoding) as f:
        f.writelines(sep.join(data))


def get_class_sizes(base_dir, class_list, print_stats=True):
    '''
    Print out the length of each class from a folder
    And return them as a dictionary
    '''
    class_lens = {c: len(os.listdir(os.path.join(base_dir,c))) for c in class_list}
    if print_stats:
        print('Files per each class:')
        print(class_lens)
    return class_lens


def get_class_files(base_dir, class_list=None):
    '''
    Get a list of all filenames for each class
    And return them as dictionary
    If class_list not provided then use all subfolders from base_dir as classes
    '''
    if class_list is None:
        class_list = os.listdir(base_dir)
    class_files = {}
    for c in class_list:
        class_dir = os.path.join(base_dir, c)
        class_files[c] = [os.path.join(base_dir,c,f) for f in os.listdir(class_dir)]
    return class_files


def pickle_data(filename, value=None, mode='w'):
    '''
    Save a value into filename as pickle if mode == 'w'
    Else if mode == 'r' then reads a value from filename
    '''
    if mode == 'w':
        assert value is not None, 'Do not overwrite filename with None'
        with open(filename, 'wb') as f:
            pkl.dump(value, f)
        return None
    elif mode == 'r':
        with open(filename, 'rb') as f:
            unpickled = pkl.load(f)
        return unpickled
    else:
        raise Exception('mode should be in ("w","r")')


def load_json(filepath, **kwargs):
    '''
    Load a json file and return it
    '''
    assert os.path.exists(filepath)
    with open(filepath, 'r') as f:
        data = json.load(f, **kwargs)
    return data


def write_json(data, filepath, **kwargs):
    '''
    Write data into a json file
    '''
    with open(filepath, 'w') as f:
        json.dump(data, f, **kwargs)
