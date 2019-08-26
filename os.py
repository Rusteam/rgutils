# -*- coding: utf-8 -*-
"""
Operations on OS
"""
import os, shutil


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
