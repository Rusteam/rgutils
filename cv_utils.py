# coding=utf-8

import os
import cv2
import numpy as np


def extract_frames(video_file, save_dir, frame_skip=0, file_exension='jpeg', video_format='mp4'):
    '''
    Extracts frames from a video file and saves them into specified dir
    ------
    Params:
    :video_file - a path to a videofile
    :save_dir - a folder to save frames
    :frame_skip - number of frames to skip
    :file_exension - save images as
    :video_format - extension of the passed video
    '''
    dir_name = os.path.join(save_dir, video_file.split('/')[-1].replace('.'+video_format, ''))
    os.makedirs(dir_name, exist_ok=True)
    cap = cv2.VideoCapture(video_file)
    frame_number = frame_skip
    file_names = []
    while cap.isOpened():
        if frame_skip > 0:
            cap.set(1, frame_number)
        ret, frame = cap.read()
        if ret:
            name = os.path.join(dir_name, '.'.join([str(frame_number), file_exension]))
            cv2.imwrite(name, frame)
            file_names.append(name)
        else:
            break
        frame_number += max(frame_skip, 1)
    cap.release()
    print('{} frames have been extracted from {}'.format(len(file_names), video_file))
    return file_names

