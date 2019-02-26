# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def image_labels_grid(n_row, n_col, image_data, fig_dims=(12,8), 
                      plot_style='seaborn-white', width_space=0.4, height_space=0.1,
                      save_fig=None):
    '''
    Plots random images with labels for a dataset with methods __len__ & __getitem__
    (Two values for each index: an image and its label)
    '''
    plt.style.use(plot_style)
    rand_indices = np.random.randint(low=0, high=len(image_data), size=(n_row, n_col))
    fig,ax = plt.subplots(n_row, n_col, sharex=True, sharey=True, squeeze=False, 
                          figsize=fig_dims)
    fig.subplots_adjust(hspace=height_space, wspace=width_space)
    for r in range(n_row):
        for c in range(n_col):
            ax[r, c].imshow(image_data[rand_indices[r,c]][0])
            ax[r, c].set_title('Label: ' + str(image_data[rand_indices[r,c]][1]))
    
    if save_fig:
        plt.savefig(save_fig)
    plt.show()

    
def plot_learning_curve(history, metrics=['loss','accuracy'], grid=(1,2),
                        fig_shape=(12,4), plot_style='fivethirtyeight', save_to=None):
    '''
    Plots learning curves both for train and validation
    for a specified list of metrics[i]ics
    -------
    '''
    plt.style.use(plot_style)
    fig,ax = plt.subplots(grid[0], grid[1], sharex=False, sharey=False, figsize=fig_shape)
    for i in range(grid[0]*grid[1]):    
        row = i // grid[1]
        col = i % grid[1]
        if grid[0] > 1:
            pos = (row, col)
        else:
            pos = (col,)
        if i >= len(metrics):
            fig.delaxes(ax[pos])
            break
        ax[pos].set_title('{} curves'.format(metrics[i]))
        ax[pos].plot(history['epoch'], history['train_{}'.format(metrics[i])], 'r--', label='train {}'.format(metrics[i]))
        ax[pos].plot(history['epoch'], history['val_{}'.format(metrics[i])], 'b--', label='validation {}'.format(metrics[i]))
        ax[pos].set_xlabel('epoch')
        ax[pos].legend()

    plt.tight_layout()
    plt.show()

    
def plot_image_pairs(pairs, grid, fig_shape=(15,7), save_fig=None, style='fivethirtyeight'):
    '''
    Plots pairs of images along 2 rows
    '''
    plt.style.use(style)
    fig,ax = plt.subplots(grid[0], grid[1], sharex=True, sharey=True, figsize=fig_shape)
    for i in range(grid[1]):
        ax[0, i].imshow(pairs[i, 0])
        ax[0, i].axis('off')
        ax[1, i].imshow(pairs[i, 1])
        ax[1, i].axis('off')
    if save_fig: 
        plt.savefig(save_fig)
    plt.show()

    
def plot_feature_importances(feature_names, feature_importances, 
                             orient='h', max_show=15, fig_shape=(8,6)):
    '''
    Plots feature importances
    '''
    feat_imp = pd.DataFrame(
        {'feature': feature_names,
         'importance': feature_importances})
    feat_imp.sort_values('importance', ascending=False, inplace=True)
    plt.figure(figsize=fig_shape)
    plt.title('Feature importances')
    sns.barplot(feat_imp['importance'].iloc[:max_show],
                feat_imp['feature'].iloc[:max_show],
                orient=orient)
    plt.show()
