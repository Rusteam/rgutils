
import matplotlib.pyplot as plt

def plot_learning_curve(history, fig_shape=(12,4), plot_style='fivethirtyeight'):
    '''
    Plots learning curve for loss and score both for train and validation
    -------
    :metrics is a dict with following keys: ['step', 'val_loss', 'loss', 'val_score', 'score'], where values are lists
    '''
    plt.style.use(plot_style)
    plt.figure(figsize=fig_shape)
    plt.subplot(121)
    plt.title('Loss curves')
    plt.plot(history['epoch'], history['train_loss'], 'r--', label='train loss')
    plt.plot(history['epoch'], history['val_loss'], 'b--', label='validation loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.subplot(122)
    plt.title('Score curves')
    plt.plot(history['epoch'], history['train_score'], 'r--', label='train score')
    plt.plot(history['epoch'], history['val_score'], 'b--', label='validation score')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
