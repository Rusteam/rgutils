
import torch
from torch import nn
import copy
from math import floor
import numpy as np


class BinaryClassificationMeter(object):
    """
    Computes binary classification metrics for torch models
    Courtesy of https://gist.github.com/kyamagu/73ab34cbe12f3db807a314019062ad43
    """
    def __init__(self):
        self.reset()

        
    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0

        
    def update(self, true_labels, predicted_probs):
        eps = 1e-6
        pred = predicted_probs >= 0.5
        true_labels = true_labels >= 0.5
        self.tp += pred.mul(true_labels).sum(0).float()
        self.tn += (1 - pred).mul(1 - true_labels).sum(0).float()
        self.fp += pred.mul(1 - true_labels).sum(0).float()
        self.fn += (1 - pred).mul(true_labels).sum(0).float()        
        self.accuracy = ((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)).item()
        self.precision = (self.tp / (self.tp + self.fp + eps)).item()
        self.recall = (self.tp / (self.tp + self.fn + eps)).item()
        self.f1_score = (2.0 * self.precision * self.recall) / (self.precision + self.recall + eps)
        
        
    def values(self):
        return {'accuracy': self.accuracy,
                'f1_score': self.f1_score,
                'precision': self.precision,
                'recall': self.recall}
        
        
class MulticlassClassificationMeter(object):
    '''
    Computes multi-class accuracy
    '''
    def __init__(self):
        self.reset()
        
        
    def reset(self):
        self.correct = 0
        self.total = 0
        self.accuracy = 0
        
        
    def update(self, true_labels, predicted_probs):
        _,y_pred_labels = torch.max(predicted_probs, dim=1)
        self.correct += torch.sum(true_labels == y_pred_labels).item()
        self.total += true_labels.size(0)
        self.accuracy = self.correct /  self.total * 100
        
        
    def values(self):
        return {'accuracy': self.accuracy}

        
classification_meters = {
    'binary': BinaryClassificationMeter(),
    'multi': MulticlassClassificationMeter(),
}
     
    
class Trainer():
    '''
    Trains a model for specified number of epochs and evaluates
    ------
    Usage:
    trainer = Trainer('binary', net, optimizer, criterion, device, input_dims, 
                  train_loader, val_loader, test_loader)
    model, history = trainer.run_training(num_epoch, ['accuracy','f1_score', 'precision', 'recall'],)
    '''
    def __init__(self, classification, model, optimizer, criterion, device, input_size,
                 train_dataloader, val_dataloader, test_dataloader=None):
        self.classification = classification
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.input_size = input_size
        self.meter = classification_meters[self.classification]
        self.train_dataloader = train_dataloader
        self.houldout_sets = {'val': val_dataloader,
                              'test': test_dataloader}
            

    def train_epoch(self,):
        '''
        Makes 1 training epoch and evaluates train error (for multi-class classification)
        -----
        Returns training loss and accuracy   
        '''
        self.model.train()

        running_loss = 0
        meter = classification_meters[self.classification]
        meter.reset()
        
        for img_data,img_labels in self.train_dataloader:
            images = img_data.view(-1, *self.input_size).to(self.device)
            labels = img_labels.to(self.device).float()

            y_pred = torch.squeeze(self.model(images))
            loss = self.criterion(y_pred, labels,)
            
            meter.update(labels, y_pred)
            running_loss += loss.item() * labels.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss = running_loss / len(self.train_dataloader.dataset)
        return epoch_loss, meter.values()


    def eval_model(self, holdout_type='val'):
        '''
        Evaluate validation error after an epoch of training
        '''
        self.model.eval()

        running_loss = 0
        meter = classification_meters[self.classification]
        meter.reset()
        
        data_loader = self.houldout_sets[holdout_type]
        with torch.no_grad():
            for img_data, img_labels in data_loader:
                images = img_data.view(-1, *self.input_size).to(self.device)
                labels = img_labels.to(self.device).float()

                y_pred = torch.squeeze(self.model(images))
                loss = self.criterion(y_pred, labels,)

                meter.update(labels, y_pred)
                running_loss += loss.item() * labels.size(0)

        val_loss = running_loss / len(data_loader.dataset)
        return val_loss, meter.values()


    def run_training(self, num_epoch, monitor_metrics, history=None, print_stats=True):
        '''
        Runs training for a specified number of epochs
        '''
        # initialize history if not given
        if history is not None:
            start_epoch = max(history['epoch']) + 1
            num_epoch = num_epoch + start_epoch
        else:
            history = {'epoch': [], 'train_loss': [], 'val_loss': []}
            for metr in monitor_metrics:
                history['_'.join(['train',metr])] = []
                history['_'.join(['val',metr])] = []
            start_epoch = 0
            
        # store best model weights
        best_model_weights = copy.deepcopy(self.model.state_dict())
        best_val_loss = 1e6
                
        # train for given number of epochs
        for ep in range(start_epoch, num_epoch):               
            # make 1 run thru training data, compute and save scores
            train_loss, train_meter = self.train_epoch()
            val_loss, val_meter = self.eval_model('val')
            history['epoch'].append(ep)
            history['train_loss'].append(train_loss), history['val_loss'].append(val_loss)
            for metr in monitor_metrics:
                history['_'.join(['train',metr])].append(train_meter[metr])
                history['_'.join(['val',metr])].append(val_meter[metr])
            
            # print stats if requested
            if print_stats:
                print('='*10, 'Epoch', ep, '='*10)
                print('%7d Loss | Score' % ep)
                print('Train %.4f | %.3f' % (train_loss, train_meter['accuracy']))
                print('Valid %.4f | %.3f' % (val_loss, val_meter['accuracy']))
                print()
                
            # store weights for best model
            if val_loss < best_val_loss:
                best_model_weights = copy.deepcopy(self.model.state_dict())
                best_val_loss = val_loss
               
        self.model.load_state_dict(best_model_weights)
        return self.model, history


def get_output_size(input_size, kernel_size, padding, stride):
    '''
    Calculates an output size of a feature map given input dimension
    '''
    items = {'input_size': input_size, 'kernel_size': kernel_size,
             'padding': padding, 'stride': stride}
    for k,v in items.items():
        if isinstance(v, (tuple, list)):
            items[k] = v[0]
    return floor((items['input_size'] - items['kernel_size'] + 2*items['padding']) / items['stride']) + 1


def reshaped_size(input_size, conv_layers, print_stats=True):
    '''
    Calculates reshaped size after last conv layer
    Returns spatial size and reshaped size
    '''
    out_chan = 1
    for _,l_ops in conv_layers.items():
        for op,op_attrs in l_ops.items():
            try:
                input_size = get_output_size(input_size, getattr(op_attrs, 'kernel_size'), 
                                     getattr(op_attrs, 'padding'), getattr(op_attrs, 'stride'))
                out_chan = getattr(op_attrs, 'out_channels', out_chan)
            except:
                continue

    reshaped_size = input_size**2 * out_chan
    if print_stats:
        print('Output spatial dimension', input_size)
        print('Output size (Width * height * channels)', reshaped_size)
    return input_size, reshaped_size


class SimpleCNN(nn.Module):
    '''
    A simple ConvNet with conv layers
    -------
    Params:
    :conv_layers - an ordered dict of layers where each layer is an ordered dict of conv operations
    :fc_layers - an ordered dict of layers where each layer is an ordered dict of fully-connected operations
    :reshaped_size - a dimension after flatenning last conv layer's output (w_out * h_out * c_out)
    -------
    Example:
    conv_layers = OrderedDict()

    conv_layers['l0'] = OrderedDict()
    conv_layers['l0']['conv1'] = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
    conv_layers['l0']['activation'] = nn.ReLU()
    conv_layers['l0']['bn'] = nn.BatchNorm2d(16)
    conv_layers['l0']['pool'] = nn.MaxPool2d(2, stride=2, padding=0)

    conv_layers['l1'] = OrderedDict()
    conv_layers['l1']['conv1'] = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
    conv_layers['l1']['activation'] = nn.ReLU()
    conv_layers['l1']['bn'] = nn.BatchNorm2d(32)
    conv_layers['l1']['pool'] = nn.MaxPool2d(2, stride=2, padding=0)
    '''
    def __init__(self, conv_layers, fc_layers, reshaped_size, ):
        super(SimpleCNN, self).__init__()
        
        conv_forward = []
        for l_name,l_ops in conv_layers.items():
            for op_name,op_attrs in l_ops.items():
                conv_forward.append(op_attrs)
        self.conv_forward = nn.Sequential(*conv_forward)
                    
        self.reshaped_size = reshaped_size
        fc_forward = []
        for l_name,l_ops in fc_layers.items():
            for op_name,op_attrs in l_ops.items():
                fc_forward.append(op_attrs)
        self.fc_forward = nn.Sequential(*fc_forward)
        
                    
    def forward(self, x):
        x = self.conv_forward(x)
        x = x.view(-1, self.reshaped_size)
        return self.fc_forward(x)
    

def parameter_number(net):
    '''
    Print out number of total and trainable parameters in a neural network
    '''
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total parameters:', total_params)
    print('Trainable parameters:', trainable_params)


def autoencoder_output(model, data_loader, num_show, input_dim, output_dim, device,):
    '''
    Randomly selects input images and returns a paired output from an autoencoder model
    '''
    rand_indices = np.random.randint(0, len(data_loader.dataset), size=num_show)
    paired_arrays = []
    for idx in rand_indices:
        x,_ = data_loader.dataset[idx]
        x_reshaped = x.view(*input_dim).to(device)
        
        y = model(x_reshaped)
        x_out = np.array(x_reshaped.view(*output_dim).detach())
        y_out = np.array(y.view(*output_dim).detach())
        paired_arrays.append([x_out, y_out])
    return np.array(paired_arrays)
