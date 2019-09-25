
import torch
from torch import nn, optim
from tqdm import tqdm
import copy
import re
from math import floor
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
for mod in ['mlflow',]:
    try:
        exec(f'import {mod}')
    except ImportError as e:
        print(e)


class BinaryClassificationMeter(object):
    """
    Computes binary classification metrics for torch models
    Courtesy of https://gist.github.com/kyamagu/73ab34cbe12f3db807a314019062ad43
    """
    def __init__(self):
        self.reset()

        
    def reset(self):
        self.correct = 0
        self.total = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.precision = 0
        self.f1_score = 0
        self.batches = 0

        
    def update(self, true_labels, predicted_probs):
        pred = predicted_probs >= 0.5
        true_labels = true_labels >= 0.5
        self.correct += torch.sum(true_labels == pred).item()
        self.total += true_labels.size(0)
        self.accuracy = self.correct /  self.total * 100
        self.f1_score += f1_score(true_labels.cpu().numpy(), pred.cpu().numpy())
        self.recall += recall_score(true_labels.cpu().numpy(), pred.cpu().numpy())
        self.precision += precision_score(true_labels.cpu().numpy(), pred.cpu().numpy())
        self.batches += 1
        
        
    def values(self):
        return {'accuracy': self.accuracy,
                'f1_score': self.f1_score / self.batches,
                'precision': self.precision / self.batches,
                'recall': self.recall / self.batches}
        
        
class MulticlassClassificationMeter:
    '''
    Computes multi-class accuracy
    '''
    def __init__(self):
        self.reset()
        
        
    def reset(self):
        self.correct = 0
        self.total = 0
        self.accuracy = 0
        self.f1_weighted = 0
        self.f1_macro = 0
        self.batches = 0
        
        
    def update(self, true_labels, predicted_probs):
        _,y_pred_labels = torch.max(predicted_probs, dim=1)
        self.correct += torch.sum(true_labels == y_pred_labels).item()
        self.total += true_labels.size(0)
        self.accuracy = self.correct /  self.total * 100
        self.f1_weighted += f1_score(true_labels.cpu().numpy(), y_pred_labels.cpu().numpy(), average='weighted')
        self.f1_macro += f1_score(true_labels.cpu().numpy(), y_pred_labels.cpu().numpy(), average='macro')
        self.batches += 1
        
        
    def values(self):
        return {'accuracy': self.accuracy,
                'f1_weighted': self.f1_weighted / self.batches,
                'f1_macro': self.f1_macro / self.batches,
                }

        
classification_meters = {
    'binary': BinaryClassificationMeter(),
    'multi': MulticlassClassificationMeter(),
}
     
    
class Trainer():
    '''
    Trains a model for specified number of epochs and evaluates
    ------
    Usage:
    trainer = Trainer(**training_configs)
    model, history = trainer.run_training(**running_configs)

    Note: to resume training, run trainer.run_training() and it will return aggregate history
    '''
    def __init__(self, classification, model, optimizer, criterion, device, input_size,
                 train_dataloader, val_dataloader, test_dataloader=None, data_dtype=torch.float32,
                 mlflow_tracking=None,
                 tqdm_off=True):
        '''
        Set training configs:
            classification - binary or multi
            model - neural net itself
            optimizer - either torch.optim.Optimizer or (Optimizer str, {lr: f, ...})
            criterion - loss function
            device - torch.device object: cuda or cpu
            input_size - input size to a model
            train_dataloader, val_dataloader, test_dataloader - data loaders, test could be omitted
            data_dtype - convert X to this type before passing to the model
            mlflow_tracking - provide configs for using mlflow tracking api: {name: expirement name, params: training configs}
            tqdm_off - hide or unhide tqdm for epoch tracking
        '''
        self.classification = classification
        self.model = model
        if isinstance(optimizer, (list,tuple)):
            assert len(optimizer) == 2, f'Provide optimizer as a tuple of (Name, {kwargs})'
            trainable_parameters = filter(lambda x: x.requires_grad, self.model.parameters())
            self.optimizer = getattr(optim, optimizer[0])(trainable_parameters, **optimizer[1])
        else:
            self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.input_size = input_size
        self.meter = classification_meters[self.classification]
        self.train_dataloader = train_dataloader
        self.houldout_sets = {'val': val_dataloader,
                              'test': test_dataloader}
        self.tqdm_off = tqdm_off
        self.history = None
        self.data_dtype = data_dtype

        if mlflow_tracking:
            tracking_keys = {'params':dict, 'name':str}
            for k,tp in tracking_keys.items():
                assert k in list(mlflow_tracking.keys()) \
                        and isinstance(mlflow_tracking[k], tp), f'{k}:{tp} should be in mlflow_tracking argument'
            mlflow.set_experiment(mlflow_tracking['name'])
            self.mlflow_tracking = mlflow_tracking
        else:
            self.mlflow_tracking = None
            

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
            images = img_data.view(-1, *self.input_size).to(self.device, dtype=self.data_dtype)
            labels_dtype = torch.long if self.classification == 'multi' else torch.float32
            labels = img_labels.to(self.device, dtype=labels_dtype)

            y_pred = torch.squeeze(self.model(images), dim=1)
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
                images = img_data.view(-1, *self.input_size).to(self.device, dtype=self.data_dtype)
                labels_dtype = torch.long if self.classification == 'multi' else torch.float32
                labels = img_labels.to(self.device, dtype=labels_dtype)

                y_pred = torch.squeeze(self.model(images), dim=1)
                loss = self.criterion(y_pred, labels,)

                meter.update(labels, y_pred)
                running_loss += loss.item() * labels.size(0)

        val_loss = running_loss / len(data_loader.dataset)
        return val_loss, meter.values()


    def run_training(self, num_epoch, monitor_metrics, early_stopping=None, best_metric='loss',
                     unfreeze_after=None, unfreeze_options=None, unfreeze_optim=None,
                     print_stats=True):
        '''
        Runs training for a specified number of epochs
        If called again, resumes traning
        -------
        Params:
            num_epoch - number of epochs to train for
            monitor_metrics - list of metrics to minotor
            early_stopping - number of epochs to wait before stopping if best metric does not improve
            best_metric - a metric to choose best weights and used in early stopping
            unfreeze_after - unfreeze layers during training
            unfreeze_options - add inclusion or exclusion patterns for unfreezing layers
            unfreeze_optim - optimizer to use for newly unfreezed layers
        Returns:
            trained model, history
        '''
        # if running again then add up to a history
        assert best_metric == 'loss' or best_metric in monitor_metrics
        if unfreeze_after:
            assert unfreeze_options is not None, 'If using unfreeze provide unfreeze options as well (could be empty dict)'
        if self.history is not None: #resumes training
            history = copy.deepcopy(self.history)
            start_epoch = max(self.history['epoch']) + 1
            num_epoch = num_epoch + start_epoch
            if self.mlflow_tracking:
                mlflow.start_run(self.mlflow_run.info.run_id)
        else:
            history = {'epoch': [], 'train_loss': [], 'val_loss': []}
            for metr in monitor_metrics:
                history['_'.join(['train',metr])] = []
                history['_'.join(['val',metr])] = []
            start_epoch = 0
            if self.mlflow_tracking:
                self.mlflow_run = mlflow.start_run()
                mlflow.log_params(self.mlflow_tracking['params'])
        # store best model weights
        best_model_weights = copy.deepcopy(self.model.state_dict())
        best_val_score = 1e6
        no_improvement = 0
        # train for given number of epochs
        for ep in tqdm(range(start_epoch, num_epoch), disable=self.tqdm_off):
            # unfreeze layers if it is a time and update optimizer
            if ep == unfreeze_after:
                unfreeze_layers(self.model, **unfreeze_options)
                if unfreeze_optim:
                    assert isinstance(unfreeze_optim, (tuple,list)) and len(unfreeze_optim) == 2, \
                                f'Provide unfreeze_optimizer as a tuple of (Name, {kwargs})'
                    trainable_parameters = filter(lambda x: x.requires_grad, self.model.parameters())
                    old_sd = self.optimizer.state_dict()
                    self.optimizer = getattr(optim, unfreeze_optim[0])(trainable_parameters, **unfreeze_optim[1])
                    new_sd = self.optimizer.state_dict()
                    new_sd['state'].update(old_sd['state'])
                    self.optimizer.load_state_dict(new_sd)
                parameter_number(self.model)
            # make 1 run thru training data, compute and save scores
            train_loss, train_meter = self.train_epoch()
            val_loss, val_meter = self.eval_model('val')
            history['epoch'].append(ep)
            history['train_loss'].append(train_loss), history['val_loss'].append(val_loss)
            mlflow_metrics = {'train_loss': train_loss, 'val_loss': val_loss}
            for metr in monitor_metrics:
                t_n = '_'.join(['train',metr])
                history[t_n].append(train_meter[metr])
                v_n = '_'.join(['val',metr])
                history[v_n].append(val_meter[metr])
                # add to mlflow dict
                mlflow_metrics.update({t_n: train_meter[metr], v_n: val_meter[metr]})
            # log params
            if self.mlflow_tracking:
                mlflow.log_metrics(mlflow_metrics, step=ep)
            # print stats if requested
            if print_stats:
                print('='*10, 'Epoch', ep, '='*10)
                print('%7d Loss | Score' % ep)
                print('Train %.4f | %.3f' % (train_loss, train_meter['accuracy']))
                print('Valid %.4f | %.3f' % (val_loss, val_meter['accuracy']))
                print()
            # store weights for best model
            val_score = history[f'val_{best_metric}'][-1]
            if best_metric != 'loss':
                val_score = val_score * (-1)
            if val_score < best_val_score:
                best_model_weights = copy.deepcopy(self.model.state_dict())
                best_val_score = val_score
                no_improvement = 0
            else:
                no_improvement += 1
            # early stopping if specified
            if early_stopping:
                if no_improvement >= early_stopping:
                    print(f'Early stopping at {ep} epoch')
                    break
        # get best model
        self.model.load_state_dict(best_model_weights)
        # run test metrics
        if self.houldout_sets['test'] is not None:
            test_loss,test_results = self.eval_model(holdout_type='test')
            history['test_results'] = (test_loss,test_results)
            print()
            print('='*20)
            print(f'Test loss {test_loss:.4f}')
            print('Test results:', test_results)
        # log test and end run
        if self.mlflow_tracking:
            if self.houldout_sets['test'] is not None:
                test_results = {f'test_{k}':round(v,4) for k,v in test_results.items()}
                test_results.update({'test_loss':test_loss})
                mlflow.log_metrics(test_results)
            mlflow.end_run()
        self.history = copy.deepcopy(history)
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
        x = self.conv_forward(x.float())
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


def unfreeze_layers(model, include_pattern=None, except_pattern=None):
    '''
    Unfreeze all layers for training
    If including is specified then only those (first test)
    If excluding is specified then all except those (second test)
    '''
    for n,p in model.named_parameters():
        if include_pattern or except_pattern:
            if include_pattern and re.search(include_pattern, n) is not None:
                p.requires_grad = True
            if except_pattern and re.search(except_pattern, n) is not None:
                p.requires_grad = False
        else:
            p.requires_grad = True
