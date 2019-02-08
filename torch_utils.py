
import torch
from math import floor


def train_epoch(model, dataloader, input_size, device, optimizer, criterion,):
    '''
    Makes 1 training epoch and evaluates train error (for multi-class classification)
    -----
    Returns training loss and accuracy   
    '''
    model.train()
    
    running_loss = 0
    num_correct = 0
    for img_data,img_labels in dataloader:
        images = img_data.view(-1, *input_size).to(device)
        labels = img_labels.to(device)
        
        y_pred = model(images)
        loss = criterion(y_pred, labels,)
        
        _,y_pred_labels = torch.max(y_pred, dim=1)
        num_correct += torch.sum(labels == y_pred_labels).item()
        running_loss += loss.item() * labels.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = num_correct / len(train_loader.dataset) * 100
    return epoch_loss, epoch_acc


def eval_epoch(model, dataloader, input_size, device, criterion):
    '''
    Evaluate validation error after an epoch of training
    '''
    model.eval()
    running_loss = 0
    num_correct = 0
    for img_data, img_labels in dataloader:
        images = img_data.view(-1, *input_size).to(device)
        labels = img_labels.to(device)
        
        y_pred = model(images)
        loss = criterion(y_pred, labels,)
        
        _,y_pred_labels = torch.max(y_pred, dim=1)
        num_correct += torch.sum(labels == y_pred_labels).item()
        running_loss += loss.item() * labels.size(0)
    
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = num_correct / len(val_loader.dataset) * 100
    return val_loss, val_acc


def run_training(num_epoch, model, dataloaders, input_size, optimizer, criterion, 
                 device, history=None, print_stats=True):
    '''
    Runs training for a specified number of epochs
    '''
    if history is not None:
        start_epoch = max(history['epoch']) + 1
        num_epoch = num_epoch + start_epoch
    else:
        history = {'epoch': [], 'train_score': [], 'val_score': [], 'train_loss': [], 'val_loss': []}
        start_epoch = 0
    for ep in range(start_epoch, num_epoch):
        if print_stats:
            print('='*10, 'Epoch', ep, '='*10)

        train_loss, train_score = train_epoch(model, dataloaders['train'], input_size, device, optimizer, criterion)
        val_loss, val_score = eval_epoch(model, dataloaders['val'], input_size, device, criterion)
        history['epoch'].append(ep)
        history['train_score'].append(train_score), history['train_loss'].append(train_loss)
        history['val_score'].append(val_score), history['val_loss'].append(val_loss)
        
        if print_stats:
            print('%7d Loss | Score' % ep)
            print('Train %.4f | %.1f' % (train_loss, train_score))
            print('Valid %.4f | %.1f' % (val_loss, val_score))
            print()
    return model, history


def get_output_size(input_size, kernel_size, padding, stride):
    '''
    Calculates an output size of a feature map given input dimension
    '''
    return floor((input_size - kernel_size + 2*padding) / stride) + 1
