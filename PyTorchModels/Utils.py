import numpy as np 
import json
import torch
import random
import torch.nn as nn 
import torchvision
import matplotlib.pyplot as plt
from datetime import datetime


# Device Discovery
def check_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

# Initialize Optimizer
def optim_init(model, optim_dict, scheduler_dict):
    optim_TYPE = optim_dict['optim_TYPE']
    scheduler_TYPE = scheduler_dict['schedule_TYPE']
    if optim_TYPE == 'SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr = optim_dict['learning_rate'],
                                momentum = optim_dict['momentum'],
                                weight_decay = optim_dict['weight_decay'],
                                nesterov = optim_dict['nesterov'])
    elif optim_TYPE == 'Adams':
        optim = torch.optim.Adam(model.parameters(),
                                 lr = optim_dict['learning_rate'],
                                 betas = optim_dict['betas'],
                                 weight_decay = optim_dict['weight_decay'],
                                 amsgrad = optim_dict['amsgrad'])
    elif optim_TYPE == 'AdaGrad':
        optim = torch.optim.Adagrad(model.parameters(),
                                    lr = optim_dict['learning_rate'],
                                    lr_decay = optim_dict['lr_decay'],
                                    weight_decay = optim_dict['weight_decay'])
    if scheduler_TYPE == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                         milestones = scheduler_dict['milestones'],
                                                         gamma = scheduler_dict['gamma'])
    return optim, scheduler

# Initialize Weight
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain = 1.0)
        # nn.init.xavier_normal_(m.weight, gain = 1.0)
        # nn.init.kaiming_uniform_(m.weight, a = 0, mode = 'fan_in', nonlinearity = 'relu')
        # nn.init.normal_(m.weight, 0, 1)
        # m.bias.data.fill_(0.00)
    elif isinstance(m, nn.Conv2d):
        # nn.init.xavier_uniform_(m.weight, gain = 1.0)
        nn.init.kaiming_uniform_(m.weight, a = 0, mode = 'fan_in', nonlinearity = 'relu')
        if m.bias is not None:
            nn.init.normal_(m.bias, 0, 1)


# Initialize DataAug
def transform_init():
    pass


# Dataset Loader
def DatasetLoader(name, Train_transform, Test_transform, batch_size, TrainShrink = 1.0):
    if name == 'CIFAR10':
        X_train = torchvision.datasets.CIFAR10('./dataset/',
                                                train = True,
                                                transform = Train_transform,
                                                download = True)
        X_test = torchvision.datasets.CIFAR10('./dataset/',
                                                train = False,
                                                transform = Test_transform,
                                                download = True)
        selectedIdx = random.sample(range(len(X_train)), int(len(X_train) * TrainShrink))
        X_train = torch.utils.data.Subset(X_train, selectedIdx)
        TrainLoader = torch.utils.data.DataLoader(  dataset = X_train,
                                                    batch_size = batch_size,
                                                    num_workers = 4,
                                                    shuffle = True)
        TestLoader  = torch.utils.data.DataLoader(  dataset = X_test,
                                                    batch_size = 2 * batch_size,
                                                    num_workers = 4,
                                                    shuffle = False)
        labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return [TrainLoader, TestLoader], labels


# Utiles Function for Showing Time
def showTime(func):
    def wrapper(*args, **kw):
        startTime = datetime.now()
        value = func(*args, **kw)
        print('\nTime Elapsed: {0}'.format(datetime.now() - startTime))
        return value
    return wrapper


# Coarse Validation
def coarse_validate(model, dataLoader, device):
    total = correct = 0.
    with torch.no_grad():
        for i, data in enumerate(dataLoader):
            image, label = data[0].to(device), data[1].to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return 100 * correct / total


# Fine Validation
def fine_validate(model, dataLoader, device, labels):
    num_labels = len(labels)
    confusionMx = [[0. for _ in range(num_labels)] for _ in range(num_labels)]
    class_correct = [0. for _ in range(num_labels)]
    class_total = [0. for _ in range(num_labels)]
    with torch.no_grad():
        for i, data in enumerate(dataLoader):
            images, true_label = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == true_label).squeeze()
            for i in range(len(true_label)):
                label, pred = true_label[i].item(), predicted[i].item()
                class_correct[label] += (label == pred)
                class_total[label] += 1
                confusionMx[label][pred] += 1.
    return [100 * class_correct[i] / class_total[i] for i in range(num_labels)], confusionMx
    


# Train One Epoch
@showTime
def train_one_epoch_(model, dataLoader, lossFunc, optimizer, device):
    running_loss = 0.
    total_iter = len(dataLoader)
    print('Progress: >', end = '\r')
    for i, data in enumerate(dataLoader):
        image, label = data[0].to(device), data[1].to(device)    
        optimizer.zero_grad()
        # Forward + Backward
        output = model(image)
        loss = lossFunc(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print('Progress: ' + '=' * (i * 75 // total_iter) + '>', end = '\r')
    return running_loss

# Train the Model
def train(model, dataLoader_all, lossFunc, optimizer, device, total_epoch, scheduler):
    print('Training Starts')
    train_acc_log, test_acc_log, loss_log = [], [], []
    for epoch in range(total_epoch):
        # loss = train_one_epoch_(model, dataLoader_all[0], lossFunc, optimizer, device)
        model.train()
        loss = train_one_epoch_(model, dataLoader_all[0], lossFunc, optimizer, device)
        print('Loss %.3f of Epoch %d' % (loss, epoch + 1), end = ' ')
        loss_log.append(loss)
        # Validation
        model.eval()
        train_acc_log.append(coarse_validate(model, dataLoader_all[0], device))
        test_acc_log.append(coarse_validate(model, dataLoader_all[1], device))
        print('Train-Acc: %.3f %%, Validate-Acc: %.3f %%' % (train_acc_log[-1], test_acc_log[-1]))
        # Step in lr_Scheduler
        scheduler.step()
    print('Training Finished')
    return train_acc_log, test_acc_log, loss_log

# Visualization
def visualize_plt(plot_name, train_log, test_log, loss_log):
    x = [i for i in range(len(train_log))]
    # Graph 1: Training / Test Acc
    plt.subplot(1, 2, 1)
    plt.plot(x, train_log, c = 'Red', lw = 1)
    plt.plot(x, test_log, c = 'Blue', lw = 1) 
    plt.title('Performance Curve (Epoch)')
    plt.legend(['Train', 'Test'], loc = 'lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    # Graph 2: Training Loss
    plt.subplot(1, 2, 2)
    plt.plot(x, loss_log, c = 'Red', lw = 1)
    plt.legend(['Loss'], loc = 'upper right')
    plt.title('Loss')
    plt.suptitle('Training-Progress')
    plt.savefig(plot_name + '.png', dpi = 400)

# Save the Model
def Save_Model(model, PATH, using_state_dic = True):
    if using_state_dic:
        torch.save(model.state_dict(), PATH)
    else:
        torch.save(model, PATH)

# Load the Model
def Load_Model(model, PATH, using_state_dic = True):
    if using_state_dic:
        model.load_state_dict(torch.load(PATH))
    else:
        model = torch.load(PATH)
    # Eval mode is turned on by default
    model.eval()
# Checkpoint Model
def Checkpoint_Save_Training_(epoch, model, optim, loss, PATH):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss}, PATH)

def Checkpoint_Load_Training_(epoch, model, optim, loss, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.train()
    return epoch, loss
