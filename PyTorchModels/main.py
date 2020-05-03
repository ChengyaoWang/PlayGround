import numpy as np 
import torch
import torch.nn as nn 
import torchvision.transforms as transforms
# Other Library
from datetime import datetime
# Self-Defined Files
import Utils
import Config
import Recorder
# Models
import AlexNet, ResNet, LeNet5, MobileNet, NiNet
import SqueezeNet, myNet

# Recorder Instantiate
myRecorder = Recorder.Recorder()
torch.cuda.empty_cache()

# Fetch Models, Init & Copy
# model = ResNet.Fetch_ResNet('ResNet8v1')
# model = AlexNet.AlexNet()
# model = LeNet5.LeNet5()
# model = MobileNet.MobileNetv1()
# model = NiNet.NetworkInNetwork()
# model = SqueezeNet.Fetch_SqueezeNet('Bypass_Simple')
# model = SqueezeNet.SqueezeNet('Basic')
model = myNet.myNet(ResidualDepth = 2, DepthShrink = 0.5)



# print(model)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# raise ValueError('aaaaaa')


model.apply(Utils.weight_init)
myRecorder.add_record_('Model', model.model_name)
myRecorder.add_record_('Model Size Trainable', sum(p.numel() for p in model.parameters() if p.requires_grad))
myRecorder.add_record_('Model Size Total', sum(p.numel() for p in model.parameters()))
myRecorder.add_record_('Model Structure', [layer[2:] for layer in str(model).split('\n')[1:-1]])


# Define some Hyperparameters
batch_size = 128
total_epoch = 350
device = Utils.check_device()
model.to(device)
myRecorder.add_record_('Batch Size', batch_size)
myRecorder.add_record_('Total Epoch', total_epoch)
myRecorder.add_record_('Device', str(device))

# Loss Function
lossFunc = nn.CrossEntropyLoss()
myRecorder.add_record_('Loss Function', str(lossFunc)[:-2])
# Optimizer & Learning Rate Scheduler
optim_dict = Config.Optimizer['SGD']
scheduler_dict = Config.lr_Scheduler['MultiStepLR']
optim, scheduler = Utils.optim_init(model, optim_dict, scheduler_dict)
myRecorder.add_record_('Optimizer', {'Type': optim_dict['optim_TYPE'],
                                     'State': optim.state_dict()['state'],
                                     'param_groups': optim.state_dict()['param_groups'][0]})
myRecorder.add_record_('lr Scheduler', {'Type': scheduler_dict['schedule_TYPE'],
                                        'State': scheduler.state_dict()})

# DataSet Fetch & Augmentation
Train_transform = transforms.Compose([transforms.RandomCrop(32, padding = 4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
Test_transform = transforms.Compose([ transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
DataLoader, label = Utils.DatasetLoader('CIFAR10', Train_transform, Test_transform, batch_size, TrainShrink = 0.03125)
myRecorder.add_record_('TrainSet Size', 0.03125)
myRecorder.add_record_('Train Transform', [trans[4:] for trans in str(Train_transform).split('\n')[1:-1]])
myRecorder.add_record_('Test Transform', [trans[4:] for trans in str(Test_transform).split('\n')[1:-1]])
myRecorder.add_record_('Dataset', 'CIFAR10')


# Starts Training
# Record Start Time, this serves as time stamp for all the files stored
TimeStamp = str(datetime.now())
StartTime = datetime.now()
train_acc, test_acc, loss = Utils.train(model, DataLoader, lossFunc, optim, device, total_epoch, scheduler)
myRecorder.add_record_('TrainingTime', str(datetime.now() - StartTime))
StartTime = datetime.now()
class_prob, ConfuMx = Utils.fine_validate(model, DataLoader[1], device, label)
myRecorder.add_record_('InferenceTime', str(datetime.now() - StartTime))
Utils.visualize_plt('../save/' + TimeStamp, train_acc, test_acc, loss)
Utils.Save_Model(model, '../save/' + TimeStamp)
# Save Config into JSON
myRecorder.add_record_('Performance', { 'Best_Train': max(train_acc), 'Final_Train': train_acc[-1],
                                        'Best_Test' : max(test_acc),  'Final_Test': test_acc[-1],
                                        'Best_Loss':  max(loss), 'Final_Loss': loss[-1]})
myRecorder.add_record_('Class Performance', {label[i]: class_prob[i] for i in range(len(label))})
myRecorder.add_record_('Confusion Matrix',  {label[i]: str(ConfuMx[i]) for i in range(len(label))})

# Clean Up
myRecorder.json_dump_('../save/' + TimeStamp)
torch.cuda.empty_cache()
