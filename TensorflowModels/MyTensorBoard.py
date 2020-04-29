# This is a self-implemented utils library for Tensorboard in Pytorch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 
import numpy as np 

class MyTensorBoard():
    def __init__(self, net, LabelStr, EventDir):
        self.labelStr = labelStr
        self.writer = SummaryWriter(EventDir + '/')
        self.net = net

    def matplotlib_imshow(self, img, one_channel = True):
        if one_channel:
            img = img.mean(dim = 0)
        img = img / 2 + 0.5
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap = 'Greys')
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def ImageVisualize(self, images, labels):
        img_grid = torchvision.utils.make_grid(images)
        self.matplotlib_imshow(img_grid, one_channel = True)
        self.writer.add_image('Images', img_grid)
        self.writer.close()

    # Add Net structure to Tensorboard
    def NetVisualize(self, sampleInput):
        self.writer.add_graph(self.net, sampleInput)
        self.writer.close()

    def images_to_probs(self, images):
        '''
        Generates predictions and corresponding probabilities from a trained network
        and a list of images
        '''
        output = net(images)
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())
        return preds, [F.softmax(el, dim = 0)[i].item() for i, el in zip(preds, output)]

    def plot_classes_preds(self, images, labels):
        '''
            Generates matplotlib Figure using a trained network, along with images and labels
            from a batch, that shows the network's top predictions along with its probability,
            alongside the actual label, coloring this information based on whether the predictions
            was correct or not. Uses the "Images_to_probs" function
        '''
        preds, probs = images_to_probs(images)
        fig = plt.figure(figsize = (12, 48))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx + 1, xticks = [], yticks = [])
            matplotlib_imshow(images[idx], one_channel = True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(self.labelStr[preds[idx]],
                                                              probs[idx] * 100.0,
                                                              self.labelStr[labels[idx]]),
                                                              color = ("green" if preds[idx] == labels[idx].item() else "red"))
        return fig

    def ScalarVisualize(self, graphTitle, loss, currentStep):
        '''
            Log scalar values to plots, e.g. loss, acc, during training
        '''
        self.writer.add_scalar(graphTitle, loss, currentStep)
        self.writer.close()
    
    def PredVisualize(self, images, labels, currentStep):
        '''
            Log matplotlib figures of model's predictions to specified mini-batch
        '''
        self.writer.add_figure('predictions vs. actuals',
                               self.plot_classes_preds(images, labels),
                               global_step = currentStep)
        self.writer.close()

    def ProjVisualize(self, data, labels, use_rand_instance = True, num_rand = 100):
        '''
            Add projection visualization to Tensorboard
        '''
        assert len(data) == len(labels)
        if use_rand_instance:
            perm = torch.randperm(len(data))
            images, labels = data[perm][:num_rand], labels[perm][:num_rand]
        else:
            images, labels = data, labels
        class_labels = [self.labelStr[lab] for lab in labels]
        features = images.view(-1, 28 * 28)
        self.writer.add_embedding(features,
                                  metadata = class_labels,
                                  label_img = images.unsqueeze(1))
        self.writer.close()

    def PRcurveVisualize(self, test_probs, test_preds):
        '''
            Plot the Precision - Recall curve in Tensorboard, per - class wise
        '''
        for class_index in range(len(self.labelStr)):
            tensorboard_preds = test_preds == class_index
            tensorboard_probs = test_probs[:, class_index]
            self.writer.add_pr_curve(self.labelStr[class_index],
                                     tensorboard_preds,
                                     tensorboard_probs,
                                     global_step = 0)
        self.writer.close()
