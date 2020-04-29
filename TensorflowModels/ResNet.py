import os, cProfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
#from tensorflow.nn import conv2d, max_pool2d, batch_normalization, relu, softmax, avg_pool2d
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D, GaussianNoise
from tensorflow.keras.layers import Layer, add
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean, SparseTopKCategoricalAccuracy
from tensorflow.keras.utils import to_categorical

#In tensorflow 2.0, eager execution is turned on by default
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
#Change all floating point precision to float32
tf.keras.backend.set_floatx('float32')


class toyCNN(Layer):
  def __init__(self):
    super(toyCNN, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

class ResNetLayer(Layer):
    def __init__(self, filter_depth = 64, 
                       kernel_size = 3, 
                       strides = 1, 
                       batch_normalization = True, 
                       conv_first = True,
                       activation = 'relu'):
        super(ResNetLayer, self).__init__()
        self.conv = Conv2D(filters = filter_depth,
                           kernel_size = kernel_size,
                           strides = strides,
                           padding = 'same',
                           activation = None,
                           kernel_regularizer = l2(5e-4),
                           bias_regularizer = None,
                           kernel_initializer = 'he_normal')
        self.bn = BatchNormalization(axis = -1,
                                     momentum = 0.99,
                                     epsilon = 0.001,
                                     beta_initializer = 'zeros',
                                     gamma_initializer = 'ones')
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.conv_first = conv_first
    def call(self, input):
        x = input
        if self.conv_first:
            x = self.conv(x)
            if self.batch_normalization:     x = self.bn(x)
            if self.activation is not None:  x = Activation(self.activation)(x)
        else:
            if self.batch_normalization:     x = self.bn(x)
            if self.activation is not None:  x = Activation(self.activation)(x)
            x = self.conv(x)
        return x

class ResNetBlock(Model):
    def __init__(self, filter_depth, strides, is_bottleNeck = False):
        super(ResNetBlock, self).__init__(name = '')
        self.is_bottleNeck = is_bottleNeck
        if not is_bottleNeck:
            self.identity = ResNetLayer(filter_depth = filter_depth, 
                                        kernel_size = 1,
                                        strides = strides, 
                                        batch_normalization = False,
                                        activation = None)
            self.conv1 = ResNetLayer(filter_depth = filter_depth,
                                     strides = strides)
            self.conv2 = ResNetLayer(filter_depth = filter_depth,
                                     activation = None)
        else:
            self.identity = ResNetLayer(filter_depth = filter_depth,
                                        kernel_size = 1,
                                        strides = 4 * strides,
                                        activation = None,
                                        batch_normalization = False)
            self.conv1 = ResNetLayer(filter_depth = filter_depth,
                                     kernel_size = 1,
                                     strides = strides)
            self.conv2 = ResNetLayer(filter_depth = filter_depth)
            self.conv3 = ResNetLayer(filter_depth = 4 * filter_depth,
                                     kernel_size = 1,
                                     activation = None)
    def call(self, input):
        if not self.is_bottleNeck:
            shortcut = self.identity(input)
            input = self.conv1(input)
            input = self.conv2(input)
            input = add([shortcut, input])
            input = Activation('relu')(input)
        if self.is_bottleNeck:
            shortcut = self.identity(input)
            input = self.conv1(input)
            input = self.conv2(input)
            input = self.conv3(input)
            input = add([shortcut, input])
            input = Activation('relu')(input)
        return input

class MyResNet(Model):
    def __init__(self, model_type = 'ResNet18v1', num_classes = 100, std_of_noise = 0):
        super(MyResNet, self).__init__(name = '')
        self.ResBlock_Recurrence(model_type)
        self.std_of_noise = std_of_noise
        self.num_classes = num_classes
        self.conv1 = ResNetLayer(filter_depth = 64,
                                 kernel_size = 7,
                                 strides = 1)
        self.maxpool = MaxPooling2D(pool_size = 3,
                                    strides = 2,
                                    padding = 'same')
        self.avgpool = AveragePooling2D(pool_size = 2,
                                        strides = 1,
                                        padding = 'same')
        self.flatten = Flatten
        self.fc = Dense(num_classes,
                        activation = 'softmax',
                        kernel_initializer = 'he_normal',
                        bias_initializer = None,
                        kernel_regularizer = None,
                        bias_regularizer = None)
        self.ResBlockList = []
        filter_depth = 64
        for block_num in self.recurList:
            self.ResBlockList.append([])
            for iter in range(block_num):
                strides = 1
                if iter == 0 and filter_depth != 64:
                    strides = 2
                self.ResBlockList[-1].append(ResNetBlock(filter_depth = filter_depth, 
                                                         strides = strides,
                                                         is_bottleNeck = self.has_bottleNeck))
                filter_depth *= 2

    def ResBlock_Recurrence(self, model_type):
        if model_type == 'ResNet18v1':
            self.recurList, self.has_bottleNeck = [2, 2, 2, 2], False
        elif model_type == 'ResNet34v1':
            self.recurList, self.has_bottleNeck = [3, 4, 6, 3], False
        elif model_type == 'ResNet50v1':
            self.recurList, self.has_bottleNeck = [3, 4, 6, 3], True
        elif model_type == 'ResNet101v1':
            self.recurList, self.has_bottleNeck = [3, 4, 23, 3], True
        elif model_type == 'ResNet152v1':
            self.recurList, self.has_bottleNeck = [3, 8, 36, 3], True
        else:
            raise Exception('We do not support the Version of ResNet specified')
    def call(self, input):
        #tf.reshape(input, [-1, 32, 32, 3])
        x = GaussianNoise(self.std_of_noise)(input)
        x = self.conv1(x)
        x = self.maxpool(x)
        for convx_x in self.ResBlockList:
            for layer in convx_x:
                x = layer(x)
        x = self.avgpool(x)
        x = self.flatten()(x)
        outputs = self.fc(x)
        return outputs

class ResNet(Model):
    def __init__(self, total_epoch = 200, frameworkTest = True):
        super(ResNet, self).__init__(name = '')
        self.max_epoch = total_epoch
        self.dataPreparation()
        self.modelInstantiate(frameworkTest)
        self.training_objects()

    def dataPreparation(self, dataset_name = 'cifar10', 
                              staticPreprocessing = True,
                              batch_size = 32):
        #Need further update to address the problem of difference dimensions
        if dataset_name == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            num_classes = 10
        elif dataset_name == 'fasion_mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            num_classes = 10
        elif dataset_name == 'cifar10':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            num_classes = 10
        elif dataset_name == 'cifar100':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
            num_classes = 100
        x_train, x_test = x_train.astype('float64'), x_test.astype('float64')
        #y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)
        if staticPreprocessing == True:
            x_train, x_test = x_train / 255.0, x_test / 255.0
            x_train_mean = np.mean(x_train, axis=0)
            x_train -= x_train_mean
            x_test -= x_train_mean
        self.dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size).repeat()
        self.testset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        print('Successfully Loaded Dataset:', dataset_name, 'With:') #With Dimension
        print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
        print(x_train.shape[0], 'train samples', x_test.shape[0], 'test samples')

    def modelInstantiate(self, frameworkTest = True):
        #Toy Model For Debugging
        if frameworkTest == True:
            self.myModel = toyCNN()
        elif frameworkTest == False:
            self.myModel = MyResNet(model_type = 'ResNet18v1',
                                    num_classes = 10,
                                    std_of_noise = 0)
        print('Model instantiated, are using: ')
    
    def training_objects(self):
        self.loss = SparseCategoricalCrossentropy(from_logits = True)
        #Adam Optimizer is best for fixed learning rates
        self.optimizer = Adam(learning_rate = 0.001,
                         beta_1 = 0.9, 
                         beta_2 = 0.999,
                         epsilon = 1e-7,
                         amsgrad = False)
        #SGD
        #self.optimizer = SGD(learning_rate = 0.01,
        #                     momentum = 0.0,
        #                     nesterov = False)
        self.train_loss = Mean(name = 'train_loss')
        self.test_loss = Mean(name = 'test_loss')
        self.train_accuracy = SparseCategoricalAccuracy(name = 'train_accuracy')
        self.test_accuracy = SparseCategoricalAccuracy(name = 'test_accuracy')
        self.train_k_accuracy = SparseTopKCategoricalAccuracy(name = 'train_k_accuracy', k = 5)
        self.test_k_accuracy = SparseTopKCategoricalAccuracy(name = 'test_k_accuracy', k = 5)
        print('Training objects instantiated.')

    def lr_schedule(self, epoch):
        lr_init = 1e-3
        #Cosine Decay / Linear Warmup
        if epoch > 10:
            lr = lr_init * (1 + np.cos(3.1415 * epoch / self.max_epoch)) / 2
        else:
            lr = lr_init * epoch / 10
        return lr

    #Train Step
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.myModel(images)
            loss = self.loss(labels, predictions)
        gradients= tape.gradient(loss, self.myModel.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.myModel.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        self.train_k_accuracy(labels, predictions)
    #Test Step
    @tf.function
    def test_step(self, images, labels):
        predictions = self.myModel(images)
        t_loss = self.loss(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        self.test_k_accuracy(labels, predictions)

    def training_body(self):
        print('Initialization Completed, Starting Training')
        for epoch in range(self.max_epoch):
            training_instance_indicator = 0
            for images, labels in self.dataset:
                print(training_instance_indicator, '/50000', 'with epoch = ', epoch + 1, end = '\r')
                training_instance_indicator += 1
                self.train_step(images, labels)

            for test_images, test_labels in self.testset:
                self.test_step(test_images, test_labels)

            template = 'Epoch {}, Loss: {.5f}, Accuracy: {.5f}, Test Loss: {.5f}, Test Accuracy: {.5f} '
            template += 'Train_Topk_Accuracy: {.5f}, Test_Topk_Accuracy: {.5f}'
            print(template.format(epoch+1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result()*100,
                                  self.test_loss.result(),
                                  self.test_accuracy.result()*100,
                                  self.train_k_accuracy.result()*100,
                                  self.test_k_accuracy.result()*100))

            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.train_k_accuracy.reset_states()
            self.test_k_accuracy.reset_states()
        print('Training Complete.....')

model = ResNet(3, True)
model.training_body()