'''
    This is the Configure Options for Training Parameters
'''

Optimizer = {
    'SGD': {
            'optim_TYPE': 'SGD',
            'learning_rate': 0.004,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'nesterov': False
    },
    'Adams':{
            'optim_TYPE': 'Adams',
            'learning_rate': 0.01,
            'betas': (0.9, 0.999),
            'weight_decay': 0,
            'amsgrad': False
    },
    'Adagrad':{
            'optim_TYPE': 'Adagrad',
            'learning_rate': 0.001,
            'lr_decay': 0,
            'weight_decay': 0
    },
    'RMSprop':{
            'optim_TYPE': 'RMSprop',
            'learning_rate': 0.01,
            'alpha': 0.99,
            'weight_decay': 0,
            'momentum': 0
    }
}

lr_Scheduler = {
    'MultiStepLR':{
                    'schedule_TYPE': 'MultiStepLR',
                    'milestones': [150, 250],
                    'gamma': 0.1
    },
    'FixedLR':{
                    'schedule_TYPE': 'MultiStepLR',
                    'milestones': [500],
                    'gamma': 0.1
    },
    'ReduceLROnPlateau':{
                    'schedule_TYPE': 'ReduceLROnPlateau',
                    'mode': 'min',
                    'factor': 0.1,
                    'patience': 10,
                    'verbose': False,
                    'threshold': 1e-4,
                    'cooldown': 0,
                    'min_lr': 1e-6
    },
    'ExponentialLR':{
                    'schedule_TYPE': 'ExponentialLR',
                    'gamma': 0.95
    },
    'CosineAnnealingWarmRestarts':{
                    'schedule_TYPE': 'CosineAnnealingWarmRestarts',
                    'T_0': 10,
                    'T_mult': 1 
    }
}

data_Augment = {
    'RandomCrop':{
                    'size': 32,
                    'padding': 4
    },
    'RandomResizedCrop':{
                    'size': 32,
                    'scale': (0.08, 1.0),
                    'ratio': (0.75, 1.333333333333),
                    'interpolation': 2
    },
    'RandomAffine':{
                    'degrees': (0, 360),
                    'translate': (0.1, 0.1),
                    'scale': (0.5, 2),
                    'fillcolor': (0, 0, 0)
    },
    'ColorJitter':{
                    'brightness': 0.5,
                    'contrast': 0.5,
                    'saturation': 0.5,
                    'hue': 0.1,
    },
    'RandomHorizontalFlip':{
                    'p': 0.5
    }
}








