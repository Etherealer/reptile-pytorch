from torch.optim import SGD, Adam


def make(params, name='Adam', lr=0.005, weight_decay=0):
    if name == 'SGD':
        optimizer = SGD(params, lr, weight_decay=weight_decay)
    elif name == 'Adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay,  betas=(0, 0.999))
    else:
        raise ValueError('invalid optimizer')

    return optimizer
